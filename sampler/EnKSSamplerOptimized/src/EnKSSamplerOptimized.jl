module EnKSSamplerOptimized

using MKL
using LinearAlgebra, Random, Statistics, Distributions
using Base.Threads
using ProgressMeter
using StaticArrays
using SIMD
using BenchmarkTools

export run, benchmark

# ---------------------------------------------------------------------
# Helper Function: Apply Inverse Covariance via Woodbury Identity
#
# Instead of forming the full inverse covariance matrix, this function 
# computes the product:
#
#   result .= innovation/ε - (1/ε)*Z*(F \ (Z'*(innovation/ε)))
#
# where F is the Cholesky factorization of 
#   M = (Ne-1)*I + (1/ε)*(Z' * Z)
#
# This avoids the expensive inversion of a large matrix.
# ---------------------------------------------------------------------
function apply_inverse_covariance_Z!(result::AbstractMatrix{Float64}, 
                                       Z::AbstractMatrix{Float64},
                                       innovation::AbstractMatrix{Float64}; 
                                       epsilon::Float64=1e-5)
    # Compute temp = innovation/ε
    temp = innovation ./ epsilon
    # Compute intermediate product: W = Z' * temp.
    W = Z' * temp
    Ne = size(Z, 2)
    # Build the small ensemble-space matrix:
    M = (Ne - 1) * I + (1/epsilon) * (Z' * Z)
    M = Symmetric(M + 1e-8 * I)  # ensures positive definiteness
    # Factorize using Cholesky.
    F = cholesky(M)
    # Solve for correction: correction = Z*(F \ W)
    correction = Z * (F \ W)
    # Final result: result = temp - correction/ε.
    result .= temp .- correction ./ epsilon
    return result
end

# ---------------------------------------------------------------------
# Precompute Neighbor Indices for Propagation
#
# Converts the provided neighbor indices from 0-based to 1-based indexing.
# The expected order in the input is: [idx, left, right, up, down].
# ---------------------------------------------------------------------
function precompute_neighbors(neighbour_locs::Matrix{Int})
    N = size(neighbour_locs, 1)
    neighbors = Vector{NTuple{5, Int}}(undef, N)
    @inbounds for i in 1:N
        neighbors[i] = (neighbour_locs[i, 1] + 1,
                        neighbour_locs[i, 2] + 1,
                        neighbour_locs[i, 3] + 1,
                        neighbour_locs[i, 4] + 1,
                        neighbour_locs[i, 5] + 1)
    end
    return neighbors
end

# ---------------------------------------------------------------------
# Propagation Step: Forecasting the Ensemble
#
# Propagates the state by updating each location using its neighbors.
# The expected order of neighbor indices (after shifting) is:
#   (center, left, right, up, down)
#
# The update is:
#
#   new_state[i,j] = (1-4β)*Y_prev[center,j] +
#                    (β+vₓ)*Y_prev[left,j] +
#                    (β-vₓ)*Y_prev[right,j] +
#                    (β+v_y)*Y_prev[up,j] +
#                    (β-v_y)*Y_prev[down,j] + noise[i,j]
#
# ---------------------------------------------------------------------
function propagate(Y_prev::Matrix{Float64}, neighbors::Vector{NTuple{5, Int}},
                   beta::Float64, nu_t::AbstractVector{Float64}, 
                   sigma_eta_sq::Float64)
    N, N_ensemble = size(Y_prev)
    new_state = similar(Y_prev)
    v_x, v_y = nu_t[1], nu_t[2]
    center_coef = 1 - 4 * beta
    left_coef   = beta + v_x
    right_coef  = beta - v_x
    up_coef     = beta + v_y    # corrected: use plus v_y for up
    down_coef   = beta - v_y    # corrected: use minus v_y for down
    # Generate noise once for all points.
    noise = randn(N, N_ensemble) .* sqrt(sigma_eta_sq)
    
    @threads for i in 1:N
        # Unpack neighbor indices in the proper order.
        center, left, right, up, down = neighbors[i]
        @inbounds @simd for j in 1:N_ensemble
            new_state[i, j] = center_coef * Y_prev[center, j] +
                              left_coef   * Y_prev[left, j] +
                              right_coef  * Y_prev[right, j] +
                              up_coef     * Y_prev[up, j] +
                              down_coef   * Y_prev[down, j] +
                              noise[i, j]
        end
    end
    return new_state
end

# ---------------------------------------------------------------------
# Update Step: Smoothing Update Over a Lag Window
#
# Uses the efficient covariance application to update ensemble states.
# ---------------------------------------------------------------------
function update_step!(Y_analysis::Array{Float64,3}, observations::Matrix{Float64},
                      t::Int, lags::Int, N_ensemble::Int, 
                      sigma_epsilon_sq::Float64; epsilon::Float64=1e-5)
    N = size(Y_analysis, 1)
    # Generate observation noise for the update step.
    noise_matrix = randn(N, N_ensemble) .* sqrt(sigma_epsilon_sq)
    Z_tilde = Y_analysis[:, :, t] .+ noise_matrix
    # Extract the observation vector (note: observations use t-1 indexing).
    obs_col = @view observations[:, t-1]
    # Compute innovation (broadcast subtraction yields a matrix).
    innovation = obs_col .- Z_tilde
    precomputed_term = similar(Z_tilde)
    # Apply the efficient inverse covariance operation in-place.
    apply_inverse_covariance_Z!(precomputed_term, Z_tilde, innovation; epsilon=epsilon)
    
    # Compute anomalies for the pseudo-observations.
    m_Z_tilde = mean(Z_tilde, dims=2)
    Z_anom = Z_tilde .- m_Z_tilde
    ZaT_precomp = Z_anom' * precomputed_term
    factor = 1.0 / (N_ensemble - 1)
    
    l_start = max(1, t - lags)
    @threads for l in l_start:t
        Y_l = @view Y_analysis[:, :, l]
        m_Y_l = mean(Y_l, dims=2)
        Y_anom = Y_l .- m_Y_l
        update = (Y_anom * ZaT_precomp) * factor
        Y_l .+= update
    end
    
    return Z_tilde
end

# ---------------------------------------------------------------------
# Main EnKS Function
#
# Runs the Ensemble Kalman Smoother with the specified parameters.
# Returns a 3D array of shape (N, N_ensemble, T_obs+1) where the first slice 
# is the initial state.
#
# The initial state is generated as: Normal(m_state, sqrt(v_state))
# ---------------------------------------------------------------------
function run(observations::Matrix{Float64}, neighbour_locs::Matrix{Int},
             N_ensemble::Int, lags::Int, beta::Float64, nu::Matrix{Float64},
             sigma_eta_sq::Float64, sigma_epsilon_sq::Float64, 
             initial_state::AbstractMatrix{Float64}; epsilon::Float64=1e-5) :: Array{Float64,3}

    N, T_obs = size(observations)
    Y_analysis = Array{Float64,3}(undef, N, N_ensemble, T_obs + 1)

    # Initialize the ensemble at time t=1 using the provided initial_state.
    # It is assumed that initial_state is a matrix of shape (N, N_ensemble)
    Y_analysis[:, :, 1] .= initial_state

    # Precompute neighbor indices for faster propagation.
    neighbors = precompute_neighbors(neighbour_locs)
    
    for t in 2:(T_obs + 1)
        nu_t = @view nu[t-1, :]
        Y_analysis[:, :, t] .= propagate(Y_analysis[:, :, t-1], neighbors, beta, nu_t, sigma_eta_sq)
        update_step!(Y_analysis, observations, t, lags, N_ensemble, sigma_epsilon_sq; epsilon=epsilon)
    end
    return Y_analysis
end


# ---------------------------------------------------------------------
# Benchmark Function
# ---------------------------------------------------------------------
function benchmark(observations::Matrix{Float64}, neighbour_locs::Matrix{Int};
                   N_ensemble::Int=100, lags::Int=5, beta::Float64=0.1, 
                   nu::Matrix{Float64}=zeros(size(observations,2),2),
                   sigma_eta_sq::Float64=0.1, sigma_epsilon_sq::Float64=0.1,
                   m_state::Float64=0.0, v_state::Float64=1.0,
                   epsilon::Float64=1e-5)
    @info "Benchmarking EnKS with $(Threads.nthreads()) threads"
    @time result = run(observations, neighbour_locs, N_ensemble, lags, beta, 
                      nu, sigma_eta_sq, sigma_epsilon_sq, 
                      m_state, v_state; epsilon=epsilon)
    return result
end

end # module EnKSSamplerOptimized
