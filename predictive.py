import pickle
import numpy as np
from scipy.stats import truncnorm, invgamma

from simulation import (
    initialize_simulation_params,
    create_neighbour_locs,
    simulate_advection,
    simulate_state,
    simulate_observations,
    get_mcmc_initializations
)
from sampler.gibbs import GibbsSampler
from evaluation.evaluate_metrics import compute_spatial_validation_metrics, compute_mspe, compute_rmspe, compute_mape, compute_crps

def sample_prior_predictive(params: dict, n_samples: int = 10, fixed_sigmas: bool = False) -> np.ndarray:
    """
    Generate samples from the prior predictive distribution.

    For each predictive sample:
      1. Sample model parameters from their prior distributions.
      2. Generate advection parameters using an AR(1) process.
      3. Simulate the latent state process Y using the advection-diffusion evolution.
      4. Generate observations Z by adding observation noise.

    The resulting samples are stored in an array of shape (N, n_samples, T).

    Parameters:
        params (dict): The simulation parameters including prior hyperparameters.
        n_samples (int): Number of predictive samples to generate.

    Returns:
        np.ndarray: An array of generated predictive observations Z 
                    with shape (N, n_samples, T)
    """
    # Set up spatial and temporal dimensions
    neighbour_locs = create_neighbour_locs(params['grid_size_x'], params['grid_size_y'])
    grid_shape = params['grid_shape']
    N = params['N']
    T = params['time_steps']
    # Allocate array with shape (N, n_samples, T)
    predictive_Z = np.zeros((N, n_samples, T))

    # Prior hyperparameters  
    prior = params['prior_params']

    for i in range(n_samples):
        # --- Sample parameters from their priors ---

        # Sample α ~ TruncatedNormal(m_α, v_α) on (0, 1)
        m_alpha = prior['autoregression']['m_alpha']
        v_alpha = prior['autoregression']['v_alpha']
        a_alpha = (0 - m_alpha) / np.sqrt(v_alpha)
        b_alpha = (1 - m_alpha) / np.sqrt(v_alpha)
        alpha = truncnorm.rvs(a=a_alpha, b=b_alpha, loc=m_alpha, scale=np.sqrt(v_alpha))

        # Sample β ~ N(m_β, v_β)
        m_beta = prior['diffusion']['m_beta']
        v_beta = prior['diffusion']['v_beta']
        beta = np.random.normal(m_beta, np.sqrt(v_beta))


        if fixed_sigmas:
            sigma_eta_sq = prior['process']['sigma_eta_sq']
            sigma_epsilon_sq = prior['observation']['sigma_epsilon_sq']
        
        else:
            # Sample σ²_η ~ Inverse-Gamma(a_η, b_η)
            a_eta = prior['process']['a_eta']
            b_eta = prior['process']['b_eta']
            sigma_eta_sq = invgamma.rvs(a_eta, scale=b_eta)

            # Sample σ²_ε ~ Inverse-Gamma(a_ε, b_ε)
            a_epsilon = prior['observation']['a_epsilon']
            b_epsilon = prior['observation']['b_epsilon']
            sigma_epsilon_sq = invgamma.rvs(a_epsilon, scale=b_epsilon)

        # Sample σ²_ν ~ Inverse-Gamma(a_ν, b_ν)
        a_nu = prior['advection']['a_nu']
        b_nu = prior['advection']['b_nu']
        sigma_nu_sq = invgamma.rvs(a_nu, scale=b_nu)

        # Sample the initial advection parameter 
        # ν₀ ~ 𝒩([mean_nu_x_zero, mean_nu_y_zero], var_nu_zero * I₂)
        mean_nu = np.array([prior['advection']['mean_nu_x_zero'], prior['advection']['mean_nu_y_zero']])
        var_nu = prior['advection']['var_nu_zero']
        nu0 = np.random.multivariate_normal(mean_nu, var_nu * np.eye(2))

        # Sample the initial state Y₀ ~ 𝒩(m_state, σ²_Y I)
        m_state = prior['initial_state']['m_state']
        v_state = prior['initial_state']['v_state']
        Y0 = np.random.normal(m_state, np.sqrt(v_state), N)

        # --- Simulate the process with the sampled parameters ---
        # Simulate advection parameters (ν_t) for t = 0,...,T using AR(1)
        nu = simulate_advection(alpha, sigma_nu_sq, T, nu0)
        # Simulate the latent state evolution Y for t = 1,...,T
        # (Note: simulate_state returns an array of shape (N, T))
        state = simulate_state(Y0, nu, beta, sigma_eta_sq, grid_shape, neighbour_locs)
        # Simulate the observed data Z using the observation model:
        # Z_t ~ 𝒩(Y_t, σ²_ε I)
        Z = simulate_observations(state, sigma_epsilon_sq)

        # Save the predictive sample such that the output array is (N, n_samples, T)
        predictive_Z[:, i, :] = Z

    return predictive_Z

def sample_posterior_predictive(posterior_samples: dict) -> np.ndarray:
    """
    Generate samples from the posterior predictive distribution.

    For each saved MCMC sample (which includes a latent state realization Y and the 
    observation error variance σ²_ε), a predictive observation is generated as:
        Z ~ 𝒩(Y, σ²_ε I)

    **Note:** It is assumed that the latent state samples `Y_samples` from the Gibbs 
    sampler include the initial state, i.e., they have shape (N, T+1, num_samples). 
    This function drops the initial state and returns an array with shape (N, num_samples, T).

    Parameters:
        posterior_samples (dict): Dictionary with posterior samples. Expected keys:
            - 'Y_samples': array of shape (N, T+1, num_samples)
            - 'sigma_epsilon_sq_samples': array of shape (num_samples,)

    Returns:
        np.ndarray: Posterior predictive samples Z with shape (N, num_samples, T)
    """
    # Y_samples includes initial state, so its shape is (N, T+1, num_samples)
    Y_samples = posterior_samples['Y_samples']
    sigma_epsilon_sq_samples = posterior_samples['sigma_epsilon_sq_samples']

    N, T_plus_1, S = Y_samples.shape
    # Exclude the initial state; T is T+1 - 1.
    T = T_plus_1 - 1
    # Allocate array with shape (N, num_samples, T)
    Z_ppd = np.zeros((N, S, T))

    for s in range(S):
        # Exclude the initial state (index 0); use t = 1,...,T.
        Y = Y_samples[:, 1:, s]
        sigma_eps = sigma_epsilon_sq_samples[s]
        noise = np.random.normal(0, np.sqrt(sigma_eps), (N, T))
        Z_ppd[:, s, :] = Y + noise

    return Z_ppd

def main():
    """
    Demonstrates the use of the predictive sampling functions.
    """
    # Initialize simulation parameters and priors
    params = initialize_simulation_params()

    # -------------------------
    # Prior Predictive Sampling
    # -------------------------
    n_prior_samples = 400
    print("Generating Prior Predictive Samples...")
    params['prior_params']['process']['sigma_eta_sq'] = 0.09178419788419002
    params['prior_params']['observation']['sigma_epsilon_sq'] = 0.09178419788419002
    prior_Z = sample_prior_predictive(params, n_samples=n_prior_samples, fixed_sigmas=True)
    print(f"Prior predictive samples shape: {prior_Z.shape}")  # Expected: (N, n_prior_samples, T)

    # ----------------------------
    # Posterior Predictive Sampling
    # ----------------------------
    neighbour_locs = create_neighbour_locs(params['grid_size_x'], params['grid_size_y'])
    nu = simulate_advection(params['alpha'], params['sigma_nu_sq'],
                            params['time_steps'], params['nu0'])
    state = simulate_state(params['initial_state'], nu, params['beta'], 
                           params['sigma_eta_sq'], params['grid_shape'], neighbour_locs)
    # Simulate observations using the observation model
    observations = simulate_observations(state, params['sigma_epsilon_sq'])

    with open('samples_fixed_sigmas_2000_1200_2.pkl', 'rb') as f:
        posterior_samples = pickle.load(f)

    print(posterior_samples['sigma_eta_sq_samples'][1])

    post_pred_Z = sample_posterior_predictive(posterior_samples)
    print(f"Posterior predictive samples shape: {post_pred_Z.shape}")  # Expected: (N, num_saved_samples, T)

    # Evaluate the predictive performance of the posterior predictive samples
    # Compute the mean of the predictive samples
    Z_mean = np.mean(post_pred_Z, axis=1)  # shape: (N, T)

    # Compute the spatial validation metrics
    # We need to reshape post_pred_Z to match the expected input shape (N, num_samples, T)
    post_pred_Z_reshaped = np.transpose(post_pred_Z, (0, 1, 2))  # Ensure correct shape
    V_bias, V_normRMSE, V3 = compute_spatial_validation_metrics(observations, post_pred_Z_reshaped)
    print("\nSpatial validation metrics:")
    print(f"Mean V_bias: {np.nanmean(V_bias):.4f}")
    print(f"Mean V_normRMSE: {np.nanmean(V_normRMSE):.4f}")
    print(f"Mean V3: {np.mean(V3):.4f}")

    # Compute the overall metrics
    mspe_val = compute_mspe(observations, Z_mean)
    rmspe_val = compute_rmspe(observations, Z_mean)
    mape_val = compute_mape(observations, Z_mean)
    crps_val = compute_crps(observations, post_pred_Z_reshaped)

    print("\nOverall metrics:")
    print(f"MSPE: {mspe_val:.4f}")
    print(f"RMSPE: {rmspe_val:.4f}")
    print(f"MAPE: {mape_val:.4f}")
    print(f"CRPS: {crps_val:.4f}")

if __name__ == '__main__':
    main() 