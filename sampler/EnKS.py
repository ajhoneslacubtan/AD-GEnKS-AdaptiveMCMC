import numpy as np
from numba import njit, prange
from tqdm import tqdm
from sampler.utils import inverse_covariance_Z
from julia.api import Julia

# Initialize the Julia runtime (disable compiled modules for compatibility)
jl = Julia(compiled_modules=False)

from julia import Pkg
# Activate the Julia project where EnKSSamplerOptimized is located.
Pkg.activate("sampler/EnKSSamplerOptimized")

# Import the Julia module
from julia import EnKSSamplerOptimized

class EnKS:
    def __init__(self, N_ensemble, lags, beta, nu, sigma_eta_sq, sigma_epsilon_sq, prior_params):
        self.N_ensemble = N_ensemble
        self.lags = lags
        self.beta = beta
        self.nu = nu
        self.sigma_eta_sq = sigma_eta_sq
        self.sigma_epsilon_sq = sigma_epsilon_sq
        self.prior_params = prior_params

    @staticmethod
    @njit(cache=True)
    def propagate(Y_prev, neighbour_locs, beta, nu_t, sigma_eta_sq):
        N, Ne = Y_prev.shape
        new_state = np.zeros((N, Ne))
        noise = np.random.normal(0, np.sqrt(sigma_eta_sq), (N, Ne))
        v_x, v_y = nu_t
        
        for loc in prange(N):
            idx, left, right, up, down = neighbour_locs[loc]
            new_state[loc] = (
                (1 - 4 * beta) * Y_prev[loc] +
                (beta + v_x) * Y_prev[left] +
                (beta - v_x) * Y_prev[right] +
                (beta + v_y) * Y_prev[up] +
                (beta - v_y) * Y_prev[down] +
                noise[loc]
            )
        return new_state

    def run(self, observations, neighbour_locs):
        N, T_obs = observations.shape
        Y_analysis = np.zeros((N, self.N_ensemble, T_obs + 1))
        Y_analysis[:, :, 0] = np.random.normal(
            self.prior_params['initial_state']['m_state'], 
            np.sqrt(self.prior_params['initial_state']['v_state']), 
            (N, self.N_ensemble)
        )

        for t in tqdm(range(1, T_obs + 1), desc="EnKS", unit="step", leave=False):
            # Forecast step
            Y_analysis[:, :, t] = self.propagate(
                Y_analysis[:, :, t-1], neighbour_locs, 
                self.beta, self.nu[t-1], self.sigma_eta_sq
            )

            # Generate pseudo-observations Z_tilde
            Z_tilde = Y_analysis[:, :, t] + np.random.normal(
                0, np.sqrt(self.sigma_epsilon_sq), (N, self.N_ensemble)
            )
            
            hat_Sigma_t_zz_inv = inverse_covariance_Z(Z_tilde)

            # Precompute terms for all l
            innovation = observations[:, t-1][:, None] - Z_tilde
            precomputed_term = hat_Sigma_t_zz_inv @ innovation  # (No x Ne)
            m_Z_tilde = np.mean(Z_tilde, axis=1, keepdims=True)
            Z_anom = Z_tilde - m_Z_tilde

            # Iterate over l indices to update
            l_start = max(0, t - self.lags)
            for l in range(l_start, t + 1):
                Y_l = Y_analysis[:, :, l].copy()  # Ensure contiguous array
                m_Y_l = np.mean(Y_l, axis=1, keepdims=True)
                Y_anom = Y_l - m_Y_l

                # Compute update efficiently
                term = Z_anom.T @ precomputed_term  # (Ne x Ne)
                update = (Y_anom @ term) / (self.N_ensemble - 1)
                Y_analysis[:, :, l] += update

        return Y_analysis

def EnKS_Optimized(observations: np.ndarray,
                    neighbour_locs: np.ndarray,
                    N_ensemble: int,
                    lags: int,
                    beta: float,
                    nu: np.ndarray,
                    sigma_eta_sq: float,
                    sigma_epsilon_sq: float,
                    m_state: float,
                    v_state: float,
                    epsilon: float = 1e-5) -> np.ndarray:
    """
    Python wrapper for the Julia EnKSSamplerOptimized.run function.
    
    Parameters:
        observations (np.ndarray): 2D array of shape (N, T_obs) containing observations.
        neighbour_locs (np.ndarray): 2D array of shape (N, 5) with neighbor indices (0-based).
        N_ensemble (int): Number of ensemble members.
        lags (int): Number of lags for the smoothing update.
        beta (float): Model parameter.
        nu (np.ndarray): 2D array of shape (T_obs, 2) with [v_x, v_y] per time step.
        sigma_eta_sq (float): Process noise variance.
        sigma_epsilon_sq (float): Observation noise variance.
        m_state (float): Mean state for initializing the ensemble.
        v_state (float): Variance state for initializing the ensemble.
        epsilon (float, optional): Regularization parameter. Default is 1e-5.
        
    Returns:
        np.ndarray: 3D array of shape (N, N_ensemble, T_obs+1) representing the analysis ensemble.
    """
    # Call the Julia run function. PyJulia automatically converts NumPy arrays to Julia arrays.
    result = EnKSSamplerOptimized.run(
        observations,
        neighbour_locs,
        N_ensemble,
        lags,
        beta,
        nu,
        sigma_eta_sq,
        sigma_epsilon_sq,
        m_state,
        v_state,
        epsilon=epsilon
    )
    
    # Convert the result to a NumPy array if necessary.
    return np.array(result)