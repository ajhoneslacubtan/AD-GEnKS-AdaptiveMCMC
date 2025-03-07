"""
predictive_simulation.py

This module implements functions for generating predictive distributions via composition sampling.
It provides simulation functions that generate the latent process and observations without burn‑in
or missing observations. In addition, it provides functions to generate both the posterior predictive
distribution (using MCMC samples) and the prior predictive distribution (by sampling parameters from the
prior).

The observation model is:
    Z[t] = Y[t] + εₜ,   where εₜ ~ N(0, sigma_epsilon_sq).

Usage:
    - Use generate_posterior_predictive(Y_samples, sigma_epsilon_sq_samples) when you have MCMC samples.
    - Use generate_prior_predictive(num_samples, simulation_params, prior_params, neighbour_locs) to
      generate predictive samples from the prior.
"""

import numpy as np
from scipy.stats import truncnorm, invgamma

def simulate_advection_pred(alpha: float, sigma_nu_sq: float, T_desired: int) -> np.ndarray:
    """
    Simulate advection parameters over T_desired time steps (without burn‑in).
    
    The process is given by:
        nu[t] = alpha * nu[t-1] + ωₜ,
    where ωₜ ~ N(0, sigma_nu_sq * I).
    
    Parameters:
        alpha (float): Autoregressive coefficient.
        sigma_nu_sq (float): Variance for the advection noise.
        T_desired (int): Desired number of time steps (after time 0).
    
    Returns:
        np.ndarray: Array of shape (T_desired+1, 2) with simulated advection values
                    from time 0 to T_desired.
    """
    nu = np.zeros((T_desired + 1, 2))
    nu[0, :] = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_nu_sq * np.eye(2))
    for t in range(1, T_desired + 1):
        omega_t = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_nu_sq * np.eye(2))
        nu[t, :] = alpha * nu[t - 1, :] + omega_t
    return nu

def simulate_state_pred(initial_state: np.ndarray, nu: np.ndarray, beta: float, 
                        sigma_eta_sq: float, grid_shape: tuple, neighbour_locs: np.ndarray) -> np.ndarray:
    """
    Simulate latent states on a grid without burn‑in.
    
    The process is given by:
    
        Y[i, t] = (1 - 4*beta)*Y[i, t-1] 
                  + (beta - nu_x[t])*Y[right, t-1] 
                  + (beta + nu_x[t])*Y[left, t-1]
                  + (beta - nu_y[t])*Y[down, t-1]
                  + (beta + nu_y[t])*Y[up, t-1]
                  + εₜ,
                  
    where εₜ ~ N(0, sigma_eta_sq).
    
    Parameters:
        initial_state (np.ndarray): Array of shape (N,) representing the initial state (time 0).
        nu (np.ndarray): Advection parameters of shape (T_desired+1, 2).
        beta (float): Diffusion coefficient.
        sigma_eta_sq (float): Process error variance.
        grid_shape (tuple): Grid dimensions as (grid_size_x, grid_size_y).
        neighbour_locs (np.ndarray): Array of shape (N, 5) with neighbor indices.
        
    Returns:
        np.ndarray: Latent state trajectory of shape (N, T_desired+1) from time 0 to T_desired.
    """
    N = initial_state.shape[0]
    T = nu.shape[0] - 1  # since nu includes time 0
    state = np.zeros((N, T + 1))
    state[:, 0] = initial_state
    
    left_neighbors = neighbour_locs[:, 1].astype(int)
    right_neighbors = neighbour_locs[:, 2].astype(int)
    up_neighbors = neighbour_locs[:, 3].astype(int)
    down_neighbors = neighbour_locs[:, 4].astype(int)
    
    for t in range(1, T + 1):
        nu_x = nu[t, 0]
        nu_y = nu[t, 1]
        prev_state = state[:, t - 1]
        state[:, t] = ((1 - 4 * beta) * prev_state +
                       (beta - nu_x) * prev_state[right_neighbors] +
                       (beta + nu_x) * prev_state[left_neighbors] +
                       (beta - nu_y) * prev_state[down_neighbors] +
                       (beta + nu_y) * prev_state[up_neighbors] +
                       np.random.normal(0, np.sqrt(sigma_eta_sq), N))
    return state

def simulate_observations_pred(state: np.ndarray, sigma_epsilon_sq: float) -> np.ndarray:
    """
    Generate predictive observations by adding Gaussian noise to the latent states.
    
    The observation model is:
        Z[t] = Y[t] + εₜ,   where εₜ ~ N(0, sigma_epsilon_sq).
    
    Parameters:
        state (np.ndarray): Latent state trajectory of shape (N, T+1).
        sigma_epsilon_sq (float): Observation error variance.
    
    Returns:
        np.ndarray: Predictive observations of shape (N, T), where the initial state is discarded.
    """
    # Discard the initial state (time 0) since there are no observations at time 0.
    state_no0 = state[:, 1:]
    N, T = state_no0.shape
    observations = np.zeros((N, T))
    for t in range(T):
        observations[:, t] = state_no0[:, t] + np.random.normal(0, np.sqrt(sigma_epsilon_sq), N)
    return observations

def generate_posterior_predictive(Y_samples: np.ndarray, sigma_epsilon_sq_samples: np.ndarray) -> np.ndarray:
    """
    Generate posterior predictive samples via composition sampling.
    
    Parameters:
        Y_samples: np.ndarray of shape (N, T, M)
            Posterior samples of the latent state (e.g., from your Gibbs sampler), where
            N = number of spatial locations, T = number of time steps, and M = number of samples.
        sigma_epsilon_sq_samples: np.ndarray of shape (M,)
            Posterior samples of the observation error variance.
    
    Returns:
        np.ndarray of shape (N, T, M) containing predictive samples of the data.
    """
    Y_samples = Y_samples[:, 1:, :].copy()  # discard time 0
    N, T, M = Y_samples.shape
    ppd = np.zeros((N, T, M))
    for m in range(M):
        ppd[:, :, m] = np.random.normal(loc=Y_samples[:, :, m],
                                        scale=np.sqrt(sigma_epsilon_sq_samples[m]))
    return ppd # shape: (N, T, M)

def generate_prior_predictive(num_samples: int, simulation_params: dict, prior_params: dict, neighbour_locs: np.ndarray, fixed_sigma: bool = True) -> dict:
    """
    Generate prior predictive samples via composition sampling, along with the corresponding parameter samples.
    
    For each predictive sample, the function:
      1. Samples parameters from their prior distributions.
      2. Simulates the latent process Y using the predictive simulation functions.
      3. Generates observations using the observation model.
      
    Parameters:
        num_samples (int): Number of predictive samples to generate.
        simulation_params (dict): Dictionary with simulation settings. Expected keys include:
            - 'T_desired': number of time steps (excluding time 0).
            - 'grid_shape': tuple of grid dimensions.
            - 'N': total number of spatial locations.
        prior_params (dict): Dictionary of prior hyperparameters. Expected keys include:
            - 'autoregression': with keys 'm_alpha' and 'v_alpha'.
            - 'diffusion': with keys 'm_beta' and 'v_beta'.
            - 'process': with keys 'a_eta' and 'b_eta'.
            - 'observation': with keys 'a_epsilon' and 'b_epsilon'.
            - 'advection': with keys 'a_nu' and 'b_nu'.
            - 'initial_state': with keys 'm_state' and 'v_state'.
        neighbour_locs (np.ndarray): Array of neighbor indices for the grid.
    
    Returns:
        dict: Dictionary containing:
            - 'predictive_samples': np.ndarray of shape (N, T, num_samples) with predictive observations.
            - 'alpha_samples': np.ndarray of shape (num_samples,)
            - 'beta_samples': np.ndarray of shape (num_samples,)
            - 'sigma_eta_sq_samples': np.ndarray of shape (num_samples,)
            - 'sigma_epsilon_sq_samples': np.ndarray of shape (num_samples,)
            - 'sigma_nu_sq_samples': np.ndarray of shape (num_samples,)
    """
    predictive_samples_list = []
    alpha_samples = []
    beta_samples = []
    sigma_eta_sq_samples = []
    sigma_epsilon_sq_samples = []
    sigma_nu_sq_samples = []
    advection = np.zeros((simulation_params['time_steps'] + 1, 2, num_samples))
    states = np.zeros((simulation_params['N'], simulation_params['time_steps'] + 1, num_samples))
    
    T_desired = simulation_params['time_steps']
    N = simulation_params['N']
    grid_shape = simulation_params['grid_shape']
    
    for s in range(num_samples):
        # Sample parameters from their priors:
        m_alpha = prior_params['autoregression']['m_alpha']
        v_alpha = prior_params['autoregression']['v_alpha']
        a_alpha, b_alpha = (0 - m_alpha) / np.sqrt(v_alpha), (1 - m_alpha) / np.sqrt(v_alpha)
        alpha = truncnorm.rvs(a_alpha, b_alpha, loc=m_alpha, scale=np.sqrt(v_alpha))
        
        m_beta = prior_params['diffusion']['m_beta']
        v_beta = prior_params['diffusion']['v_beta']
        beta = np.random.normal(m_beta, np.sqrt(v_beta))
        
        if fixed_sigma:
            sigma_eta_sq = prior_params['process']['sigma_eta_sq']
            sigma_epsilon_sq = prior_params['observation']['sigma_epsilon_sq']
        else:
            sigma_eta_sq = invgamma.rvs(a=prior_params['process']['a_eta'], scale=prior_params['process']['b_eta'])
            sigma_epsilon_sq = invgamma.rvs(a=prior_params['observation']['a_epsilon'], scale=prior_params['observation']['b_epsilon'])

        sigma_nu_sq = invgamma.rvs(a=prior_params['advection']['a_nu'], scale=prior_params['advection']['b_nu'])
        
        # Save parameter samples
        alpha_samples.append(alpha)
        beta_samples.append(beta)
        sigma_eta_sq_samples.append(sigma_eta_sq)
        sigma_epsilon_sq_samples.append(sigma_epsilon_sq)
        sigma_nu_sq_samples.append(sigma_nu_sq)
        
        # Simulate the advection process (without burn-in):
        nu = simulate_advection_pred(alpha, sigma_nu_sq, T_desired)
        advection[:, :, s] = nu
        
        # Simulate the initial state:
        initial_state = np.random.normal(loc=prior_params['initial_state']['m_state'],
                                         scale=np.sqrt(prior_params['initial_state']['v_state']),
                                         size=N)
        
        # Simulate the latent process (without burn-in):
        state = simulate_state_pred(initial_state, nu, beta, sigma_eta_sq, grid_shape, neighbour_locs)
        states[:, :, s] = state
        
        # Generate predictive observations (discard time 0 so that shape is (N, T))
        observations = simulate_observations_pred(state, sigma_epsilon_sq)
        predictive_samples_list.append(observations)
    
    # Convert list of predictive samples into an array and reshape to (N, T, num_samples)
    predictive_samples = np.stack(predictive_samples_list, axis=-1)  # shape: (N, T, num_samples)
    
    return {
        'predictive_samples': predictive_samples, # shape: (N, T, num_samples)
        'alpha_samples': np.array(alpha_samples), # shape: (num_samples,)
        'beta_samples': np.array(beta_samples), # shape: (num_samples,)
        'sigma_eta_sq_samples': np.array(sigma_eta_sq_samples), # shape: (num_samples,)
        'sigma_epsilon_sq_samples': np.array(sigma_epsilon_sq_samples), # shape: (num_samples,)
        'sigma_nu_sq_samples': np.array(sigma_nu_sq_samples), # shape: (num_samples,)
        'nu_samples': np.transpose(advection, (1, 0, 2)), # shape: (2, T_desired+1, num_samples)
        'Y_samples': states # shape: (N, T_desired+1, num_samples)
    }