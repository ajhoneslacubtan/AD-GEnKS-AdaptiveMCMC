"""
simulation.py

This module simulates a spatiotemporal process with latent states and 
advection parameters. It then generates observations by adding noise, 
and also provides initializations for a future Gibbs/MCMC sampler.

The latent state evolution on a regular grid (with periodic boundary conditions)
is written in a vectorized manner using np.roll. In this simulation:
 
    Y[i, t] = (1 - 4*beta)*Y[i, t-1] 
              + (beta - nu_x)*Y[right, t-1] 
              + (beta + nu_x)*Y[left, t-1]
              + (beta - nu_y)*Y[down, t-1]
              + (beta + nu_y)*Y[up, t-1]
              + ε, 
              
where ε ~ N(0, sigma_eta_sq).
"""

import numpy as np

def simulate_advection(alpha: float, sigma_nu_sq: float, T_desired: int, burn_in_fraction: float = 0.5) -> np.ndarray:
    """
    Simulate advection parameters using an autoregressive process over a time horizon that includes burn-in.
    
    The process is given by:
        nu[t] = alpha * nu[t-1] + ωₜ,
    where ωₜ ~ N(0, sigma_nu_sq * I).
    
    The simulation runs for total_steps = T_desired + burn_in, where 
        burn_in = int(burn_in_fraction * T_desired).
        
    Parameters:
        alpha (float): Autoregressive coefficient.
        sigma_nu_sq (float): Variance for the advection noise.
        T_desired (int): Number of desired time steps (after discarding burn-in).
        burn_in_fraction (float): Fraction of T_desired to use as burn-in.
    
    Returns:
        np.ndarray: Array of shape (total_steps+1, 2) with simulated advection values.
                   (total_steps = T_desired + burn_in)
    """
    burn_in = int(burn_in_fraction * T_desired)
    total_steps = T_desired + burn_in

    nu = np.zeros((total_steps + 1, 2))
    nu[0, :] = np.random.multivariate_normal(mean=np.zeros(2), cov= sigma_nu_sq * np.eye(2))
    for t in range(1, total_steps + 1):
        omega_t = np.random.multivariate_normal(mean=np.zeros(2), cov=sigma_nu_sq * np.eye(2))
        nu[t, :] = alpha * nu[t - 1, :] + omega_t
    return nu

def create_neighbour_locs(grid_size_x: int, grid_size_y: int) -> np.ndarray:
    """
    Create a matrix of neighbor locations with periodic boundary conditions.
    
    Parameters:
        grid_size_x (int): Number of grid points in x direction
        grid_size_y (int): Number of grid points in y direction
        
    Returns:
        np.ndarray: Array of shape (N, 5) containing [index, left, right, up, down] for each grid point
    """
    neighbour_locs = []
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            index = i * grid_size_y + j
            left = ((i - 1) % grid_size_x) * grid_size_y + j    # Left neighbor with wrap-around
            right = ((i + 1) % grid_size_x) * grid_size_y + j   # Right neighbor with wrap-around
            up = i * grid_size_y + (j - 1) % grid_size_y        # Up neighbor with wrap-around
            down = i * grid_size_y + (j + 1) % grid_size_y      # Down neighbor with wrap-around
            neighbour_locs.append([index, left, right, up, down])
    return np.array(neighbour_locs)

def simulate_state(sigma_eta_sq: float, nu: np.ndarray, beta: float, 
                   grid_shape: tuple, neighbour_locs: np.ndarray, 
                   T_desired: int, burn_in_fraction: float = 0.5) -> np.ndarray:
    """
    Simulate latent states on a grid with periodic boundary conditions, including a burn-in period.
    
    The initial state is generated from N(0, sigma_eta_sq). The simulation is run for
    total_steps = T_desired + burn_in, where burn_in = burn_in_fraction * T_desired.
    The function returns the state trajectory from time 0 to T_desired (i.e., T_desired+1 states)
    after discarding the burn-in.
    
    Parameters:
        sigma_eta_sq (float): Process error variance (used for both process noise and generating the initial state).
        nu (np.ndarray): Array of advection parameters of shape (total_steps+1, 2), where total_steps = T_desired + burn_in.
        beta (float): Diffusion coefficient.
        grid_shape (tuple): Grid dimensions as (grid_size_x, grid_size_y).
        neighbour_locs (np.ndarray): Array of shape (N, 5) containing [index, left, right, up, down] for each grid point.
        T_desired (int): Desired number of time steps (not counting the initial state) after burn-in.
        burn_in_fraction (float): Fraction of T_desired to use as burn-in (default is 0.5).
    
    Returns:
        np.ndarray: Array of shape (N, T_desired+1) containing the latent state trajectory.
        np.ndarray: Array of shape (T_desired+1, 2) containing the advection parameters after burn-in.
    """
    grid_size_x, grid_size_y = grid_shape
    N = grid_size_x * grid_size_y
    burn_in = int(burn_in_fraction * T_desired)
    total_steps = T_desired + burn_in
    
    # Allocate array for states from t = 0 to t = total_steps
    temp_state = np.zeros((N, total_steps + 1))
    temp_state[:, 0] = np.random.normal(0, np.sqrt(sigma_eta_sq), size=N)

    # Extract neighbor indices
    # indices = neighbour_locs[:, 0].astype(int)
    left_neighbors = neighbour_locs[:, 1].astype(int)
    right_neighbors = neighbour_locs[:, 2].astype(int)
    up_neighbors = neighbour_locs[:, 3].astype(int)
    down_neighbors = neighbour_locs[:, 4].astype(int)
    
    # Simulate the state evolution over total_steps time steps
    for t in range(1, total_steps + 1):
        nu_x = nu[t-1, 0]
        nu_y = nu[t-1, 1]
        prev_state = temp_state[:, t - 1]
        
        temp_state[:, t] = ((1 - 4 * beta) * prev_state +
                            (beta - nu_x) * prev_state[right_neighbors] +
                            (beta + nu_x) * prev_state[left_neighbors] +
                            (beta - nu_y) * prev_state[down_neighbors] +
                            (beta + nu_y) * prev_state[up_neighbors] +
                            np.random.normal(0, np.sqrt(sigma_eta_sq), N))
    
    # Discard the burn-in portion. The returned state has times 0,...,T_desired (T_desired+1 states).
    return temp_state[:, burn_in:], nu[burn_in:]


def simulate_observations(state: np.ndarray, sigma_epsilon_sq: float, missing_rate: float = 0.2) -> np.ndarray:
    """
    Simulate the observed data by adding observation noise to the latent states and 
    introducing missing values under a Missing-at-Random (MAR) mechanism.
    
    For t = 1, …, T:
        Z[t] = Y[t] + εₜ,
    where εₜ ~ N(0, sigma_epsilon_sq).
    
    The MAR mechanism here assumes that the probability of an observation being missing 
    depends on its value. For example, lower values are more likely to be missing. 
    We use a logistic function to determine the missing probability for each observation,
    and then scale it so that the overall missing fraction is approximately missing_rate.
    
    Parameters:
        state (np.ndarray): Latent state array of shape (N, time_steps).
        sigma_epsilon_sq (float): Observation error variance.
        missing_rate (float): Desired overall missing fraction (approximately).
    
    Returns:
        np.ndarray: Observations array of shape (N, time_steps) with missing values (np.nan).
    """
    state = state[:, 1:].copy()  # Discard the initial state (t=0) as it is not observed
    N, T = state.shape
    observations = np.zeros((N, T))
    
    # Generate noisy observations from the latent state
    for t in range(T):
        epsilon_t = np.random.normal(0, np.sqrt(sigma_epsilon_sq), N)
        observations[:, t] = state[:, t] + epsilon_t

    # --- Introduce missingness under a MAR mechanism ---
    # Here we assume that lower observations are more likely to be missing.
    # Compute a logistic function to obtain a raw missing probability:
    # a controls the steepness and b is set to the median observation (as a reference level).
    a = 5.0  # slope parameter
    b = np.median(observations)
    raw_missing_prob = 1 / (1 + np.exp((observations - b) * a))
    
    # Scale the raw probabilities so that the overall mean is approximately missing_rate.
    scaling = missing_rate / np.mean(raw_missing_prob)
    p_missing = np.clip(raw_missing_prob * scaling, 0, 1)
    
    # For each element, draw a uniform random number and set to missing if it is below p_missing.
    missing_indicator = np.random.uniform(0, 1, size=(N, T)) < p_missing
    observations[missing_indicator] = np.nan

    return observations

def initialize_simulation_params() -> dict:
    """
    Initialize simulation settings and prior parameters for the MCMC/Gibbs sampler.
    
    Returns:
        dict: A dictionary containing both the simulation parameters and prior hyperparameters.
    """
    params = {}
    # Domain settings (with corrected grid spacing to match "10 km" spacing)
    params['dx'] = 10000.0  # 10 km (in meters)
    params['dy'] = 10000.0  # 10 km (in meters)
    params['grid_size_x'] = 63
    params['grid_size_y'] = 89
    params['grid_shape'] = (params['grid_size_x'], params['grid_size_y'])
    params['N'] = params['grid_size_x'] * params['grid_size_y']
    
    # Time settings
    params['time_steps'] = 100  # Number of time steps
    
    # True simulation parameters
    params['alpha'] = 0.6          # True autoregression coefficient
    params['beta'] = 0.0001         # True diffusion coefficient
    params['sigma_nu_sq'] = 0.01    # True advection error variance
    params['sigma_eta_sq'] = 0.01    # True process error variance
    params['sigma_epsilon_sq'] = 0.01 # True observation error variance
    
    # Prior hyperparameters for the MCMC initialization
    params['prior_params'] = {
        'advection': {
            'mean_nu_x_zero': 0.0,
            'mean_nu_y_zero': 0.0,
            'var_nu_zero': 0.01,
            'a_nu': 10,
            'b_nu': 6
        },
        'process': {
            'm_state': 0,
            'v_state': 1.0,
            'a_eta': 5.0,
            'b_eta': 0.1,
            'sigma_eta_sq': 0.01 # If fixed, use this value for the process error variance
        },
        'observation': {
            'a_epsilon': 5.0,
            'b_epsilon': 0.1,
            'sigma_epsilon_sq': 0.01 # If fixed, use this value for the observation error variance
        },
        'autoregression': {
            'm_alpha': 0.5,
            'v_alpha': 0.01,
        },
        'diffusion': {
            'm_beta': 0.125,
            'v_beta': 0.001736
        },
        'initial_state': {
            'm_state': 0.2, # mean of the initial state
            'v_state': 1.0 # variance of the initial state
        }
    }
    
    # Fixed parameters for the EnKS (Ensemble Kalman Smoother)
    params['N_ensemble'] = 100
    params['smoothing_window'] = 3
    params['burn_in_fraction'] = 0.1
    
    return params

def get_mcmc_initializations(params: dict, observations: np.ndarray = None) -> dict:
    """
    Generate initial guesses for the Gibbs/MCMC sampler based on the specified priors.
    
    This includes initialization of parameters such as alpha, beta, the advection parameters,
    the error variances, and the state (for t>=1).

    Parameters:
        params (dict): Simulation and prior parameter dictionary.
        observations (np.ndarray)
    
    Returns:
        dict: Dictionary containing initial values for MCMC initialization.
    """
    prior_params = params['prior_params']
    time_steps = params['time_steps']
    N = params['N']
    
    init = {}
    
    # Initialize autoregression (alpha) and diffusion (beta) parameters
    init['alpha'] = prior_params['autoregression']['m_alpha']
    init['beta'] = np.random.normal(prior_params['diffusion']['m_beta'], np.sqrt(prior_params['diffusion']['v_beta']))
    
    # Initialize advection parameters for each time step (using the prior mean)
    nu_init = np.zeros((2, time_steps+1))
    nu_init[0, 0] = np.random.normal(prior_params['advection']['mean_nu_x_zero'], np.sqrt(prior_params['advection']['var_nu_zero']))
    nu_init[1, 0] = np.random.normal(prior_params['advection']['mean_nu_y_zero'], np.sqrt(prior_params['advection']['var_nu_zero']))
    for t in range(1, time_steps+1):
        nu_init[0, t] = init['alpha'] * nu_init[0, t - 1] + np.random.normal(prior_params['advection']['mean_nu_y_zero'], np.sqrt(prior_params['advection']['var_nu_zero']))
        nu_init[1, t] = init['alpha'] * nu_init[1, t - 1] + np.random.normal(prior_params['advection']['mean_nu_y_zero'], np.sqrt(prior_params['advection']['var_nu_zero']))
    init['nu'] = nu_init.T  # shape: (time_steps, 2)

    init['sigma_nu_sq'] = np.random.gamma(prior_params['advection']['a_nu'], 1 / prior_params['advection']['b_nu'])
    
    # Informed initialization for the error variances:
    # For real applications the latent state is not known so we turn to the observed data.
    if observations is not None:
        # Compute temporal differences (across time) for each spatial location.
        obs_diff = np.diff(observations, axis=1)  # shape: (N, time_steps-1)
        # Get an overall idea of the scale by averaging the per-location variance, ignoring nans.
        var_diff = np.mean(np.nanvar(obs_diff, axis=1))
        # Under the model: Var(Z[t]-Z[t-1]) ≈ sigma_eta_sq + 2*sigma_epsilon_sq.
        # Assuming (roughly) equal contributions, we split the variance equally.
        init['sigma_eta_sq'] = var_diff / 3
        init['sigma_epsilon_sq'] = var_diff / 3
    else:
        # Fall back to sampling from the Gamma priors.
        init['sigma_eta_sq'] = np.random.gamma(prior_params['process']['a_eta'], 
                                               1 / prior_params['process']['b_eta'])
        init['sigma_epsilon_sq'] = np.random.gamma(prior_params['observation']['a_epsilon'], 
                                                   1 / prior_params['observation']['b_epsilon'])
    
    # Initialize the state for the MCMC (for time steps 1,...,T)
    init['state'] = np.random.normal(prior_params['process']['m_state'],
                                     np.sqrt(prior_params['process']['v_state']),
                                     (N, time_steps + 1))
    
    return init
