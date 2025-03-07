import pickle
import numpy as np
import arviz as az
from simulation import (
    initialize_simulation_params,
    simulate_advection,
    simulate_state,
    simulate_observations,
    create_neighbour_locs,
    get_mcmc_initializations
)
from sampler.gibbs import GibbsSampler
from sampler.predictive_distributions import generate_posterior_predictive, generate_prior_predictive

def main():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Initialize simulation parameters and priors
    params = initialize_simulation_params()
    # Modify the parameters for the Gibbs sampler
    params['N_ensemble'] = 100
    params['smoothing_window'] = 3
    params['time_steps'] = 50
    
    # Create neighbor locations matrix
    neighbour_locs = create_neighbour_locs(params['grid_size_x'], params['grid_size_y'])
    
    # Simulate advection parameters using an AR(1) process
    nu = simulate_advection(params['alpha'], params['sigma_nu_sq'], params['time_steps'], 
                            burn_in_fraction=params['burn_in_fraction'])
    
    # Simulate the latent state evolution
    state = simulate_state(params['sigma_eta_sq'], nu, params['beta'], params['grid_shape'], 
                           neighbour_locs, params['time_steps'], params['burn_in_fraction'])
    
    # Simulate the observations (data) by adding observation noise
    observations = simulate_observations(state, params['sigma_epsilon_sq'], missing_rate=0.1)
    
    # Generate initializations for the Gibbs sampler
    mcmc_init = get_mcmc_initializations(params, observations)

    # Burn-in, thinning and number of iterations for the Gibbs sampler
    burn_in = 10
    thin = 2
    iter = 20

    # Define fixed sigma values
    sigma_eta_sq = 0.01
    sigma_epsilon_sq = 0.01

    # Initialize the Gibbs sampler with fixed sigmas
    gibbs_sampler = GibbsSampler(
        observations=observations,
        neighbour_locs=neighbour_locs,
        num_iterations=iter,
        burn_in=burn_in,
        thin=thin,
        alpha_init=mcmc_init['alpha'],
        beta_init=mcmc_init['beta'],
        sigma_eta_sq_init=sigma_eta_sq,
        sigma_nu_sq_init=mcmc_init['sigma_nu_sq'],
        sigma_epsilon_sq_init=sigma_epsilon_sq,
        nu_init=mcmc_init['nu'],
        state_init=mcmc_init['state'],
        prior_params=params['prior_params'],
        N_ensemble=params['N_ensemble'],
        smoothing_window=params['smoothing_window'],
        fixed_sigmas=True,
        generate_forecasts=True
    )

    samples = gibbs_sampler.sample()

    # # Save the raw samples using pickle
    # file_name = f'samples_fixed_sigmas_{iter}_{burn_in}_{thin}.pkl'
    # with open(file_name, 'wb') as f:
    #     pickle.dump(samples, f)

    # ====================
    # Create the InferenceData dictionaries
    # ====================

    # For scalar parameters we add a chain dimension (axis 0)
    posterior = {
        'alpha': np.expand_dims(samples['alpha_samples'], axis=0),  # shape: (1, num_draws)
        'beta': np.expand_dims(samples['beta_samples'], axis=0),
        'sigma_nu_sq': np.expand_dims(samples['sigma_nu_sq_samples'], axis=0),
        'sigma_eta_sq': np.expand_dims(samples['sigma_eta_sq_samples'], axis=0),
        'sigma_epsilon_sq': np.expand_dims(samples['sigma_epsilon_sq_samples'], axis=0),
        # For arrays with extra dims, we transpose so that draws come first then add chain dim.
        'nu': np.expand_dims(np.transpose(samples['nu_samples'], (2, 0, 1)), axis=0),       # (1, num_draws, 2, T+1)
        'state': np.expand_dims(np.transpose(samples['Y_samples'], (2, 0, 1)), axis=0)  # (1, num_draws, N, T+1)
    }

    log_likelihood = {
        'log_complete_samples': np.expand_dims(samples['log_complete_samples'], axis=0),
        'log_obs_samples': np.expand_dims(samples['log_obs_samples'], axis=0),
    }

    # Generate Prior Predictive Samples
    num_samples = samples['alpha_samples'].shape[0] 
    prior_samples = generate_prior_predictive(num_samples, params, params['prior_params'], neighbour_locs)
    prior = {
        'alpha': np.expand_dims(prior_samples['alpha_samples'], axis=0),
        'beta': np.expand_dims(prior_samples['beta_samples'], axis=0),
        'sigma_nu_sq': np.expand_dims(prior_samples['sigma_nu_sq_samples'], axis=0),
        'sigma_eta_sq': np.expand_dims(prior_samples['sigma_eta_sq_samples'], axis=0),
        'sigma_epsilon_sq': np.expand_dims(prior_samples['sigma_epsilon_sq_samples'], axis=0),
        'nu': np.expand_dims(np.transpose(prior_samples['nu_samples'], (2, 0, 1)), axis=0),
        'state': np.expand_dims(np.transpose(prior_samples['Y_samples'], (2, 0, 1)), axis=0)
    }
    
    # Process Prior Predictive Samples:
    # Expected shape is (N, T, num_samples). We transpose so that draws come first and add chain.
    predictive_Z = prior_samples.pop('predictive_samples')
    predictive_Z = np.expand_dims(np.transpose(predictive_Z, (2, 0, 1)), axis=0)  # shape: (1, num_samples, N, T)
    prior_predictive = {'Z': predictive_Z}

    # Posterior Predictive Samples
    # Expected shape is (N, T, M). We process similarly.
    ppd = generate_posterior_predictive(samples['Y_samples'], samples['sigma_epsilon_sq_samples'])
    posterior_predictive = {'Z': np.expand_dims(np.transpose(ppd, (2, 0, 1)), axis=0)}

    # --- Define Coordinates and Dimensions ---
    # Derive time dimension from the state samples (T+1 time steps)
    num_draws = samples['alpha_samples'].shape[0]
    N = samples['Y_samples'].shape[0]         # number of spatial locations
    T_state = samples['Y_samples'].shape[1]     # T+1 time steps for state variables (e.g., 51)
    T_obs = ppd.shape[1]                       # T time steps for observations (e.g., 50)
    
    coords = {
        "draw": np.arange(num_draws),
        "location": np.arange(N),
        "time_state": np.arange(T_state),
        "time_obs": np.arange(T_obs),
        "component": np.array(["comp1", "comp2"])
    }

    dims = {
        "alpha": ["draw"],
        "beta": ["draw"],
        "sigma_nu_sq": ["draw"],
        "sigma_eta_sq": ["draw"],
        "sigma_epsilon_sq": ["draw"],
        "nu": ["draw", "component", "time_state"],
        "state": ["draw", "location", "time_state"],
        "log_complete_samples": ["draw"],
        "log_obs_samples": ["draw"],
        # For both prior and posterior predictive, we use the same time coordinate as state
        "Z": ["draw", "location", "time_obs"]
    }

    # --- Create the InferenceData Object ---
    idata = az.from_dict(
        posterior=posterior,
        log_likelihood=log_likelihood,
        prior=prior,
        posterior_predictive=posterior_predictive,
        prior_predictive=prior_predictive,
        coords=coords,
        dims=dims
    )

    # Print a summary of the posterior samples
    print(az.summary(idata, round_to=8))

    # Save the InferenceData object to a netCDF file
    az.to_netcdf(idata, 'inference_data_fixed_sigmas.nc')


if __name__ == '__main__':
    main()
