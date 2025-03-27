import zarr
import numpy as np
import arviz as az
import pickle
from utils.logging_utils import setup_logger

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
    # Set up logging
    logger = setup_logger("MCMC", "logs/mcmc.log")
    logger.info("Starting MCMC simulation")
    
    # Set the random seed for reproducibility
    np.random.seed(42)
    logger.info("Random seed set to 42")

    # Initialize simulation parameters and priors
    logger.info("Initializing simulation parameters")
    params = initialize_simulation_params()
    # Modify the parameters for the Gibbs sampler
    params['N_ensemble'] = 100
    params['smoothing_window'] = 12
    params['time_steps'] = 60
    logger.debug(f"Parameters initialized: {params}")
    
    # Create neighbor locations matrix
    neighbour_locs = create_neighbour_locs(params['grid_size_x'], params['grid_size_y'])
    
    # Simulate advection parameters using an AR(1) process
    nu = simulate_advection(params['alpha'], params['sigma_nu_sq'], params['time_steps'], 
                            burn_in_fraction=params['burn_in_fraction'])
    
    # Simulate the latent state evolution
    state, nu = simulate_state(params['sigma_eta_sq'], nu, params['beta'], params['grid_shape'], 
                           neighbour_locs, params['time_steps'], params['burn_in_fraction'])
    
    # Simulate the observations (data) by adding observation noise
    observations = simulate_observations(state, params['sigma_epsilon_sq'], missing_rate=0.1)
    
    # Generate initializations for the Gibbs sampler
    mcmc_init = get_mcmc_initializations(params, observations)

    # Burn-in, thinning and number of iterations for the Gibbs sampler
    burn_in = 3000
    thin = 2
    iter = 3800

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
        beta_init=0.000001,
        sigma_eta_sq_init=sigma_eta_sq,
        sigma_nu_sq_init=mcmc_init['sigma_nu_sq'],
        sigma_epsilon_sq_init=sigma_epsilon_sq,
        nu_init=mcmc_init['nu'],
        state_init=mcmc_init['state'],
        prior_params=params['prior_params'],
        N_ensemble=params['N_ensemble'],
        smoothing_window=params['smoothing_window'],
        fixed_sigmas=True,
        beta_sampling_method='adaptive'
    )

    # Load the samples dictionary containing both in-memory and zarr paths
    samples = gibbs_sampler.sample()
    
    # Read the large arrays from zarr
    store = zarr.open('gibbs_results.zarr', mode='r')
    
    # Combine in-memory samples with zarr-stored samples
    samples.update({
        'Y_samples': store['Y_samples'][:],
        'nu_samples': store['nu_samples'][:]
    })

    # ====================
    # Create the InferenceData dictionaries
    # ====================
    posterior = {
        'alpha': np.expand_dims(samples['alpha_samples'], axis=0),
        'beta': np.expand_dims(samples['beta_samples'], axis=0),
        'sigma_nu_sq': np.expand_dims(samples['sigma_nu_sq_samples'], axis=0),
        'sigma_eta_sq': np.expand_dims(samples['sigma_eta_sq_samples'], axis=0),
        'sigma_epsilon_sq': np.expand_dims(samples['sigma_epsilon_sq_samples'], axis=0),
        'nu': np.expand_dims(np.transpose(samples['nu_samples'], (2, 0, 1)), axis=0),  # (1, num_draws, 2, T+1)
        'state': np.expand_dims(np.transpose(samples['Y_samples'], (2, 0, 1)), axis=0)  # (1, num_draws, N, T+1)
    }

    log_likelihood = {
        'log_complete_samples': np.expand_dims(samples['log_complete_samples'], axis=0)
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
    
    predictive_Z = prior_samples.pop('predictive_samples')
    predictive_Z = np.expand_dims(np.transpose(predictive_Z, (2, 0, 1)), axis=0)
    prior_predictive = {'Z': predictive_Z}

    # Posterior Predictive Samples
    ppd = generate_posterior_predictive(samples['Y_samples'], samples['sigma_epsilon_sq_samples'])
    posterior_predictive = {'Z': np.expand_dims(np.transpose(ppd, (2, 0, 1)), axis=0)}

    # Create observed_data and constant_data dictionaries
    observed_data = {
        'observed_variable': observations  # Replace 'observed_variable' with your variable name
    }

    constant_data = {
        'true_state': state,
        'true_advection': nu.T
    }

    # Create coords and dims dictionaries
    num_draws = samples['alpha_samples'].shape[0]
    N = samples['Y_samples'].shape[0]
    T_state = samples['Y_samples'].shape[1]
    T_obs = ppd.shape[1]
    
    coords = {
        "draw": np.arange(num_draws),
        "location": np.arange(N),
        "time_state": np.arange(T_state),
        "time_obs": np.arange(T_obs),
        "component": np.array(["v_x", "v_y"])
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
        "Z": ["draw", "location", "time_obs"],
        "observed_variable": ["location", "time_obs"],
        "true_state": ["location", "time_state"],
        "true_advection": ["component", "time_state"]
    }

    # (After creating coords and dims dictionaries)
    
    # Save posterior samples and observations as InferenceData
    logger.info("Saving posterior samples and observations")
    idata_posterior = az.from_dict(
        posterior=posterior,
        observed_data=observed_data,
        coords={
            "draw": coords["draw"],
            "location": coords["location"],
            "time_state": coords["time_state"],
            "time_obs": coords["time_obs"]
        },
        dims={
            "alpha": dims["alpha"],
            "beta": dims["beta"],
            "sigma_nu_sq": dims["sigma_nu_sq"],
            "sigma_eta_sq": dims["sigma_eta_sq"],
            "sigma_epsilon_sq": dims["sigma_epsilon_sq"],
            "nu": dims["nu"],
            "state": dims["state"],
            "observed_variable": dims["observed_variable"],
        }
    )
    az.to_netcdf(idata_posterior, 'inference_data.nc')
    logger.info("Results saved to inference_data.nc")

    # Print Posterior Summary
    logger.info("Generating posterior summary")
    summary = az.summary(idata_posterior.posterior, var_names=["alpha", "beta", "sigma_nu_sq"])
    print(summary)
    logger.info("Simulation completed successfully!")

if __name__ == '__main__':
    main()
    print("Simulation completed successfully!")
