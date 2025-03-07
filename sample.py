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
    nu = simulate_advection(params['alpha'], params['sigma_nu_sq'], params['time_steps'], burn_in_fraction=params['burn_in_fraction'])
    
    # Simulate the latent state evolution
    state = simulate_state(params['sigma_eta_sq'], nu, params['beta'], params['grid_shape'], neighbour_locs, params['time_steps'], params['burn_in_fraction'])
    
    # Simulate the observations (data) by adding observation noise
    observations = simulate_observations(state, params['sigma_epsilon_sq'], missing_rate=0.1)
    
    # Generate initializations for the Gibbs sampler
    mcmc_init = get_mcmc_initializations(params, observations)

    # Test the EnKS
    # Run the EnKS 

    # # # Augment missing observations
    # augmented_observations = observations.copy()
    # missing_mask = np.isnan(augmented_observations)
    # augmented_observations[missing_mask] = np.random.normal(
    #     loc=state[:, 1:][missing_mask],
    #     scale=np.sqrt(params['sigma_epsilon_sq'])
    # )


    # start_time = time.time()
    # for i in range(1):
    #     enks = EnKS(params['N_ensemble'], params['smoothing_window'], params['beta'], mcmc_init['nu'], params['sigma_eta_sq'], params['sigma_epsilon_sq'], params['prior_params'])
    #     Y_analysis = enks.run(augmented_observations, neighbour_locs)
    # end_time = time.time()
    # print(f"EnKS vectorized time taken: {end_time - start_time} seconds")

    # # Sample the state from the EnKS
    # state_enks = Y_analysis[:, np.random.randint(0, params['N_ensemble']), :]

    # # Evaluate the EnKS
    # evaluation = evaluate_enks_sample(state_enks, state, params['grid_shape'])
    # print(evaluation)


    # Burn-in, thinning and number of iterations for the Gibbs sampler
    burn_in = 10
    thin = 2
    iter = 4000

    # Initialize the Gibbs sampler with estimated sigmas
    # Define the fixed sigmas
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
        fixed_sigmas=True
    )

    samples = gibbs_sampler.sample()

    # Save the samples to a file. Use pickle to save the dictionary
    file_name = f'samples_fixed_sigmas_{iter}_{burn_in}_{thin}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(samples, f)

    posterior = {
    # Scalar parameters: shape (1, num_draws)
    'alpha': np.expand_dims(samples['alpha_samples'], axis=0),
    'beta': np.expand_dims(samples['beta_samples'], axis=0),
    'sigma_nu_sq': np.expand_dims(samples['sigma_nu_sq_samples'], axis=0),
    'sigma_eta_sq': np.expand_dims(samples['sigma_eta_sq_samples'], axis=0),
    'sigma_epsilon_sq': np.expand_dims(samples['sigma_epsilon_sq_samples'], axis=0),
    'nu': np.expand_dims(samples['nu_samples'].transpose(2, 0, 1), axis=0),
    'state': np.expand_dims(samples['state_samples'].transpose(2, 0, 1), axis=0)
    }

    log_likelihood = {
        'log_complete_samples': np.expand_dims(samples['log_complete_samples'], axis=0),
        'log_obs_samples': np.expand_dims(samples['log_obs_samples'], axis=0),
    }

    # Generate Prior Predictive Samples
    num_samples = samples['alpha_samples'].shape[0] 
    prior_samples = generate_prior_predictive(num_samples, params, neighbour_locs, params['prior_params'])
    prior = {
    'alpha': np.expand_dims(prior_samples['alpha_samples'], axis=0),
    'beta': np.expand_dims(prior_samples['beta_samples'], axis=0),
    'sigma_nu_sq': np.expand_dims(prior_samples['sigma_nu_sq_samples'], axis=0),
    'sigma_eta_sq': np.expand_dims(prior_samples['sigma_eta_sq_samples'], axis=0),
    'sigma_epsilon_sq': np.expand_dims(prior_samples['sigma_epsilon_sq_samples'], axis=0),
    'nu': np.expand_dims(prior_samples['nu_samples'].transpose(2, 0, 1), axis=0),
    'state': np.expand_dims(prior_samples['state_samples'].transpose(2, 0, 1), axis=0)
    }
    
    # Predictive Samples
    prior_predictive = {'Z': prior_samples.pop('predictive_samples')}
    posterior_predictive = {'Z': generate_posterior_predictive(posterior, params, num_samples=num_samples)}

    # Create the InferenceData object
    idata = az.from_dict(posterior=posterior, 
                         log_likelihood=log_likelihood, 
                         prior=prior, 
                         posterior_predictive=posterior_predictive, 
                         prior_predictive=prior_predictive)
    
    # Print a summary of the posterior samples
    print(az.summary(idata, round_to =  8))

    # Save the InferenceData object to a netCDF file
    az.to_netcdf(idata, 'inference_data_fixed_sigmas.nc')


if __name__ == '__main__':

    main()
