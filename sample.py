import time
import pickle
import numpy as np
from simulation import (
    initialize_simulation_params,
    simulate_advection,
    simulate_state,
    simulate_observations,
    create_neighbour_locs,
    get_mcmc_initializations
)
from sampler_cython.gibbs import GibbsSampler
from sampler_cython.EnKS import EnKS
# from evaluation.evaluate_EnKS import evaluate_enks_sample

def main():
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Initialize simulation parameters and priors
    params = initialize_simulation_params()
    # Modify the parameters for the Gibbs sampler
    params['N_ensemble'] = 100
    params['smoothing_window'] = 3
    params['time_steps'] = 70
    
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

    # # Augment missing observations
    augmented_observations = observations.copy()
    missing_mask = np.isnan(augmented_observations)
    augmented_observations[missing_mask] = np.random.normal(
        loc=state[:, 1:][missing_mask],
        scale=np.sqrt(params['sigma_epsilon_sq'])
    )


    start_time = time.time()
    for i in range(1):
        enks = EnKS(params['N_ensemble'], params['smoothing_window'], params['beta'], mcmc_init['nu'], params['sigma_eta_sq'], params['sigma_epsilon_sq'], params['prior_params'])
        Y_analysis = enks.run(augmented_observations, neighbour_locs)
    end_time = time.time()
    print(f"EnKS vectorized time taken: {end_time - start_time} seconds")

    # Sample the state from the EnKS
    state_enks = Y_analysis[:, np.random.randint(0, params['N_ensemble']), :]

    # # Evaluate the EnKS
    # evaluation = evaluate_enks_sample(state_enks, state, params['grid_shape'])
    # print(evaluation)


    # # Burn-in, thinning and number of iterations for the Gibbs sampler
    # burn_in = 1200
    # thin = 2
    # iter = 2000

    # # Initialize the Gibbs sampler with estimated sigmas
    # # Define the fixed sigmas
    # sigma_eta_sq = 0.01
    # sigma_epsilon_sq = 0.01

    # # Initialize the Gibbs sampler with fixed sigmas
    # gibbs_sampler = GibbsSampler(
    #     observations=observations,
    #     neighbour_locs=neighbour_locs,
    #     num_iterations=iter,
    #     burn_in=burn_in,
    #     thin=thin,
    #     alpha_init=mcmc_init['alpha'],
    #     beta_init=mcmc_init['beta'],
    #     sigma_eta_sq_init=sigma_eta_sq,
    #     sigma_nu_sq_init=mcmc_init['sigma_nu_sq'],
    #     sigma_epsilon_sq_init=sigma_epsilon_sq,
    #     nu_init=mcmc_init['nu'],
    #     state_init=mcmc_init['state'],
    #     prior_params=params['prior_params'],
    #     N_ensemble=params['N_ensemble'],
    #     smoothing_window=params['smoothing_window'],
    #     fixed_sigmas=True
    # )

    # samples = gibbs_sampler.sample()

    # # Save the samples to a file. Use pickle to save the dictionary
    # file_name = f'samples_fixed_sigmas_{iter}_{burn_in}_{thin}.pkl'
    # with open(file_name, 'wb') as f:
    #     pickle.dump(samples, f)


if __name__ == '__main__':

    main()
