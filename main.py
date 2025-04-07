import matplotlib.pyplot as plt
import matplotlib.animation as animation
import zarr
import numpy as np
import arviz as az
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
    initial_state = np.load("data/train_t10.npy")
    # Modify the parameters for the Gibbs sampler
    params['N_ensemble'] = 100
    params['smoothing_window'] = 6
    params['time_steps'] = 35
    params['grid_size_x'] = initial_state.shape[0]
    params['grid_size_y'] = initial_state.shape[1]
    params['grid_shape'] = initial_state.shape
    params['N'] = params['grid_size_x'] * params['grid_size_y']
    params['beta'] = 0.2
    params['alpha'] = 0.8
    params['sigma_nu_sq'] = 0.01
    params['sigma_eta_sq'] = 225
    params['sigma_epsilon_sq'] = 800


    logger.debug(f"Parameters initialized: {params}")
    
    # Create neighbor locations matrix
    neighbour_locs = create_neighbour_locs(params['grid_size_x'], params['grid_size_y'])
    
    # Simulate advection parameters using an AR(1) process
    nu = simulate_advection(params['alpha'], params['sigma_nu_sq'], params['time_steps'], 
                            burn_in_fraction=params['burn_in_fraction'])
    
    # Simulate the latent state evolution
    
    state, nu = simulate_state(params['sigma_eta_sq'], nu, params['beta'], params['grid_shape'], 
                           neighbour_locs, params['time_steps'], params['burn_in_fraction'],
                           initial_state=initial_state)
    
    # Simulate the observations (data) by adding observation noise
    observations = simulate_observations(state, params['sigma_epsilon_sq'], missing_rate=0.05)
    
    # Generate initializations for the Gibbs sampler
    mcmc_init = get_mcmc_initializations(params, observations)

    # # Set up the plot for observations animation
    # fig, ax = plt.subplots(figsize=(8, 6))
    
    # # Compute min and max values for consistent colorbar scale, ignoring NaNs
    # vmin = float(np.nanmin(observations))
    # vmax = float(np.nanmax(observations))
    
    # # Initialize the plot with first frame
    # im = ax.imshow(observations[:, 0].reshape(params['grid_shape']), 
    #                origin='lower', cmap='viridis',
    #                vmin=vmin, vmax=vmax)
    
    # # Add colorbar
    # plt.colorbar(im, ax=ax)
    # title = ax.set_title(f'Time step: 0')
    
    # # Animation update function
    # def update(frame):
    #     # Reshape the data for current frame
    #     data_frame = observations[:, frame].reshape(params['grid_shape'])
    #     im.set_data(data_frame)
    #     title.set_text(f'Time step: {frame}')
    #     return im, title
    
    # # Create animation
    # ani = animation.FuncAnimation(fig, update, 
    #                             frames=observations.shape[1],
    #                             interval=200, # 200ms between frames
    #                             blit=True)
    
    # # Save animation
    # ani.save("observations_animation.gif", writer="pillow", fps=5)
    # plt.close()
    # logger.info("Saved observations animation as observations_animation.gif")


    # Burn-in, thinning and number of iterations for the Gibbs sampler
    burn_in = 1000
    thin = 2
    iter = 2000

    # Initialize the Gibbs sampler with fixed sigmas
    gibbs_sampler = GibbsSampler(
        observations=observations,
        neighbour_locs=neighbour_locs,
        num_iterations=iter,
        burn_in=burn_in,
        thin=thin,
        alpha_init=mcmc_init['alpha'],
        beta_init=mcmc_init['beta'],
        sigma_eta_sq_init=params['sigma_eta_sq'],
        sigma_nu_sq_init=params['sigma_nu_sq'],
        sigma_epsilon_sq_init=params['sigma_epsilon_sq'],
        nu_init=mcmc_init['nu'],
        state_init=mcmc_init['state'],
        prior_params=params['prior_params'],
        N_ensemble=params['N_ensemble'],
        smoothing_window=params['smoothing_window'],
        fixed_sigmas=True,
        beta_sampling_method='normal'
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
        # 'true_state': state,
        'true_nu': nu.T
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
        "true_nu": ["component", "time_state"]
    }
    
    # Save posterior samples and observations as InferenceData
    logger.info("Saving posterior samples and observations")
    idata_posterior = az.from_dict(
        posterior=posterior,
        observed_data=observed_data,
        constant_data=constant_data,
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
            "true_nu": dims["true_nu"],
        }
    )
    az.to_netcdf(idata_posterior, 'inference_data_lag6.nc')
    logger.info("Results saved to inference_data.nc")

    # Print Posterior Summary
    logger.info("Generating posterior summary")
    summary = az.summary(idata_posterior.posterior, var_names=["alpha", "beta", "sigma_nu_sq"])
    print(summary)
    logger.info("Simulation completed successfully!")

if __name__ == '__main__':
    main()
    print("Simulation completed successfully!")
