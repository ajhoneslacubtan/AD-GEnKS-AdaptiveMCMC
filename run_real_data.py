import xarray as xr
import numpy as np
import zarr
import arviz as az

from simulation import (
    initialize_simulation_params,
    create_neighbour_locs,
    get_mcmc_initializations
)
from sampler.gibbs import GibbsSampler

def main():
    # -------------------------------
    # 1. Load and preprocess the real SPI data
    # -------------------------------
    ds = xr.open_dataset("data/mindanao_GSMaP_spi_gamma_12_month.nc")
    spi = ds['spi_gamma_12_month']  # shape: (lat, lon, time) e.g., (63, 89, 325)

    # Choose training period: t=240 to t=300 (Python indices 240:300 yields 60 time steps)
    train_spi = spi.values[:, :, 240:300]   # shape: (63, 89, 60)
    # Define test set: t=300 to t=325 (25 time steps)
    test_spi  = spi.values[:, :, 300:325]     # shape: (63, 89, 25)

    grid_size_x, grid_size_y = train_spi.shape[:2]
    N = grid_size_x * grid_size_y

    # Flatten spatial dimensions so that each row is one grid point
    observations = train_spi.reshape((N, train_spi.shape[2]))  # shape: (N, 60)
    test_observations = test_spi.reshape((N, test_spi.shape[2]))  # shape: (N, 25)

    # -------------------------------
    # 2. Configure the Gibbs sampler settings
    # -------------------------------
    params = initialize_simulation_params()
    # Update grid-related parameters according to the real data dimensions
    params['grid_size_x'] = grid_size_x
    params['grid_size_y'] = grid_size_y
    params['grid_shape'] = (grid_size_x, grid_size_y)
    params['N'] = N
    # Set time steps based on the training observations (note: observations are from t=1,...,T; initial state is unobserved)
    params['time_steps'] = observations.shape[1]  # T = 60 for training set
    params['smoothing_window'] = 6

    # Create neighbor index array
    neighbour_locs = create_neighbour_locs(grid_size_x, grid_size_y)  # shape: (N, 5)

    # Generate initializations for the Gibbs sampler, using the training observations
    mcmc_init = get_mcmc_initializations(params, observations)

    # -------------------------------
    # 3. Initialize and run the Gibbs sampler
    # -------------------------------
    # Adjust sampler settings as desired (number of iterations, burn-in, thinning)
    burn_in = 2800
    thin = 2
    num_iterations = 4000

    # For this example, we let the sampler update sigma_eta_sq and sigma_epsilon_sq.
    sigma_eta_sq = mcmc_init['sigma_eta_sq']
    sigma_epsilon_sq = mcmc_init['sigma_epsilon_sq']

    gibbs_sampler = GibbsSampler(
        observations=observations,
        neighbour_locs=neighbour_locs,
        num_iterations=num_iterations,
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
        generate_forecasts=False
    )

    samples = gibbs_sampler.sample()

    # Load large arrays from zarr
    store = zarr.open('gibbs_results.zarr', mode='r')
    samples.update({
        'Y_samples': store['Y_samples'][:],
        'nu_samples': store['nu_samples'][:]
    })
    if 'forecast_states' in store:
        samples.update({
            'forecast_states': store['forecast_states'][:],
            'forecast_observations': store['forecast_observations'][:]
        })

    # -------------------------------
    # 4. Build InferenceData and save results (posterior only in this example)
    # -------------------------------
    posterior = {
        'alpha': np.expand_dims(samples['alpha_samples'], axis=0),
        'beta': np.expand_dims(samples['beta_samples'], axis=0),
        'sigma_nu_sq': np.expand_dims(samples['sigma_nu_sq_samples'], axis=0),
        'sigma_eta_sq': np.expand_dims(samples['sigma_eta_sq_samples'], axis=0),
        'sigma_epsilon_sq': np.expand_dims(samples['sigma_epsilon_sq_samples'], axis=0),
        'nu': np.expand_dims(np.transpose(samples['nu_samples'], (2, 0, 1)), axis=0),
        'state': np.expand_dims(np.transpose(samples['Y_samples'], (2, 0, 1)), axis=0)
    }

    log_likelihood = {
        'log_complete_samples': np.expand_dims(samples['log_complete_samples'], axis=0),
        'log_obs_samples': np.expand_dims(samples['log_obs_samples'], axis=0)
    }

    coords = {
        "draw": np.arange(posterior['alpha'].shape[1]),
        "location": np.arange(N),
        "time_state": np.arange(observations.shape[1]+1),
        "time_obs": np.arange(observations.shape[1]),
        "test_time_obs": np.arange(test_observations.shape[1]),
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
        "observed_variable": ["location", "time_obs"],
        "observed_test": ["location", "test_time_obs"]
    }

    idata_posterior = az.from_dict(
        posterior=posterior,
        observed_data={
            'observed_variable': observations,
            'observed_test': test_observations
        },
        log_likelihood=log_likelihood,
        coords={
            "draw": coords["draw"],
            "location": coords["location"],
            "time_state": coords["time_state"],
            "time_obs": coords["time_obs"],
            "test_time_obs": coords["test_time_obs"],
            "component": coords["component"]
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
            "observed_test": dims["observed_test"],
            "log_complete_samples": dims["log_complete_samples"],
            "log_obs_samples": dims["log_obs_samples"]
        }
    )
    az.to_netcdf(idata_posterior, 'real_inference_data.nc')
    
    print("Real data analysis completed successfully!")

if __name__ == '__main__':
    main()
