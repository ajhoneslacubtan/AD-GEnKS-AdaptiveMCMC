import xarray as xr
import numpy as np
import zarr
import arviz as az
import matplotlib.pylab as plt
import matplotlib.animation as animation

from simulation import (
    initialize_simulation_params,
    create_neighbour_locs,
    get_mcmc_initializations
)
from sampler.gibbs import GibbsSampler

def main():
    # -------------------------------
    # 1. Load and preprocess the real data
    # -------------------------------
    ds = xr.open_dataset('data/spatiotemporal_swr_mindanao.nc', engine="netcdf4", decode_times=True)

    # Convert the time coordinate from UTC to Asia/Manila (UTC+8)
    # This creates a timezone-aware DatetimeIndex.
    time_local = ds.indexes['time'].tz_localize('UTC').tz_convert("Asia/Manila")
    ds = ds.assign_coords(time=time_local)

    # Define the train and test set time boundaries using timestamps (in Asia/Manila time)
    # Train set: 8:00 AM to 2:00 PM PST on 2025-04-04
    # Test set: 2:00 PM to 3:00 PM PST on 2025-04-04
    train_start = "2025-04-04T08:00:00+08:00"
    train_end   = "2025-04-04T14:00:00+08:00"
    test_start  = "2025-04-04T14:00:00+08:00"
    test_end    = "2025-04-04T15:00:00+08:00"

    # Subset the dataset by time using the timestamps
    train_set = ds.sel(time=slice(train_start, train_end))
    test_set  = ds.sel(time=slice(test_start, test_end))

    # select SWR directly as a DataArray
    swr_train_da = train_set["SWR"].transpose("latitude", "longitude", "time")
    swr_test_da  = test_set["SWR"].transpose("latitude", "longitude", "time")

    # now .to_numpy() works because netCDF4 backend supports vindex
    train = np.float64(swr_train_da.to_numpy())
    test  = np.float64(swr_test_da.to_numpy())

    grid_size_x, grid_size_y = train.shape[:2]
    N = grid_size_x * grid_size_y

    # Flatten spatial dimensions so that each row is one grid point
    observations = train.reshape((N, train.shape[2]))  # shape: (N, T)
    test_observations = test.reshape((N, test.shape[2]))  # shape: (N, T)

    # Compute the min and max values for the SWR variable, ignoring NaNs
    vmin = float(np.nanmin(observations))
    vmax = float(np.nanmax(observations))

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # For each time frame, reshape the 1D column vector to the 2D grid (124, 176)
    im = ax.imshow(observations[:, 0].reshape(124, 176), origin='lower',
                cmap='viridis', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    title = ax.set_title(str(observations[0]))

    # Define the update function for the animation
    def update(frame):
        # Reshape the data for the current frame from (21824,) to (124, 176)
        data_frame = observations[:, frame].reshape(124, 176)
        im.set_data(data_frame)
        return im, title

    # Create the animation; adjust the interval (in milliseconds) as needed
    ani = animation.FuncAnimation(fig, update, frames=observations.shape[1],
                                interval=500, blit=True)

    # Save the animation as a GIF using the pillow writer
    ani.save("swr_animation_train.gif", writer="pillow", fps=2)
    plt.close()
    print("GIF saved as swr_animation_train.gif")

    # Set up the plot for test set animation
    fig, ax = plt.subplots(figsize=(8, 6))
    # For each time frame, reshape the 1D column vector to the 2D grid (124, 176)
    im = ax.imshow(test_observations[:, 0].reshape(124, 176), origin='lower',
                cmap='viridis', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    title = ax.set_title(str(test_observations[0]))

    # Define the update function for the test animation
    def update(frame):
        # Reshape the data for the current frame from (21824,) to (124, 176)
        data_frame = test_observations[:, frame].reshape(124, 176)
        im.set_data(data_frame)
        return im, title

    # Create the animation; adjust the interval (in milliseconds) as needed
    ani = animation.FuncAnimation(fig, update, frames=test_observations.shape[1],
                                interval=500, blit=True)

    # Save the animation as a GIF using the pillow writer
    ani.save("swr_animation_test.gif", writer="pillow", fps=2)
    plt.close()
    print("GIF saved as swr_animation_test.gif")

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
    burn_in = 1000
    thin = 2
    num_iterations = 2000

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
        beta_sampling_method='normal'
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
        'nu': np.expand_dims(np.transpose(samples['nu_samples'], (2, 0, 1)), axis=0),
        'state': np.expand_dims(np.transpose(samples['Y_samples'], (2, 0, 1)), axis=0)
    }

    log_likelihood = {
        'log_complete_samples': np.expand_dims(samples['log_complete_samples'], axis=0)
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
        "nu": ["draw", "component", "time_state"],
        "state": ["draw", "location", "time_state"],
        "log_complete_samples": ["draw"],
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
            "nu": dims["nu"],
            "state": dims["state"],
            "observed_variable": dims["observed_variable"],
            "observed_test": dims["observed_test"],
            "log_complete_samples": dims["log_complete_samples"]
        }
    )
    az.to_netcdf(idata_posterior, 'real_inference_data_gibbs_swr.nc')
    
    print("Real data analysis completed successfully!")

if __name__ == '__main__':
    main()
