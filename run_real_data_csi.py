import xarray as xr
import numpy as np
import zarr
import arviz as az
import matplotlib.pylab as plt
import matplotlib.animation as animation
import pandas as pd

from sampler.simulation import (
    initialize_simulation_params,
    create_neighbour_locs,
    get_mcmc_initializations
)

from sampler.gibbs import GibbsSampler

def main():
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # -------------------------------
    # 1. Load and preprocess the real data
    # -------------------------------
    ds = xr.open_dataset('data/clear_sky_index_mindanao.nc', engine="netcdf4", decode_times=True)

    # Convert the time coordinate from UTC to Asia/Manila (UTC+8)
    # This creates a timezone-aware DatetimeIndex.
    time_local = ds.indexes['time'].tz_localize('UTC').tz_convert("Asia/Manila")
    ds = ds.assign_coords(time=time_local)

    # Define the train and test set time boundaries using timestamps (in Asia/Manila time)
    # Train set: 8:00 AM to 2:00 PM PST on 2025-04-04
    # Test set: 2:00 PM to 3:00 PM PST on 2025-04-04
    train_start = "2025-04-04T11:00:00"
    train_end   = "2025-04-04T13:21:00"
    test_start  = "2025-04-04T13:21:00"
    test_end    = "2025-04-04T14:21:00"

    # Subset the dataset by time using the timestamps
    train_set = ds.sel(time=slice(train_start, train_end))
    test_set  = ds.sel(time=slice(test_start, test_end))

    # select CSI directly as a DataArray
    csi_train_da = train_set["CSI"].transpose("latitude", "longitude", "time")
    csi_test_da  = test_set["CSI"].transpose("latitude", "longitude", "time")

    # Print dataset to check the time range in a readable format
    print("Train set time range:", pd.Timestamp(train_set.time.values[0]).strftime('%Y-%m-%d %H:%M:%S'), "to", 
          pd.Timestamp(train_set.time.values[-1]).strftime('%Y-%m-%d %H:%M:%S'))
    print("Test set time range:", pd.Timestamp(test_set.time.values[0]).strftime('%Y-%m-%d %H:%M:%S'), "to",
          pd.Timestamp(test_set.time.values[-1]).strftime('%Y-%m-%d %H:%M:%S'))
    # now .to_numpy() works because netCDF4 backend supports vindex
    train = np.float64(csi_train_da.to_numpy())
    test  = np.float64(csi_test_da.to_numpy())

    grid_size_x, grid_size_y = train.shape[:2]
    N = grid_size_x * grid_size_y

    # Flatten spatial dimensions so that each row is one grid point
    observations = train.reshape((N, train.shape[2]))  # shape: (N, T)
    test_observations = test.reshape((N, test.shape[2]))  # shape: (N, T)

    # Compute global color scale limits, ignoring NaNs
    vmin = float(np.nanmin(observations))
    vmax = float(np.nanmax(observations))

    # --- TRAIN SET ANIMATION ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(observations[:, 0].reshape(124, 176),
                   cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("CSI Value")
    title = ax.set_title("Frame 0")

    def update_train(frame):
        data_frame = observations[:, frame].reshape(124, 176)
        im.set_data(data_frame)
        title.set_text(f"Train Set – Frame {frame}")
        return im, title

    ani = animation.FuncAnimation(fig, update_train, frames=observations.shape[1],
                                  interval=500, blit=True)
    ani.save("csi_animation_train.gif", writer="pillow", fps=2)
    plt.close()
    print("CSI saved as csi_animation_train.gif")

    # --- TEST SET ANIMATION ---
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(test_observations[:, 0].reshape(124, 176),
                   cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("CSI Value")
    title = ax.set_title("Frame 0")

    def update_test(frame):
        data_frame = test_observations[:, frame].reshape(124, 176)
        im.set_data(data_frame)
        title.set_text(f"Test Set – Frame {frame}")
        return im, title

    ani = animation.FuncAnimation(fig, update_test, frames=test_observations.shape[1],
                                  interval=500, blit=True)
    ani.save("csi_animation_test.gif", writer="pillow", fps=2)
    plt.close()
    print("GIF saved as csi_animation_test.gif")

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
    params['time_steps'] = observations.shape[1]
    params['smoothing_window'] = 3
    params['sigma_eta_sq'] = 0.05
    params['sigma_epsilon_sq'] = 0.1
    params['prior_params']['initial_state']['v_state'] = 1

    # Create neighbor index array
    neighbour_locs = create_neighbour_locs(grid_size_x, grid_size_y)  # shape: (N, 5)

    # -------------------------------
    # 3. Compute initial state mean for data-driven initialization
    # -------------------------------
    # Leverage 5 pre-training data around 7am PST.
    pre_train_start = "2025-04-04T10:40:00"
    pre_train_end   = "2025-04-04T11:10:00"  # 5 frames at 10-min resolution
    pre_train_set = ds.sel(time=slice(pre_train_start, pre_train_end))
    csi_pre_train_da = pre_train_set["CSI"].transpose("latitude", "longitude", "time")

    # Print dataset to check the time range in a readable format
    print("Pre-train set time range:", pd.Timestamp(csi_pre_train_da.time.values[0]).strftime('%Y-%m-%d %H:%M:%S'), "to", 
          pd.Timestamp(csi_pre_train_da.time.values[-1]).strftime('%Y-%m-%d %H:%M:%S'))

    pre_train = np.float64(csi_pre_train_da.to_numpy())
    # Flatten spatial dimensions: (N, T_pre) where T_pre should equal 5
    pre_train = pre_train.reshape((N, pre_train.shape[2]))
    # Compute the mean of the pre-training observations at each grid point
    initial_state_mean = np.mean(pre_train, axis=1)  # shape: (N,)

    # Generate initializations for the Gibbs sampler (MCMC initialization for the latent state)
    mcmc_init = get_mcmc_initializations(params, observations)

    # -------------------------------
    # 4. Initialize and run the Gibbs sampler
    # -------------------------------
    # Adjust sampler settings as desired (number of iterations, burn-in, thinning)
    burn_in = 1000
    thin = 2
    num_iterations = 2000

    # Fixed Sigma
    gibbs_sampler = GibbsSampler(
        observations=observations,
        neighbour_locs=neighbour_locs,
        num_iterations=num_iterations,
        burn_in=burn_in,
        thin=thin,
        alpha_init=mcmc_init['alpha'],
        beta_init=0.001,
        sigma_eta_sq_init=params['sigma_eta_sq'],
        sigma_nu_sq_init=0.01,
        sigma_epsilon_sq_init=params['sigma_epsilon_sq'],
        nu_init=mcmc_init['nu'],
        state_init=mcmc_init['state'],
        initial_ensemble_mean=initial_state_mean,
        prior_params=params['prior_params'],
        N_ensemble=params['N_ensemble'],
        smoothing_window=params['smoothing_window'],
        fixed_sigmas=True,
        beta_sampling_method="normal"
    )

    samples = gibbs_sampler.sample()
    # -------------------
    # Load Gibbs samples from zarr and in-memory dictionary
    # -------------------
    store = zarr.open('gibbs_results.zarr', mode='r')

    samples.update({
        'Y_samples': store['Y_samples'][:],
        'nu_samples': store['nu_samples'][:]
    })

    # -------------------
    # Create InferenceData dictionaries (excluding sigma samples)
    # -------------------
    posterior = {
        'alpha': np.expand_dims(samples['alpha_samples'], axis=0),  # shape: (1, draws)
        'beta': np.expand_dims(samples['beta_samples'], axis=0),    # shape: (1, draws)
        'nu': np.expand_dims(np.transpose(samples['nu_samples'], (2, 0, 1)), axis=0),     # shape: (1, draws, 2, T+1)
        'state': np.expand_dims(np.transpose(samples['Y_samples'], (2, 0, 1)), axis=0)    # shape: (1, draws, N, T+1)
    }

    log_likelihood = {
        'log_complete_samples': np.expand_dims(samples['log_complete_samples'], axis=0)  # shape: (1, draws)
    }

    # Observed data only (no constant data for real data analysis)
    observed_data = {
        'observed_variable': observations,  # shape: (N, T_obs)
        'observed_test': test_observations  # shape: (N, T_obs)
    }

    # -------------------
    # Create coords and dims
    # -------------------
    num_draws = samples['alpha_samples'].shape[0]
    N = samples['Y_samples'].shape[0]
    T_state = samples['Y_samples'].shape[1]
    T_obs = T_state - 1
    T_obs_test = test_observations.shape[1]

    coords = {
        "draw": np.arange(num_draws),
        "location": np.arange(N),
        "time_state": np.arange(T_state),
        "time_obs": np.arange(T_obs),
        "time_obs_test": np.arange(T_obs_test),
        "component": np.array(["v_x", "v_y"])
    }

    dims = {
        "alpha": ["draw"],
        "beta": ["draw"],
        "nu": ["draw", "component", "time_state"],
        "state": ["draw", "location", "time_state"],
        "log_complete_samples": ["draw"],
        "observed_variable": ["location", "time_obs"],
        "observed_test": ["location", "time_obs_test"]
    }

    # -------------------
    # Format filenames for real data
    # -------------------
    smoothing_window = params['smoothing_window']
    sigma_eta_sq = params['sigma_eta_sq']
    sigma_eta_str = f"{sigma_eta_sq:.1f}"

    posterior_filename = f"real_posterior_sw{smoothing_window}_sigmaEta{sigma_eta_str}.nc"
    rest_filename = f"real_inference_data_rest_sw{smoothing_window}_sigmaEta{sigma_eta_str}.nc"

    # -------------------
    # Save posterior samples
    # -------------------
    idata_posterior = az.from_dict(
        posterior=posterior,
        coords={
            "draw": coords["draw"],
            "location": coords["location"],
            "time_state": coords["time_state"],
            "component": coords["component"]
        },
        dims={
            "alpha": dims["alpha"],
            "beta": dims["beta"],
            "nu": dims["nu"],
            "state": dims["state"],
            "log_complete_samples": dims["log_complete_samples"]
        }
    )
    az.to_netcdf(idata_posterior, posterior_filename)
    print(f"Posterior samples saved to '{posterior_filename}'")

    # -------------------
    # Save observed data and log likelihood
    # -------------------
    idata_rest = az.from_dict(
        observed_data=observed_data,
        log_likelihood=log_likelihood,
        coords={
            "draw": coords["draw"],
            "location": coords["location"],
            "time_obs": coords["time_obs"],
            "time_obs_test": coords["time_obs_test"],
        },
        dims={
            "observed_variable": dims["observed_variable"],
            "observed_test": dims["observed_test"],
            "log_complete_samples": dims["log_complete_samples"]
        }
    )
    az.to_netcdf(idata_rest, rest_filename)
    print(f"Remaining data saved to '{rest_filename}'")

    # -------------------
    # Posterior summary
    # -------------------
    summary = az.summary(idata_posterior.posterior, var_names=["alpha", "beta"])
    print(summary)

if __name__ == '__main__':
    main()
    print("Real data analysis completed successfully!")
