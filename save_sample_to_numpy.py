import arviz as az
import numpy as np

# Load the InferenceData from NetCDF
idata = az.from_netcdf("inference_data_lag3.nc")

# Extract relevant posterior variables
posterior = idata.posterior

# Extract and convert to NumPy arrays
alpha = posterior["alpha"].values.squeeze()  # shape: (500,)
beta = posterior["beta"].values.squeeze()    # shape: (500,)
nu = posterior["nu"].values.squeeze()        # shape: (500, 2, 36)

# Save each variable as a compressed .npz file
np.savez_compressed("alpha.npz", alpha=alpha)
np.savez_compressed("beta.npz", beta=beta)
np.savez_compressed("nu.npz", nu=nu)

# Extract and squeeze the 'state' variable
state = idata.posterior["state"].values.squeeze()  # shape: (500, 21824, 36)

# Select specific time indices
time_indices = [6, 12, 18, 24, 30, 35]
state_subset = state[:, :, time_indices]  # shape: (500, 21824, 6)

# Optional: Convert to float32 to reduce file size
state_subset = state_subset.astype(np.float32)

# Save as compressed .npz
np.savez_compressed("state_subset.npz", state=state_subset)
