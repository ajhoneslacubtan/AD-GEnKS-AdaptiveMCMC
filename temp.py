import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Load the data
idata = az.from_netcdf("real_posterior_sw6_sigmaEta200.0.nc")
# extra_data = az.from_netcdf("inference_data_rest_sw_6.nc")
# true_nu = extra_data.constant_data["true_nu"].values

# Extract posterior samples of nu
nu_samples = idata.posterior["nu"].values
# Flatten chain and draw into one dimension
nu_flat = nu_samples.reshape(-1, nu_samples.shape[2], nu_samples.shape[3])

# Compute posterior summary
nu_mean = np.mean(nu_flat, axis=0)
nu_lower = np.percentile(nu_flat, 2.5, axis=0)
nu_upper = np.percentile(nu_flat, 97.5, axis=0)

# Time axis excluding t=0 and t=T
time = np.arange(1, nu_mean.shape[1] - 1)

# Create figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
fig.suptitle("Analysis 1: Window=3", fontsize=14)

# Plot v_x
axs[0].fill_between(time, nu_lower[0, 1:-1], nu_upper[0, 1:-1],
                    color='blue', alpha=0.2, label="95% Credible Interval")
axs[0].plot(time, nu_mean[0, 1:-1], 'b--', linewidth=2, label="Posterior Mean")
# axs[0].plot(time, true_nu[0, 1:-1], 'k-', linewidth=2, label="True $v_x$")
axs[0].set_ylabel("$v_x$")
axs[0].set_title("$v_x$ over Time")
axs[0].legend(loc="best")

# Plot v_y
axs[1].fill_between(time, nu_lower[1, 1:-1], nu_upper[1, 1:-1],
                    color='green', alpha=0.2, label="95% Credible Interval")
axs[1].plot(time, nu_mean[1, 1:-1], 'g--', linewidth=2, label="Posterior Mean")
# axs[1].plot(time, true_nu[1, 1:-1], 'k-', linewidth=2, label="True $v_y$")
axs[1].set_ylabel("$v_y$")
axs[1].set_xlabel("Time Index")
axs[1].set_title("$v_y$ over Time")
axs[1].legend(loc="best")

# Save the figure
plt.savefig("advection_plot.png", dpi=300)
plt.close()

# # Plot trace for log likelihood
# az.plot_trace(extra_data.log_likelihood, var_names=["log_complete_samples"])
# plt.gcf().savefig("log_complete_samples_trace.png", dpi=300, bbox_inches='tight')
# plt.close()

# rmse_vx = np.sqrt(np.mean((true_nu[0, 1:-1] - nu_mean[0, 1:-1])**2))
# rmse_vy = np.sqrt(np.mean((true_nu[1, 1:-1] - nu_mean[1, 1:-1])**2))
# print(f"RMSE for $v_x$: {rmse_vx:.4f}")
# print(f"RMSE for $v_y$: {rmse_vy:.4f}")