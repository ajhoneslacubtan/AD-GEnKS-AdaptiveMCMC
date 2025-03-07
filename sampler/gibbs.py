import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any
from scipy.stats import truncnorm, invgamma
from sampler.EnKS import EnKS_Optimized
from tqdm import tqdm

class GibbsSampler: 
    def __init__(self, 
                 observations: NDArray[np.float64],      # Shape: (N, T)
                 neighbour_locs: NDArray[np.int64],      # Shape: (N, 5)
                 num_iterations: int,
                 burn_in: int,
                 thin: int,
                 alpha_init: float,
                 beta_init: float,
                 sigma_eta_sq_init: float,
                 sigma_nu_sq_init: float,
                 sigma_epsilon_sq_init: float,
                 nu_init: NDArray[np.float64],          # Shape: (T, 2)
                 state_init: NDArray[np.float64],       # Shape: (N, T)
                 prior_params: Dict[str, Dict[str, float]],
                 N_ensemble: int,
                 smoothing_window: int,
                 fixed_sigmas: bool = False,
                 generate_forecasts: bool = False,
                 forecast_steps: int = 12) -> None:
        self.observations = observations
        self.neighbour_locs = neighbour_locs
        self.num_iterations = num_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.alpha = alpha_init
        self.beta = beta_init
        self.sigma_eta_sq = sigma_eta_sq_init
        self.sigma_nu_sq = sigma_nu_sq_init
        self.sigma_epsilon_sq = sigma_epsilon_sq_init
        self.nu = nu_init.copy()
        self.state = state_init.copy()
        self.prior_params = prior_params
        self.N_ensemble = N_ensemble
        self.smoothing_window = smoothing_window

        self.num_saved_samples = (num_iterations - burn_in) // thin
        self.N = observations.shape[0]
        self.T = observations.shape[1]

        self.alpha_samples = np.zeros(self.num_saved_samples)
        self.beta_samples = np.zeros(self.num_saved_samples)
        self.sigma_eta_sq_samples = np.zeros(self.num_saved_samples)
        self.sigma_nu_sq_samples = np.zeros(self.num_saved_samples)
        self.sigma_epsilon_sq_samples = np.zeros(self.num_saved_samples)
        self.Y_samples = np.zeros((self.N, self.T+1, self.num_saved_samples))
        self.nu_samples = np.zeros((self.T+1, 2, self.num_saved_samples))

        # Arrays to store likelihood values
        self.log_complete_samples = np.zeros(self.num_saved_samples)
        self.log_obs_samples = np.zeros(self.num_saved_samples)

        self.fixed_sigmas = fixed_sigmas

        # Forecast states and observations
        if generate_forecasts:
            self.generate_forecasts = True
            self.forecast_steps = forecast_steps
            self.forecast_states_samples = np.zeros((self.N, forecast_steps, self.num_saved_samples))
            self.forecast_observations_samples = np.zeros((self.N, forecast_steps, self.num_saved_samples))
        else:
            self.generate_forecasts = False        

        self.self_idx = neighbour_locs[:, 0]
        self.left_idx = neighbour_locs[:, 1]
        self.right_idx = neighbour_locs[:, 2]
        self.up_idx = neighbour_locs[:, 3]
        self.down_idx = neighbour_locs[:, 4]

        # Print a complete summary of the sampler
        print(f"Sampler initialized with:")
        print(f"  - Number of iterations: {self.num_iterations}")
        print(f"  - Burn-in: {self.burn_in}")
        print(f"  - Thinning: {self.thin}")
        print(f"  - Number of saved samples: {self.num_saved_samples}")
        print(f"  - Number of observations: {self.N}")
        print(f"  - Number of time steps: {self.T}")
        print(f"  - Number of ensemble members: {self.N_ensemble}")
        print(f"  - Smoothing window: {self.smoothing_window}")
        print(f"  - Shapes of variables:")
        print(f"    - observations: {self.observations.shape}")
        print(f"    - neighbour_locs: {self.neighbour_locs.shape}")
        print(f"    - nu: {self.nu.shape}")
        print(f"    - state: {self.state.shape}")

        # --- Data Augmentation for Missing Observations ---
        self.augmented_observations = observations.copy()
        missing_mask = np.isnan(self.augmented_observations)
        self.augmented_observations[missing_mask] = np.random.normal(
            loc=self.state[:, 1:][missing_mask],
            scale=np.sqrt(self.sigma_epsilon_sq)
        )

    def rolling_forecast_prediction(self,
                                    initial_state: np.ndarray,
                                    initial_nu: np.ndarray,
                                    forecast_steps: int,
                                    neighbour_locs: np.ndarray,
                                    parameters: dict,
                                    num_samples: int = 12) -> dict:
        """
        Generate out-of-sample rolling forecast predictive samples via composition sampling.
        
        For each forecast sample, the function rolls the forecast one step ahead at a time.
        Given the last observed latent state (initial_state, shape (N,)) and the last advection
        parameter (initial_nu, shape (2,)), the forecast proceeds as follows for each sample:
        
        1. For t=1,...,forecast_steps:
            - Sample a new advection parameter: nu_new = alpha * current_nu + ω, where ω ~ N(0, sigma_nu_sq * I).
            - Forecast the latent state using the process model:
            
                    Y[i, t] = (1 - 4β) Y[i, t-1] 
                            + (β - nu_x) Y[right, t-1] 
                            + (β + nu_x) Y[left, t-1]
                            + (β - nu_y) Y[down, t-1]
                            + (β + nu_y) Y[up, t-1]
                            + εₜ,   εₜ ~ N(0, sigma_eta_sq)
            
            - Generate a forecast observation: Z[t] = Y[t] + ζ, with ζ ~ N(0, sigma_epsilon_sq).
            - Update the current state and nu.
        
        The function returns a dictionary with:
        - 'forecast_states': Array of shape (N, forecast_steps+1, num_samples) (including the initial state).
        - 'forecast_observations': Array of shape (N, forecast_steps, num_samples) (forecasts for times 1,...,forecast_steps).
        
        Parameters:
            initial_state (np.ndarray): Last observed latent state, shape (N,).
            initial_nu (np.ndarray): Last estimated advection parameter, shape (2,).
            forecast_steps (int): Number of steps to forecast.
            neighbour_locs (np.ndarray): Array of shape (N, 5) with neighbor indices [index, left, right, up, down].
            parameters (dict): Dictionary containing the model parameters with keys:
                'alpha', 'beta', 'sigma_eta_sq', 'sigma_epsilon_sq', 'sigma_nu_sq'.
            num_samples (int): Number of forecast samples to generate.
        
        Returns:
            dict: Dictionary containing:
                - 'forecast_states': np.ndarray of shape (N, forecast_steps+1, num_samples)
                - 'forecast_observations': np.ndarray of shape (N, forecast_steps, num_samples)
        """
        N = initial_state.shape[0]
        
        # Precompute neighbor indices from neighbour_locs.
        left_neighbors  = neighbour_locs[:, 1].astype(int)
        right_neighbors = neighbour_locs[:, 2].astype(int)
        up_neighbors    = neighbour_locs[:, 3].astype(int)
        down_neighbors  = neighbour_locs[:, 4].astype(int)
        
        # Arrays to hold forecast samples.
        forecast_states_samples = np.zeros((N, forecast_steps + 1, num_samples))
        forecast_obs_samples    = np.zeros((N, forecast_steps, num_samples))
        
        # For each forecast sample, roll the forecast one step at a time.
        for s in range(num_samples):
            current_state = initial_state.copy()   # shape (N,)
            current_nu    = initial_nu.copy()        # shape (2,)
            
            # Save the initial state.
            forecast_states = [current_state.copy()]  # will have forecast_steps+1 entries
            forecast_obs    = []  # will have forecast_steps entries
            
            # Roll forecast step by step.
            for t in range(1, forecast_steps + 1):
                # Update advection parameter: nu_new = alpha * current_nu + noise.
                noise_nu = np.random.multivariate_normal(mean=np.zeros(2),
                                                        cov=parameters['sigma_nu_sq'] * np.eye(2))
                new_nu = parameters['alpha'] * current_nu + noise_nu

                # Process noise for the state update.
                process_noise = np.random.normal(0, np.sqrt(parameters['sigma_eta_sq']), N)
                
                # Update the state using the process model.
                # (Using periodic neighbor indexing provided by neighbour_locs.)
                new_state = ((1 - 4 * parameters['beta']) * current_state +
                            (parameters['beta'] - new_nu[0]) * current_state[right_neighbors] +
                            (parameters['beta'] + new_nu[0]) * current_state[left_neighbors] +
                            (parameters['beta'] - new_nu[1]) * current_state[down_neighbors] +
                            (parameters['beta'] + new_nu[1]) * current_state[up_neighbors] +
                            process_noise)
                
                # Observation noise.
                observation_noise = np.random.normal(0, np.sqrt(parameters['sigma_epsilon_sq']), N)
                obs = new_state + observation_noise
                
                # Update current state and nu.
                current_state = new_state.copy()
                current_nu = new_nu.copy()
                
                # Save forecasts.
                forecast_states.append(current_state.copy())
                forecast_obs.append(obs.copy())
            
            # Convert lists to arrays and store in forecast samples.
            # forecast_states: (N, forecast_steps+1), forecast_obs: (N, forecast_steps)
            forecast_states_array = np.stack(forecast_states, axis=1)
            forecast_obs_array = np.stack(forecast_obs, axis=1)
            
            forecast_states_samples[:, :, s] = forecast_states_array
            forecast_obs_samples[:, :, s] = forecast_obs_array

        return {
            'forecast_states': forecast_states_samples,      # (N, forecast_steps+1, num_samples)
            'forecast_observations': forecast_obs_samples     # (N, forecast_steps, num_samples)
        }


    def sample(self) -> Dict[str, NDArray[np.float64]]:  # Returns dict with arrays of shapes:
                                                        # alpha_samples: (num_saved_samples,)
                                                        # beta_samples: (num_saved_samples,)
                                                        # sigma_eta_sq_samples: (num_saved_samples,)
                                                        # sigma_nu_sq_samples: (num_saved_samples,)
                                                        # sigma_epsilon_sq_samples: (num_saved_samples,)
                                                        # Y_samples: (N, T+1, num_saved_samples)
                                                        # nu_samples: (T+1, 2, num_saved_samples)
        for iter in tqdm(range(self.num_iterations), desc="Gibbs Sampling Progress", unit="iteration"):

            # # Sample Y
            # enks = EnKS(
            #     N_ensemble=self.N_ensemble,
            #     lags=self.smoothing_window,
            #     beta=self.beta,
            #     nu=self.nu,
            #     sigma_eta_sq=self.sigma_eta_sq,
            #     sigma_epsilon_sq=self.sigma_epsilon_sq,
            #     prior_params=self.prior_params
            # )
            # # Draw states from the EnKS
            # Y_analysis = enks.run(self.augmented_observations, self.neighbour_locs)

            Y_analysis = EnKS_Optimized(self.augmented_observations, 
                        self.neighbour_locs, 
                        self.N_ensemble, 
                        self.smoothing_window, 
                        self.beta,
                        self.nu,
                        self.sigma_eta_sq,
                        self.sigma_epsilon_sq,
                        self.prior_params['initial_state']['m_state'],
                        self.prior_params['initial_state']['v_state']
                        )

            self.state = Y_analysis[:,np.random.randint(0, self.N_ensemble),:]

            # Augment missing observations
            augmented_observations = self.observations.copy()
            missing_mask = np.isnan(augmented_observations)
            augmented_observations[missing_mask] = np.random.normal(
                loc=self.state[:, 1:][missing_mask],
                scale=np.sqrt(self.sigma_epsilon_sq)
            )

            self.augmented_observations = augmented_observations

            # Sample α (autoregressive parameter for the advection)
            S_xy = np.sum(np.einsum('ij,ij->i', self.nu[1:], self.nu[:-1]))
            S_xx = np.sum(np.einsum('ij,ij->i', self.nu[:-1], self.nu[:-1]))
            prec_alpha = (S_xx / self.sigma_nu_sq) + (1.0 / self.prior_params['autoregression']['v_alpha'])
            sigma_alpha_sq = 1.0 / prec_alpha

            mu_alpha = sigma_alpha_sq * ((S_xy / self.sigma_nu_sq) + 
                        (self.prior_params['autoregression']['m_alpha'] / self.prior_params['autoregression']['v_alpha']))

            a_trunc = (0 - mu_alpha) / np.sqrt(sigma_alpha_sq)
            b_trunc = (1 - mu_alpha) / np.sqrt(sigma_alpha_sq)

            self.alpha = truncnorm.rvs(a_trunc, b_trunc, loc=mu_alpha, scale=np.sqrt(sigma_alpha_sq))
            
            # Sample β (diffusion parameter)
            sum_Bt_sq = 0.0
            sum_At_Bt = 0.0

            for t in range(1, self.T):
                st_self_prev = self.state[self.self_idx, t - 1]
                st_left_prev = self.state[self.left_idx, t - 1]
                st_right_prev = self.state[self.right_idx, t - 1]
                st_up_prev = self.state[self.up_idx, t - 1]
                st_down_prev = self.state[self.down_idx, t - 1]

                # Define X_{t,i}
                B_t = st_left_prev + st_right_prev + st_up_prev + st_down_prev - 4.0 * st_self_prev

                # Define the response:
                A_t = (self.state[self.self_idx, t] - st_self_prev 
                    - self.nu[t - 1, 0] * (st_right_prev - st_left_prev)
                    - self.nu[t - 1, 1] * (st_down_prev - st_up_prev))

                sum_Bt_sq += B_t.T @ B_t
                sum_At_Bt += A_t @ B_t

            a_beta = (1.0 / self.sigma_eta_sq) * sum_Bt_sq + 1 / self.prior_params['diffusion']['v_beta']
            b_beta = (1.0 / self.sigma_eta_sq) * sum_At_Bt + self.prior_params['diffusion']['m_beta'] / self.prior_params['diffusion']['v_beta']

            mu_beta = b_beta / a_beta
            sigma_beta_sq = 1 / a_beta

            self.beta = np.random.normal(mu_beta, np.sqrt(sigma_beta_sq))


            # Sample advection parameters v_{0:T}
            # v_0
            A_vec_0 = (self.state[self.self_idx, 1] - 
                      (1.0 - 4.0 * self.beta) * self.state[self.self_idx, 0] - 
                      self.beta * (self.state[self.left_idx, 0] + 
                                 self.state[self.right_idx, 0] + 
                                 self.state[self.up_idx, 0] + 
                                 self.state[self.down_idx, 0]))
            B_vec_0 = self.state[self.left_idx, 0] - self.state[self.right_idx, 0]
            C_vec_0 = self.state[self.up_idx, 0] - self.state[self.down_idx, 0]
            R_mat_0 = np.column_stack((B_vec_0, C_vec_0))
            prec_fcd_v_0 = (1.0/self.sigma_eta_sq) * R_mat_0.T @ R_mat_0 + (self.alpha**2/self.sigma_nu_sq) * np.eye(2)
            cov_fcd_v_0 = np.linalg.inv(prec_fcd_v_0)
            mean_fcd_v_0 = cov_fcd_v_0 @ ((1.0/self.sigma_eta_sq) * R_mat_0.T @ A_vec_0[:, np.newaxis] + (self.alpha / self.sigma_nu_sq) * self.nu[1, :][:, np.newaxis])
            self.nu[0, :] = np.random.multivariate_normal(mean_fcd_v_0.flatten(), cov_fcd_v_0)

            # v_{1:T-1}
            for t in range(1, self.T):
                A_vec_t = self.state[self.self_idx, t + 1] - (1.0 - 4.0 * self.beta) * self.state[self.self_idx, t] - self.beta * (
                    self.state[self.left_idx, t] + self.state[self.right_idx, t] + self.state[self.up_idx, t] + self.state[self.down_idx, t])
                B_vec_t = self.state[self.left_idx, t] - self.state[self.right_idx, t]
                C_vec_t = self.state[self.up_idx, t] - self.state[self.down_idx, t]
                R_mat_t = np.column_stack((B_vec_t, C_vec_t))

                prec_fcd_v_t = (1.0/self.sigma_eta_sq) * R_mat_t.T @ R_mat_t
                prec_fcd_v_t += ((1 + self.alpha**2)/self.sigma_nu_sq) * np.eye(2)
                cov_fcd_v_t = np.linalg.inv(prec_fcd_v_t)

                mean_fcd_v_t = cov_fcd_v_t @ ((1.0/self.sigma_eta_sq) * R_mat_t.T @ A_vec_t[:, np.newaxis] +
                                              (self.alpha / self.sigma_nu_sq) * self.nu[t-1, :][:, np.newaxis] +
                                              self.nu[t+1, :][:, np.newaxis])
                self.nu[t, :] = np.random.multivariate_normal(mean_fcd_v_t.flatten(), cov_fcd_v_t)

            # v_{T}
            self.nu[self.T, :] = np.random.multivariate_normal(self.alpha * self.nu[self.T-1, :], self.sigma_nu_sq * np.eye(2))

            # Sample σ²_η and σ²_ε            
            if self.fixed_sigmas:
                self.sigma_eta_sq = self.sigma_eta_sq
                self.sigma_epsilon_sq = self.sigma_epsilon_sq

            else:
                # Sample σ²_η (process error variance)
                rss_eta = 0.0
                for t in range(1, self.T):
                    st_self_prev  = self.state[self.self_idx, t - 1]
                    st_left_prev  = self.state[self.left_idx, t - 1]
                    st_right_prev = self.state[self.right_idx, t - 1]
                    st_up_prev    = self.state[self.up_idx, t - 1]
                    st_down_prev  = self.state[self.down_idx, t - 1]
                    
                    Y_pred = ((1 - 4 * self.beta) * st_self_prev +
                            (self.beta - self.nu[t - 1, 0]) * st_left_prev +   # left neighbor
                            (self.beta + self.nu[t - 1, 0]) * st_right_prev +  # right neighbor
                            (self.beta - self.nu[t - 1, 1]) * st_up_prev +
                            (self.beta + self.nu[t - 1, 1]) * st_down_prev)

                    residual = self.state[self.self_idx, t] - Y_pred
                    rss_eta += np.sum(residual**2)

                a_eta_post = self.prior_params['process']['a_eta'] + 0.5 * self.N * (self.T - 1)
                b_eta_post = self.prior_params['process']['b_eta'] + 0.5 * rss_eta

                self.sigma_eta_sq = invgamma.rvs(a_eta_post, scale=b_eta_post)
                # Sample σ²_ε

                residuals_epsilon = []

                for t in range(self.T):
                    residuals_epsilon_t = self.augmented_observations[:, t] - self.state[:, t]
                    residuals_epsilon.append(residuals_epsilon_t.T @ residuals_epsilon_t)

                a_epsilon = self.prior_params['observation']['a_epsilon'] + 0.5 * (self.N * (self.T))
                b_epsilon = self.prior_params['observation']['b_epsilon'] + 0.5 * np.sum(residuals_epsilon)

                self.sigma_epsilon_sq = invgamma.rvs(a_epsilon, scale=b_epsilon)

            # Sample σ²_ν (advection error variance)
            residuals_nu = []
            for t in range(1, self.T):
                residuals_nu_t = self.nu[t] - self.alpha * self.nu[t - 1]
                residuals_nu.append(residuals_nu_t.T @ residuals_nu_t)

            a_nu = self.prior_params['advection']['a_nu'] + (self.T - 1)
            b_nu = self.prior_params['advection']['b_nu'] + 0.5 * np.sum(residuals_nu)

            self.sigma_nu_sq = invgamma.rvs(a=a_nu, scale=b_nu)


            # --- Compute and Save Log Likelihoods ---
            # Log complete-data likelihood:
            log_complete = -0.5 * self.N * self.T * np.log(2 * np.pi * self.sigma_epsilon_sq)
            resid_complete = self.augmented_observations - self.state[:, 1:]
            log_complete -= 0.5 / self.sigma_epsilon_sq * np.sum(resid_complete**2)
            
            # Log observed-data likelihood (only for observed entries)
            missing_mask = np.isnan(self.observations)
            observed_mask = ~missing_mask
            resid_obs = (self.observations - self.state[:, 1:])[observed_mask]
            log_obs = -0.5 * np.sum(np.log(2 * np.pi * self.sigma_epsilon_sq) + (resid_obs**2) / self.sigma_epsilon_sq)

            # Save samples after burn-in and thinning
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                idx = (iter - self.burn_in) // self.thin
                self.alpha_samples[idx] = self.alpha
                self.beta_samples[idx] = self.beta
                self.sigma_eta_sq_samples[idx] = self.sigma_eta_sq
                self.sigma_nu_sq_samples[idx] = self.sigma_nu_sq
                self.sigma_epsilon_sq_samples[idx] = self.sigma_epsilon_sq
                self.Y_samples[:, :, idx] = self.state
                self.nu_samples[:, :, idx] = self.nu
                self.log_complete_samples[idx] = log_complete
                self.log_obs_samples[idx] = log_obs

            if self.generate_forecasts and iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                forecast_results = self.rolling_forecast_prediction(
                    initial_state=self.state[:, -1],
                    initial_nu=self.nu[-1],
                    forecast_steps=self.forecast_steps,
                    neighbour_locs=self.neighbour_locs,
                    parameters={
                        'alpha': self.alpha,
                        'beta': self.beta,
                        'sigma_eta_sq': self.sigma_eta_sq,
                        'sigma_epsilon_sq': self.sigma_epsilon_sq,
                        'sigma_nu_sq': self.sigma_nu_sq
                    },
                    num_samples=self.forecast_steps
                )

                forecast_states = forecast_results['forecast_states']
                forecast_observations = forecast_results['forecast_observations']
        
        # Reshape nu_samples to (2, T+1, num_saved_samples)
        self.nu_samples = self.nu_samples.transpose(1, 0, 2)

        return {
            'alpha_samples': self.alpha_samples, # (num_saved_samples,)
            'beta_samples': self.beta_samples, # (num_saved_samples,)
            'sigma_eta_sq_samples': self.sigma_eta_sq_samples, # (num_saved_samples,)
            'sigma_nu_sq_samples': self.sigma_nu_sq_samples, # (num_saved_samples,)
            'sigma_epsilon_sq_samples': self.sigma_epsilon_sq_samples, # (num_saved_samples,)
            'Y_samples': self.Y_samples, # (N, T+1, num_saved_samples)
            'nu_samples': self.nu_samples, # (2, T+1, num_saved_samples)
            'log_complete_samples': self.log_complete_samples, # (num_saved_samples,)
            'log_obs_samples': self.log_obs_samples, # (num_saved_samples,)
            'forecast_states': self.forecast_states_samples, # (N, forecast_steps+1, num_saved_samples)
            'forecast_observations': self.forecast_observations_samples # (N, forecast_steps, num_saved_samples)
        }