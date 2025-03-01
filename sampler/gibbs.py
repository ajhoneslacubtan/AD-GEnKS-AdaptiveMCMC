import numpy as np
from scipy.stats import truncnorm, invgamma
from sampler.EnKS import EnKS
from tqdm import tqdm
from evaluation.evaluate_EnKS import evaluate_enks_sample

class GibbsSampler: 
    def __init__(self, observations, neighbour_locs, num_iterations, burn_in, thin, alpha_init, beta_init,
                 sigma_eta_sq_init, sigma_nu_sq_init, sigma_epsilon_sq_init, nu_init, state_init, prior_params,
                 N_ensemble, smoothing_window, fixed_sigmas=False):
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
        self.Y_samples = np.zeros((self.N, self.T, self.num_saved_samples))
        self.nu_samples = np.zeros((self.T, 2, self.num_saved_samples))

        self.fixed_sigmas = fixed_sigmas

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


    def sample(self):
        for iter in tqdm(range(self.num_iterations), desc="Gibbs Sampling Progress", unit="iteration"):

            # Sample Y
            enks = EnKS(
                N_ensemble=self.N_ensemble,
                lags=self.smoothing_window,
                beta=self.beta,
                nu=self.nu,
                sigma_eta_sq=self.sigma_eta_sq,
                sigma_epsilon_sq=self.sigma_epsilon_sq,
                prior_params=self.prior_params
            )
            # Draw states from the EnKS
            Y_analysis = enks.run(self.augmented_observations, self.neighbour_locs)

            self.state = Y_analysis[:,np.random.randint(0, self.N_ensemble),:]

            # Augment missing observations
            augmented_observations = self.observations.copy()
            missing_mask = np.isnan(augmented_observations)
            augmented_observations[missing_mask] = np.random.normal(
                loc=self.state[:, 1:][missing_mask],
                scale=np.sqrt(self.sigma_epsilon_sq)
            )

            self.augmented_observations = augmented_observations

            # Sample α
            # Compute S1 and S2 properly:
            S1 = np.sum(np.einsum('ij,ij->i', self.nu[1:], self.nu[:-1]))
            S2 = np.sum(np.einsum('ij,ij->i', self.nu[:-1], self.nu[:-1]))

            # Compute the precision (inverse variance) and variance:
            prec_alpha = (S2 / self.sigma_nu_sq) + (1.0 / self.prior_params['autoregression']['v_alpha'])
            sigma_alpha_sq = 1.0 / prec_alpha

            # Compute the mean:
            mu_alpha = sigma_alpha_sq * ((S1 / self.sigma_nu_sq) + 
                        (self.prior_params['autoregression']['m_alpha'] / self.prior_params['autoregression']['v_alpha']))


            # Sample from the truncated normal on (0,1)
            self.alpha = truncnorm.rvs((0 - mu_alpha) / np.sqrt(sigma_alpha_sq), 
                                    (1 - mu_alpha) / np.sqrt(sigma_alpha_sq),
                                    loc=mu_alpha, scale=np.sqrt(sigma_alpha_sq))
            
            # Sample β
            sum_Bt_sq = 0.0
            sum_At_Bt = 0.0

            for t in range(1, self.T):
                st_self_prev = self.state[self.self_idx, t - 1]
                st_left_prev = self.state[self.left_idx, t - 1]
                st_right_prev = self.state[self.right_idx, t - 1]
                st_up_prev = self.state[self.up_idx, t - 1]
                st_down_prev = self.state[self.down_idx, t - 1]

                # Define X_{t,i} = -4Y_{t-1,i} + Y_{t-1,left} + Y_{t-1,right} + Y_{t-1,up} + Y_{t-1,down}
                B_t = st_left_prev + st_right_prev + st_up_prev + st_down_prev - 4.0 * st_self_prev

                # Define the response as: 
                # Y_{t,i} - Y_{t-1,i} - nu_x[t-1]*(Y_{t-1,right} - Y_{t-1,left])
                #           - nu_y[t-1]*(Y_{t-1,down} - Y_{t-1,up])
                A_t = (self.state[self.self_idx, t] - st_self_prev 
                    - self.nu[t - 1, 0] * (st_right_prev - st_left_prev)
                    - self.nu[t - 1, 1] * (st_down_prev - st_up_prev))

                sum_Bt_sq += B_t.T @ B_t
                sum_At_Bt += A_t @ B_t

            # Compute the posterior precision and mean for beta
            a_beta = (1.0 / self.sigma_eta_sq) * sum_Bt_sq + 1 / self.prior_params['diffusion']['v_beta']
            b_beta = (1.0 / self.sigma_eta_sq) * sum_At_Bt + self.prior_params['diffusion']['m_beta'] / self.prior_params['diffusion']['v_beta']

            mu_beta = b_beta / a_beta
            sigma_beta_sq = 1 / a_beta

            self.beta = np.random.normal(mu_beta, np.sqrt(sigma_beta_sq))


            # Sample advection v_{0:T}
            # v_0
            A_vec_0 = (self.state[self.self_idx, 1] - 
                      (1.0 - 4.0 * self.beta) * self.state[self.self_idx, 0] - 
                      self.beta * (self.state[self.left_idx, 0] + 
                                 self.state[self.right_idx, 0] + 
                                 self.state[self.up_idx, 0] + 
                                 self.state[self.down_idx, 0]))
            B_vec_0 = self.state[self.left_idx, 0] - self.state[self.right_idx, 0] # Shape: (N,)
            C_vec_0 = self.state[self.up_idx, 0] - self.state[self.down_idx, 0] # Shape: (N,)
            R_mat_0 = np.column_stack((B_vec_0, C_vec_0)) # Shape: (N, 2)
            prec_fcd_v_0 = (1.0/self.sigma_eta_sq) * R_mat_0.T @ R_mat_0 + (self.alpha**2/self.sigma_nu_sq) * np.eye(2)
            cov_fcd_v_0 = np.linalg.inv(prec_fcd_v_0) # Shape: (2, 2)
            mean_fcd_v_0 = cov_fcd_v_0 @ ((1.0/self.sigma_eta_sq) * R_mat_0.T @ A_vec_0[:, np.newaxis] + (self.alpha / self.sigma_nu_sq) * self.nu[1, :][:, np.newaxis]) # Shape: (2, 1)
            self.nu[0, :] = np.random.multivariate_normal(mean_fcd_v_0.flatten(), cov_fcd_v_0)

            # v_{1:T-1}
            for t in range(1, self.T):
                A_vec_t = self.state[self.self_idx, t + 1] - (1.0 - 4.0 * self.beta) * self.state[self.self_idx, t] - self.beta * (self.state[self.left_idx, t] + self.state[self.right_idx, t] + self.state[self.up_idx, t] + self.state[self.down_idx, t]) # Shape: (N,)
                B_vec_t = self.state[self.left_idx, t] - self.state[self.right_idx, t] # Shape: (N,)
                C_vec_t = self.state[self.up_idx, t] - self.state[self.down_idx, t] # Shape: (N,)
                R_mat_t = np.column_stack((B_vec_t, C_vec_t)) # Shape: (N, 2)

                prec_fcd_v_t = (1.0/self.sigma_eta_sq) * R_mat_t.T @ R_mat_t
                prec_fcd_v_t += ((1 + self.alpha**2)/self.sigma_nu_sq) * np.eye(2)
                cov_fcd_v_t = np.linalg.inv(prec_fcd_v_t)

                mean_fcd_v_t = cov_fcd_v_t @ ( (1.0/self.sigma_eta_sq) * R_mat_t.T @ A_vec_t[:, np.newaxis] + (self.alpha / self.sigma_nu_sq) * self.nu[t-1, :][:, np.newaxis] + self.nu[t+1, :][:, np.newaxis] )
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

            # Sample σ²_ν

            residuals_nu = []

            for t in range(1, self.T):
                residuals_nu_t = self.nu[t] - self.alpha * self.nu[t - 1]
                residuals_nu.append(residuals_nu_t.T @ residuals_nu_t)

            a_nu = self.prior_params['advection']['a_nu'] + (self.T - 1)
            b_nu = self.prior_params['advection']['b_nu'] + 0.5 * np.sum(residuals_nu)

            self.sigma_nu_sq = invgamma.rvs(a_nu, scale=b_nu)


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

        return {
            'alpha_samples': self.alpha_samples,
            'beta_samples': self.beta_samples,
            'sigma_eta_sq_samples': self.sigma_eta_sq_samples,
            'sigma_nu_sq_samples': self.sigma_nu_sq_samples,
            'sigma_epsilon_sq_samples': self.sigma_epsilon_sq_samples,
            'Y_samples': self.Y_samples,
            'nu_samples': self.nu_samples
        }