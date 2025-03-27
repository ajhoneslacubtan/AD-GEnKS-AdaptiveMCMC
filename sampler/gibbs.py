import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any
from scipy.stats import truncnorm, invgamma, norm
from sampler.EnKS import EnKS_Optimized
from tqdm import tqdm
import gc
import zarr
from utils.logging_utils import setup_logger

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
                 nu_init: NDArray[np.float64],          # Shape: (T+1, 2)
                 state_init: NDArray[np.float64],       # Shape: (N, T+1)
                 prior_params: Dict[str, Dict[str, float]],
                 N_ensemble: int,
                 smoothing_window: int,
                 fixed_sigmas: bool = False,
                 beta_sampling_method: str = "normal"
                 ) -> None:    # choose "normal" or "adaptive"
        # Set up logger
        self.logger = setup_logger("GibbsSampler", "logs/gibbs_sampler.log")
        
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

        self.fixed_sigmas = fixed_sigmas      

        self.self_idx = neighbour_locs[:, 0]
        self.left_idx = neighbour_locs[:, 1]
        self.right_idx = neighbour_locs[:, 2]
        self.up_idx = neighbour_locs[:, 3]
        self.down_idx = neighbour_locs[:, 4]

        # Option for sampling β:
        self.beta_sampling_method = beta_sampling_method  # "normal" or "adaptive"
        if self.beta_sampling_method == "adaptive":
            _scale = 1  # scale for the proposal distribution
            self._beta_proposal_var = _scale * 9.740275359702301e-08  # fixed proposal std for RWMH during burn-in
            self._beta_trials = 0         # count total proposals
            self._beta_accepted = 0       # count accepted proposals
            self._beta_burn_in_samples = []  # collect burn-in samples for β

        self.logger.info("GibbsSampler initialized with:")
        self.logger.info(f"  - Number of iterations: {self.num_iterations}")
        self.logger.info(f"  - Burn-in: {self.burn_in}")
        self.logger.info(f"  - Thinning: {self.thin}")
        self.logger.info(f"  - Number of saved samples: {self.num_saved_samples}")
        self.logger.info(f"  - Number of observations: {self.N}")
        self.logger.info(f"  - Number of time steps: {self.T}")
        self.logger.info(f"  - Number of ensemble members: {self.N_ensemble}")
        self.logger.info(f"  - Smoothing window: {self.smoothing_window}")
        self.logger.info(f"  - Shapes of variables:")
        self.logger.info(f"    - observations: {self.observations.shape}")
        self.logger.info(f"    - neighbour_locs: {self.neighbour_locs.shape}")
        self.logger.info(f"    - nu: {self.nu.shape}")
        self.logger.info(f"    - state: {self.state.shape}")

        # --- Data Augmentation for Missing Observations ---
        self.augmented_observations = observations.copy()
        missing_mask = np.isnan(self.augmented_observations)
        self.augmented_observations[missing_mask] = np.random.normal(
            loc=self.state[:, 1:][missing_mask],
            scale=np.sqrt(self.sigma_epsilon_sq)
        )

    # ------------------------------------------------------------
    # PRIVATE METHOD: Compute the log full conditional for β.
    # This is based on the model:
    #   Y_t,i ~ N( μ_{t,i}(β), σ²_η ),
    # with
    #   μ_{t,i}(β) = Y_{t-1,i} + β * B_{t,i} + ν-part,
    # where
    #   B_{t,i} = (Y_{t-1,j1} + Y_{t-1,j2} + Y_{t-1,j3} + Y_{t-1,j4} - 4 Y_{t-1,i}).
    # The log posterior (ignoring additive constants and prior, which is Uniform(0,0.25))
    # is proportional to:
    #   - 1/(2σ²_η) ∑_{t=1}^{T-1} (A_t - β B_t)^2, with the constraint β ∈ (0, 0.25).
    # ------------------------------------------------------------
    def _log_posterior_beta(self, beta: float) -> float:
        # Enforce the support of Uniform(0, 0.25)
        if beta <= 0 or beta >= 0.25:
            return -np.inf

        # We assume self.state has shape (N, T+1) and self.nu has shape (T+1, 2).
        # The indices self.self_idx, self.left_idx, etc., have length N.
        # For time steps t = 1, ..., T-1, we extract the previous states.
        # These slices will have shape (N, T-1).
        st_self_prev  = self.state[self.self_idx, :self.T-1]  
        st_left_prev  = self.state[self.left_idx, :self.T-1]
        st_right_prev = self.state[self.right_idx, :self.T-1]
        st_up_prev    = self.state[self.up_idx, :self.T-1]
        st_down_prev  = self.state[self.down_idx, :self.T-1]

        # For the current time steps t = 1, ..., T-1
        st_self_curr = self.state[self.self_idx, 1:self.T]

        # For the velocity process, we use self.nu[t-1] for t=1,...,T-1.
        # Reshape so that nu_x and nu_y have shape (1, T-1) and can broadcast over space.
        nu_x = self.nu[:self.T-1, 0].reshape(1, -1)
        nu_y = self.nu[:self.T-1, 1].reshape(1, -1)

        # Compute the "A" term for all locations and time steps:
        A = st_self_curr - st_self_prev \
            - nu_x * (st_right_prev - st_left_prev) \
            - nu_y * (st_down_prev - st_up_prev)

        # Compute the "B" term:
        B = st_left_prev + st_right_prev + st_up_prev + st_down_prev - 4.0 * st_self_prev

        # The squared error at each (location, time):
        sq_err = (A - beta * B) ** 2

        # Sum over all spatial locations and time steps.
        sse = np.sum(sq_err)

        loglike = -sse / (2 * self.sigma_eta_sq)
        return loglike

    # ------------------------------------------------------------
    # PRIVATE METHOD: RWMH update for β during burn-in.
    # ------------------------------------------------------------
    def _sample_beta_rwmh(self, current_beta: float) -> float:
        proposal = current_beta + np.random.normal(0, np.sqrt(self._beta_proposal_var))
        # Count each proposal
        self._beta_trials += 1
        # Enforce support of Uniform(0, 0.25)
        if proposal <= 0 or proposal >= 0.25:
            return current_beta
        log_post_current = self._log_posterior_beta(current_beta)
        log_post_proposal = self._log_posterior_beta(proposal)
        acceptance_prob = min(1, np.exp(log_post_proposal - log_post_current))
        if np.random.rand() < acceptance_prob:
            self._beta_accepted += 1
            
            return proposal
        else:
            return current_beta

    # ------------------------------------------------------------
    # PRIVATE METHOD: Independent MH update for β after burn-in.
    # Uses an independent Normal proposal with mean and variance from burn-in.
    # ------------------------------------------------------------
    def _sample_beta_indep(self, current_beta: float, mu_beta: float, var_beta: float) -> float:
        proposal = np.random.normal(mu_beta, np.sqrt(var_beta))
        # Enforce support
        if proposal <= 0 or proposal >= 0.25:
            return current_beta
        # Compute proposal densities:
        g_current = norm.pdf(current_beta, loc=mu_beta, scale=np.sqrt(var_beta))
        g_proposal = norm.pdf(proposal, loc=mu_beta, scale=np.sqrt(var_beta))
        log_post_current = self._log_posterior_beta(current_beta)
        log_post_proposal = self._log_posterior_beta(proposal)
        acceptance_prob = min(1, np.exp(log_post_proposal - log_post_current) * (g_current / g_proposal))
        if np.random.rand() < acceptance_prob:
            return proposal
        else:
            return current_beta

    def sample(self) -> Dict[str, Any]:
        """
        Runs the Gibbs sampling procedure, using zarr only for large arrays.
        """
        self.logger.info("Starting Gibbs sampling")
        
        # Initialize in-memory arrays for smaller parameters
        self.logger.debug("Initializing sample storage arrays")
        alpha_samples = np.zeros(self.num_saved_samples, dtype=np.float32)
        beta_samples = np.zeros(self.num_saved_samples, dtype=np.float32)
        sigma_eta_sq_samples = np.zeros(self.num_saved_samples, dtype=np.float32)
        sigma_nu_sq_samples = np.zeros(self.num_saved_samples, dtype=np.float32)
        sigma_epsilon_sq_samples = np.zeros(self.num_saved_samples, dtype=np.float32)
        log_complete_samples = np.zeros(self.num_saved_samples, dtype=np.float32)
        
        # Create zarr arrays only for large state variables
        store = zarr.open('gibbs_results.zarr', mode='w')
        Y_samples = store.create_array('Y_samples', 
                                    shape=(self.N, self.T+1, self.num_saved_samples),
                                    chunks=(self.N, min(100, self.T+1), min(100, self.num_saved_samples)),
                                    dtype=np.float32)
        nu_samples = store.create_array('nu_samples',
                                    shape=(2, self.T+1, self.num_saved_samples),
                                    chunks=(2, min(100, self.T+1), min(100, self.num_saved_samples)),
                                    dtype=np.float32)
        
        sample_idx = 0
        
        for iter in tqdm(range(self.num_iterations), desc="Gibbs Sampling Progress", unit="iteration"):
            if iter % 100 == 0:
                self.logger.debug(f"Iteration {iter}/{self.num_iterations}")
            
            # Sample Y via EnKS_Optimized
            self.logger.debug("Running EnKS optimization")
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
            # Update state using a random ensemble member
            self.state = Y_analysis[:, np.random.randint(0, self.N_ensemble), :].copy()
            del Y_analysis
            gc.collect()
            
            # Augment missing observations and free temporary array
            augmented_observations = self.observations.copy()
            missing_mask = np.isnan(augmented_observations)
            augmented_observations[missing_mask] = np.random.normal(
                loc=self.state[:, 1:][missing_mask],
                scale=np.sqrt(self.sigma_epsilon_sq)
            )
            self.augmented_observations = augmented_observations
            del augmented_observations
            gc.collect()
            
            # Sample α (autoregressive parameter for advection)
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
            if self.beta_sampling_method == "normal":
                sum_Bt_sq = 0.0
                sum_At_Bt = 0.0
                for t in range(1, self.T):
                    st_self_prev = self.state[self.self_idx, t - 1]
                    st_left_prev = self.state[self.left_idx, t - 1]
                    st_right_prev = self.state[self.right_idx, t - 1]
                    st_up_prev = self.state[self.up_idx, t - 1]
                    st_down_prev = self.state[self.down_idx, t - 1]
                    B_t = st_left_prev + st_right_prev + st_up_prev + st_down_prev - 4.0 * st_self_prev
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
            else:
                # "adaptive" method using the two-phase approach.
                if iter < self.burn_in:
                    # Burn-in phase: update β via RWMH and collect samples.
                    self.beta = self._sample_beta_rwmh(self.beta)
                    self._beta_burn_in_samples.append(self.beta)
                    # Calculate proposal parameters once when burn-in ends
                    if iter == self.burn_in - 1:
                        burn_in_array = np.array(self._beta_burn_in_samples)
                        self._proposal_mu = burn_in_array.mean()
                        self._proposal_var = burn_in_array.var()
                        # Print acceptance rate and proposal parameters
                        acceptance_rate = self._beta_accepted / self._beta_trials if self._beta_trials > 0 else 0.0
                        self.logger.info(f"Adaptive β sampling: Acceptance rate during burn-in: {acceptance_rate:.4f}")
                        self.logger.info(f"Adaptive β sampling: Proposal mu = {self._proposal_mu:.4f}, Proposal var = {self._proposal_var}")
                        # Free memory
                        del self._beta_burn_in_samples
                        gc.collect()
                else:
                    # After burn-in: use pre-computed mean and variance
                    # Sample β using independent proposal
                    self.beta = self._sample_beta_indep(self.beta, self._proposal_mu, self._proposal_var)
            
            # Sample advection parameters (nu)
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
            mean_fcd_v_0 = cov_fcd_v_0 @ ((1.0/self.sigma_eta_sq) * R_mat_0.T @ A_vec_0[:, np.newaxis] + 
                                          (self.alpha / self.sigma_nu_sq) * self.nu[1, :][:, np.newaxis])
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
                                              (self.alpha / self.sigma_nu_sq) * self.nu[t+1, :][:, np.newaxis])
                self.nu[t, :] = np.random.multivariate_normal(mean_fcd_v_t.flatten(), cov_fcd_v_t)
            
            # v_T
            self.nu[self.T, :] = np.random.multivariate_normal(self.alpha * self.nu[self.T-1, :], self.sigma_nu_sq * np.eye(2))
            
            # Sample σ²_η and σ²_ε            
            if not self.fixed_sigmas:
                rss_eta = 0.0
                for t in range(1, self.T):
                    st_self_prev  = self.state[self.self_idx, t - 1]
                    st_left_prev  = self.state[self.left_idx, t - 1]
                    st_right_prev = self.state[self.right_idx, t - 1]
                    st_up_prev    = self.state[self.up_idx, t - 1]
                    st_down_prev  = self.state[self.down_idx, t - 1]
                    Y_pred = ((1 - 4 * self.beta) * st_self_prev +
                            (self.beta - self.nu[t - 1, 0]) * st_left_prev +
                            (self.beta + self.nu[t - 1, 0]) * st_right_prev +
                            (self.beta - self.nu[t - 1, 1]) * st_up_prev +
                            (self.beta + self.nu[t - 1, 1]) * st_down_prev)
                    residual = self.state[self.self_idx, t] - Y_pred
                    rss_eta += np.sum(residual**2)
                a_eta_post = self.prior_params['process']['a_eta'] + 0.5 * self.N * (self.T - 1)
                b_eta_post = self.prior_params['process']['b_eta'] + 0.5 * rss_eta
                self.sigma_eta_sq = invgamma.rvs(a_eta_post, scale=b_eta_post)
                
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
            self.sigma_nu_sq = invgamma.rvs(a=a_nu, scale=b_nu)
            
            # Compute log complete likelihood
            log_complete = -0.5 * self.N * self.T * np.log(2 * np.pi * self.sigma_epsilon_sq)
            resid_complete = self.augmented_observations - self.state[:, 1:]
            log_complete -= 0.5 / self.sigma_epsilon_sq * np.sum(resid_complete**2)
            
            # Save samples after burn-in and thinning
            if iter >= self.burn_in and (iter - self.burn_in) % self.thin == 0:
                self.logger.debug(f"Saving samples for iteration {iter}")
                idx = sample_idx
                # Save small arrays to memory
                alpha_samples[idx] = self.alpha
                beta_samples[idx] = self.beta
                sigma_eta_sq_samples[idx] = self.sigma_eta_sq
                sigma_nu_sq_samples[idx] = self.sigma_nu_sq
                sigma_epsilon_sq_samples[idx] = self.sigma_epsilon_sq
                log_complete_samples[idx] = log_complete
                
                # Save large arrays to zarr
                Y_samples[:, :, idx] = self.state.astype(np.float32)
                nu_samples[:, :, idx] = self.nu.transpose(1, 0).astype(np.float32)
                
                sample_idx += 1
            
            # Clean up temporary arrays and run garbage collection
            gc.collect()
        
        self.logger.info("Gibbs sampling completed")
        # Return both in-memory samples and zarr paths
        result = {
            'alpha_samples': alpha_samples,
            'beta_samples': beta_samples,
            'sigma_eta_sq_samples': sigma_eta_sq_samples,
            'sigma_nu_sq_samples': sigma_nu_sq_samples,
            'sigma_epsilon_sq_samples': sigma_epsilon_sq_samples,
            'log_complete_samples': log_complete_samples,
            'Y_samples': 'gibbs_results.zarr/Y_samples',
            'nu_samples': 'gibbs_results.zarr/nu_samples'
        }
        
        return result
