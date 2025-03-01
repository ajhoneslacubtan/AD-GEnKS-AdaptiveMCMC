import numpy as np

def compute_mspe(Z: np.ndarray, Zhat: np.ndarray) -> float:
    """
    Compute Mean Squared Prediction Error (MSPE).

    Parameters:
        Z (np.ndarray): Observed data array of shape (N, T)
        Zhat (np.ndarray): Point predictions of shape (N, T)

    Returns:
        float: MSPE
    """
    return np.mean((Z - Zhat) ** 2)

def compute_rmspe(Z: np.ndarray, Zhat: np.ndarray) -> float:
    """
    Compute Root Mean Squared Prediction Error (RMSPE).

    Parameters:
        Z (np.ndarray): Observed data array of shape (N, T)
        Zhat (np.ndarray): Point predictions of shape (N, T)

    Returns:
        float: RMSPE
    """
    return np.sqrt(compute_mspe(Z, Zhat))

def compute_mape(Z: np.ndarray, Zhat: np.ndarray) -> float:
    """
    Compute Mean Absolute Prediction Error (MAPE).

    Parameters:
        Z (np.ndarray): Observed data array of shape (N, T)
        Zhat (np.ndarray): Point predictions of shape (N, T)

    Returns:
        float: MAPE
    """
    return np.mean(np.abs(Z - Zhat))

def compute_spatial_validation_metrics(Z: np.ndarray, Zhat_ens: np.ndarray):
    """
    Compute spatial validation statistics at each spatial location s_i.

    Given:
        - Z: observed data with shape (N, T)
        - Zhat_ens: ensemble predictions with shape (N, num_samples, T)

    We first form the ensemble mean:
        Zhat_mean(s_i, t) = average over ensemble of \hat{Z}(s_i, t)
    and compute the ensemble variance at each space--time location.
    
    Then, for each spatial index i, we compute:
    
    1. Normalized bias (V_bias):
       \[
       V_{bias}(s_i) = \frac{\frac{1}{T}\sum_{t=1}^T \left\{Z(s_i;t)-\hat{Z}_{mean}(s_i;t)\right\}}
         {\sqrt{\frac{1}{T}\sum_{t=1}^T \hat{V}(s_i;t)}}
       \]
       
    2. Normalized RMSE (V_normRMSE):
       \[
       V_{normRMSE}(s_i) = \sqrt{\frac{\frac{1}{T}\sum_{t=1}^T \left\{Z(s_i;t)-\hat{Z}_{mean}(s_i;t)\right\}^{2}}
       {\frac{1}{T}\sum_{t=1}^T \hat{V}(s_i;t)}}
       \]
       
    3. Unstandardized spatial RMSE (V3):
       \[
       V_3(s_i) = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\left\{ Z(s_i;t)-\hat{Z}_{mean}(s_i;t) \right\}^2}
       \]

    where \(\hat{V}(s_i;t)\) is the ensemble variance at \((s_i, t)\).

    Parameters:
        Z (np.ndarray): Observed data of shape (N, T) or (N,)
        Zhat_ens (np.ndarray): Ensemble predictions of shape (N, num_samples, T)
        
    Returns:
        tuple: (V_bias, V_normRMSE, V3) each of shape (N,)
    """
    # Ensure Z is at least 2-dimensional (N, T)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    # Compute ensemble mean and variance at each (s_i, t)
    Zhat_mean = np.mean(Zhat_ens, axis=1)  # shape (N, T)
    Zhat_var = np.var(Zhat_ens, axis=1)      # shape (N, T)
    
    N, T = Z.shape
    V_bias = np.zeros(N)
    V_normRMSE = np.zeros(N)
    V3 = np.zeros(N)
    
    for i in range(N):
        error = Z[i, :] - Zhat_mean[i, :]
        avg_error = np.mean(error)
        avg_variance = np.mean(Zhat_var[i, :])
        # Avoid division by zero in case predictive variance is 0.
        V_bias[i] = avg_error / np.sqrt(avg_variance) if avg_variance > 0 else np.nan
        
        mse = np.mean(error ** 2)
        V_normRMSE[i] = np.sqrt(mse / avg_variance) if avg_variance > 0 else np.nan
        V3[i] = np.sqrt(mse)
        
    return V_bias, V_normRMSE, V3

def compute_crps(Z: np.ndarray, Zhat_ens: np.ndarray) -> float:
    """
    Compute the Continuous Ranked Probability Score (CRPS)
    using an empirical predictive CDF.

    For each spatio-temporal index (i,t), if the ensemble forecast is
    \( \{ \hat{Z}_{1}, \dots, \hat{Z}_{M} \} \) then the CRPS at (i, t)
    is given by:
    
    \[
    \text{CRPS} = \frac{1}{M}\sum_{m=1}^{M} \left|\hat{Z}_{m} - Z(i,t)\right| -
                \frac{1}{2M^2}\sum_{m=1}^{M}\sum_{m'=1}^{M}\left|\hat{Z}_{m}-\hat{Z}_{m'}\right|
    \]
    
    The returned value is the average CRPS over all i and t.

    Parameters:
        Z (np.ndarray): Observed data of shape (N, T)
        Zhat_ens (np.ndarray): Ensemble predictions of shape (N, num_samples, T)

    Returns:
        float: Average CRPS
    """
    N, num_samples, T = Zhat_ens.shape
    crps_all = np.zeros((N, T))
    
    for i in range(N):
        for t in range(T):
            ens = Zhat_ens[i, :, t]
            obs = Z[i, t] if Z.ndim > 1 else Z[i]
            term1 = np.mean(np.abs(ens - obs))
            term2 = 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :]))
            crps_all[i, t] = term1 - term2

    return np.mean(crps_all)

def main():
    """
    Demonstration of the validation metric computations.

    For illustration, we simulate a spatio-temporal observed field and generate ensemble predictions.
    In this demo:
      - We assume a grid with N spatial locations (e.g. 15x15 grid has N=225) and T time points.
      - The ensemble of predictions is simulated by perturbing the observations.
    """
    # Set seed for reproducibility
    np.random.seed(0)
    
    # Dimensions for the demonstration (e.g., a 15x15 grid over T time points)
    N = 225      # Total spatial locations
    T = 50       # Number of time steps
    num_samples = 100  # Ensemble size
    
    # Simulate observed data Z (N, T)
    Z = np.random.normal(0, 1, (N, T))
    
    # Simulate ensemble predictions as the observed Z plus some ensemble noise.
    # This results in Zhat_ens of shape (N, num_samples, T).
    noise_ensemble = np.random.normal(0, 0.2, (N, num_samples, T))
    Zhat_ens = np.expand_dims(Z, axis=1) + noise_ensemble

    # Compute point predictions as the ensemble mean.
    Zhat = np.mean(Zhat_ens, axis=1)  # shape (N, T)
    
    # Compute overall metrics
    mspe_val   = compute_mspe(Z, Zhat)
    rmspe_val  = compute_rmspe(Z, Zhat)
    mape_val   = compute_mape(Z, Zhat)
    crps_val   = compute_crps(Z, Zhat_ens)
    
    # Compute spatial validation metrics (per spatial location)
    V_bias, V_normRMSE, V3 = compute_spatial_validation_metrics(Z, Zhat_ens)
    
    print("Validation Metrics:")
    print(f"MSPE: {mspe_val:.4f}")
    print(f"RMSPE: {rmspe_val:.4f}")
    print(f"MAPE: {mape_val:.4f}")
    print(f"Average CRPS: {crps_val:.4f}\n")
    
    print("Spatial Validation Statistics (averaged over locations):")
    print(f"Mean normalized bias (V_bias): {np.nanmean(V_bias):.4f}")
    print(f"Mean normalized RMSE (V_normRMSE): {np.nanmean(V_normRMSE):.4f}")
    print(f"Mean spatial RMSE (V3): {np.mean(V3):.4f}")

if __name__ == "__main__":
    main() 