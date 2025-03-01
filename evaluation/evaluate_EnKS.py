import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def evaluate_enks_sample(ens_state, true_state, grid_shape, plot=True, save_path=None):
    """
    Evaluate an EnKS sample against the true state and optionally create visualization plots.
    
    This function is designed for spatiotemporal state evaluations. Both the estimated and true states 
    should be 2D arrays with shape (N, T) where N is the number of spatial locations (e.g., grid_size_x * grid_size_y)
    and T is the number of time steps.
    
    The grid_shape parameter is used to reshape the state vector back into its spatial configuration for computing 
    spatially-resolved error metrics.
    
    Parameters:
        ens_state (np.ndarray): Estimated state from EnKS, shape (N, T).
        true_state (np.ndarray): True state, shape (N, T).
        grid_shape (tuple): A tuple (grid_size_x, grid_size_y) representing the spatial dimensions.
        plot (bool): Whether to generate evaluation plots.
        save_path (str, optional): Path to save the plots. If None, plots are displayed.
    
    Returns:
        dict: A dictionary containing:
            - overall_RMSE: Overall root mean squared error over space and time.
            - time_RMSE: A numpy array of RMSE for each time step (spatial average).
            - spatial_RMSE: A 2D numpy array (grid) of RMSE per spatial location (averaged over time).
            - correlation: Overall Pearson correlation coefficient between estimated and true states.
            - figure: The generated figure (if plot is True)
    """
    if ens_state.shape != true_state.shape:
        raise ValueError("The estimated state and true state must have the same shape.")
    
    N, T = true_state.shape
    grid_size_x, grid_size_y = grid_shape
    if N != grid_size_x * grid_size_y:
        raise ValueError("Mismatch between grid_shape and state dimension.")
    
    # Overall RMSE (computed over all space and time)
    overall_rmse = np.sqrt(np.mean((ens_state - true_state) ** 2))
    
    # RMSE per time step (average error across the spatial grid for each time)
    time_RMSE = np.sqrt(np.mean((ens_state - true_state) ** 2, axis=0))
    
    # Calculate spatial RMSE by reshaping into grid (grid_size_x x grid_size_y) and averaging over time
    ens_state_grid = ens_state.reshape((grid_size_x, grid_size_y, T))
    true_state_grid = true_state.reshape((grid_size_x, grid_size_y, T))
    spatial_RMSE = np.sqrt(np.mean((ens_state_grid - true_state_grid) ** 2, axis=2))
    
    # Compute Pearson correlation coefficient between the flattened estimated and true state
    correlation = np.corrcoef(ens_state.flatten(), true_state.flatten())[0, 1]
    
    metrics = {
        "overall_RMSE": overall_rmse,
        "time_RMSE": time_RMSE,
        "spatial_RMSE": spatial_RMSE,
        "correlation": correlation
    }
    
    if plot:
        fig = create_evaluation_plots(metrics, ens_state, true_state, grid_shape)
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()
        metrics["figure"] = fig
    
    # return metrics

def create_evaluation_plots(metrics, ens_state, true_state, grid_shape):
    """
    Create a comprehensive visualization of the EnKS evaluation metrics.
    
    Parameters:
        metrics (dict): Dictionary containing evaluation metrics
        ens_state (np.ndarray): Estimated state
        true_state (np.ndarray): True state
        grid_shape (tuple): Spatial dimensions (grid_size_x, grid_size_y)
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig = plt.figure(figsize=(15, 12))  # Made figure taller to accommodate new plot
    gs = GridSpec(3, 3, figure=fig)  # Changed to 3 rows
    
    # 1. Time series of RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['time_RMSE'], 'b-', label='RMSE')
    ax1.set_title('RMSE over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('RMSE')
    ax1.grid(True)
    
    # 2. Spatial RMSE heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(metrics['spatial_RMSE'], cmap='viridis')
    ax2.set_title(f'Spatial RMSE\nOverall RMSE: {metrics["overall_RMSE"]:.4f}')
    plt.colorbar(im, ax=ax2)
    
    # 3. Correlation scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(true_state.flatten(), ens_state.flatten(), alpha=0.1, s=1)
    ax3.plot([true_state.min(), true_state.max()], 
             [true_state.min(), true_state.max()], 'r--')
    ax3.set_title(f'True vs Estimated\nCorrelation: {metrics["correlation"]:.4f}')
    ax3.set_xlabel('True State')
    ax3.set_ylabel('Estimated State')
    
    # 4. Random location time series comparison
    ax4 = fig.add_subplot(gs[1, :])
    random_loc = np.random.randint(0, ens_state.shape[0])
    grid_x, grid_y = random_loc // grid_shape[1], random_loc % grid_shape[1]
    
    time_steps = np.arange(ens_state.shape[1])
    ax4.plot(time_steps, true_state[random_loc, :], 'b-', label='True State', linewidth=2)
    ax4.plot(time_steps, ens_state[random_loc, :], 'r--', label='Estimated State', linewidth=2)
    ax4.set_title(f'State Evolution at Random Location (grid position: {grid_x}, {grid_y})')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('State Value')
    ax4.grid(True)
    ax4.legend()
    
    # 5. State evolution comparison (moved to last row)
    time_points = [0, true_state.shape[1]//2, -1]  # Start, middle, end
    for i, t in enumerate(time_points):
        true_snapshot = true_state[:, t].reshape(grid_shape)
        est_snapshot = ens_state[:, t].reshape(grid_shape)
        
        ax5_true = fig.add_subplot(gs[2, i])
        im = ax5_true.imshow(true_snapshot, cmap='viridis')
        ax5_true.set_title(f'Time {t}\nTrue (top) vs Estimated (bottom)')
        plt.colorbar(im, ax=ax5_true)
        
        ax5_est = fig.add_subplot(gs[2, i])
        im = ax5_est.imshow(est_snapshot, cmap='viridis')
        plt.colorbar(im, ax=ax5_est)
    
    plt.tight_layout()
    return fig 