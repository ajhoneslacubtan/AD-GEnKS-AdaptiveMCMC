import numpy as np
from numba import njit
# from scipy.linalg import cho_factor, cho_solve

@njit(cache=True)
def inverse_covariance_Z(Z, epsilon=1e-5):
    """
    Compute the inverse of the regularized covariance matrix of Z using 
    the Sherman-Morrison-Woodbury formula with Cholesky decomposition.
    """
    No, Ne = Z.shape

    I_ne = np.eye(Ne)
    A_inv = np.eye(No) / epsilon

    middle_matrix = (Ne - 1) * I_ne + (1 / epsilon) * (Z.T @ Z)
    
    # Add a small diagonal to ensure positive definiteness
    middle_matrix += 1e-12 * np.eye(Ne)
    L = np.linalg.cholesky(middle_matrix)
    Linv = np.linalg.inv(L)
    middle_matrix_inv = Linv.T @ Linv

    Sigma_inv = A_inv - (A_inv @ Z @ middle_matrix_inv @ Z.T @ A_inv)
    return Sigma_inv


# def inverse_covariance_Z_cholesky(Z, epsilon=1e-5):
#     """
#     Compute the inverse of the regularized covariance matrix of Z using
#     the Sherman-Morrison-Woodbury formula and a Cholesky factorization
#     for the small matrix inversion.
    
#     The regularized covariance is:
#         Sigma_epsilon = epsilon * I + (1/(Ne-1)) * Z Z^T.
    
#     Parameters:
#         Z (np.ndarray): Data matrix of shape (No, Ne) (each column is a sample).
#         epsilon (float): Regularization parameter.
    
#     Returns:
#         np.ndarray: The inverse of Sigma_epsilon, shape (No, No).
#     """
#     No, Ne = Z.shape
#     # A = epsilon * I, so A_inv = (1/epsilon)*I
#     A_inv = 1.0 / epsilon

#     # Build the small matrix M = (Ne-1)*I + (1/epsilon) * (Z^T Z)
#     I_ne = np.eye(Ne)
#     M = (Ne - 1) * I_ne + (1.0 / epsilon) * (Z.T @ Z)

#     # Compute the Cholesky factorization of M
#     c, lower = cho_factor(M)

#     # Instead of inverting M, solve M X = Z.T
#     # X will be an (Ne, No) matrix.
#     X = cho_solve((c, lower), Z.T)
    
#     # Now, use the SMW formula:
#     # Sigma_inv = A_inv*I - A_inv^2 * Z * X
#     Sigma_inv = A_inv * np.eye(No) - (A_inv**2) * (Z @ X)
    
#     return Sigma_inv