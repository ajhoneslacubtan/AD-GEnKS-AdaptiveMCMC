# utils.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
cimport cython

# Declare external LAPACKE routines.
# Ensure you have LAPACKE available on your system.
cdef extern from "lapacke.h":
    int LAPACKE_dpotrf(int matrix_layout, char uplo, int n, double* a, int lda)
    int LAPACKE_dpotri(int matrix_layout, char uplo, int n, double* a, int lda)

cdef int LAPACK_ROW_MAJOR = 101

@cython.boundscheck(False)
@cython.wraparound(False)
def inverse_covariance_Z(np.ndarray[double, ndim=2, mode="c"] Z, double epsilon=1e-5):
    """
    Compute the inverse of the regularized covariance matrix of Z
    using the Sherman-Morrison-Woodbury formula and a robust inversion
    via LAPACK’s Cholesky-based routines.
    
    Parameters
    ----------
    Z : np.ndarray[double, ndim=2]
        Data matrix with shape (No, Ne). (For your case, e.g. (5607, 100) or (5607, 50).)
    epsilon : double, optional
        Regularization parameter.
    
    Returns
    -------
    Sigma_inv : np.ndarray[double, ndim=2]
        The computed inverse covariance matrix.
    """
    cdef int No = Z.shape[0]
    cdef int Ne = Z.shape[1]
    cdef int info, i, j

    # Create identity matrices.
    cdef np.ndarray[double, ndim=2] I_ne = np.eye(Ne, dtype=np.double)
    cdef np.ndarray[double, ndim=2] A_inv = np.eye(No, dtype=np.double) / epsilon

    # Compute the middle matrix:
    # middle_matrix = (Ne - 1)*I_ne + (1/epsilon)*(Z.T @ Z) + 1e-12 * I_ne
    cdef np.ndarray[double, ndim=2] ZT_Z = np.dot(Z.T, Z)  # (Ne, Ne)
    cdef np.ndarray[double, ndim=2] middle_matrix = (Ne - 1) * I_ne + (1.0/epsilon) * ZT_Z
    middle_matrix += 1e-12 * np.eye(Ne, dtype=np.double)

    # Copy middle_matrix into a C-contiguous array for LAPACKE.
    cdef np.ndarray[double, ndim=2, mode="c"] L_mat = np.copy(middle_matrix)

    # Perform Cholesky factorization. 'L' indicates we factorize into a lower-triangular matrix.
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', Ne, &L_mat[0,0], Ne)
    if info != 0:
        raise ValueError("Cholesky factorization failed, info = %d" % info)

    # Invert the matrix using the Cholesky factorization.
    info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', Ne, &L_mat[0,0], Ne)
    if info != 0:
        raise ValueError("Matrix inversion via Cholesky failed, info = %d" % info)

    # LAPACKE_dpotri writes the result in the lower triangle.
    # Copy the lower-triangular part to the upper triangle to form a full symmetric matrix.
    for i in range(Ne):
        for j in range(i+1, Ne):
            L_mat[i, j] = L_mat[j, i]
    cdef np.ndarray[double, ndim=2] middle_matrix_inv = L_mat

    # Compute Sigma_inv = A_inv - (A_inv @ Z @ middle_matrix_inv @ Z.T @ A_inv)
    cdef np.ndarray[double, ndim=2] temp
    # Multiply in the order: (A_inv @ Z) -> (@ middle_matrix_inv) -> (@ Z.T) -> (@ A_inv)
    temp = np.dot(A_inv, Z)                # shape: (No, Ne)
    temp = np.dot(temp, middle_matrix_inv) # shape: (No, Ne)
    temp = np.dot(temp, Z.T)               # shape: (No, No)
    temp = np.dot(temp, A_inv)             # shape: (No, No)
    cdef np.ndarray[double, ndim=2] Sigma_inv = A_inv - temp

    return Sigma_inv



# Import CBLAS dgemm
cdef extern from "cblas.h":
    void cblas_dgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const double alpha, const double* A, const int lda,
                     const double* B, const int ldb,
                     const double beta, double* C, const int ldc)

cdef int CblasRowMajor = 101
cdef int CblasNoTrans = 111
cdef int CblasTrans   = 112

@cython.boundscheck(False)
@cython.wraparound(False)
def update_enks(np.ndarray[double, ndim=3] Y_analysis,
                np.ndarray[double, ndim=2] hat_Sigma_t_zz_inv,
                np.ndarray[double, ndim=2] Z_tilde,
                np.ndarray[double, ndim=2] observations,
                int t, int lags):
    """
    Optimized update for EnKS run using C-level BLAS calls.
    
    Parameters
    ----------
    Y_analysis : np.ndarray[double, ndim=3]
        Array of shape (N, N_ensemble, T_obs+1).
    hat_Sigma_t_zz_inv : np.ndarray[double, ndim=2]
        Inverse covariance matrix of Z_tilde (shape: (N, N)).
    Z_tilde : np.ndarray[double, ndim=2]
        Pseudo-observations at time t (shape: (N, N_ensemble)).
    observations : np.ndarray[double, ndim=2]
        Observations array (shape: (N, T_obs)).
    t : int
        Current time index (>=1).
    lags : int
        Number of lags.
        
    This function computes:
      innovation = observations[:, t-1][:, None] - Z_tilde
      precomputed_term = hat_Sigma_t_zz_inv @ innovation
      Z_anom = Z_tilde - (row-wise mean of Z_tilde)
      term = Z_anomᵀ @ precomputed_term
    and then, for l from max(0, t-lags) to t, updates:
      Y_analysis[:,:,l] += (Y_anom @ term)/(N_ensemble - 1)
    where Y_anom is Y_analysis[:,:,l] with its row–wise mean removed.
    """
    cdef int N = Y_analysis.shape[0]
    cdef int N_ensemble = Y_analysis.shape[1]
    cdef int l, l_start, i, j, k

    # Allocate innovation (N x N_ensemble)
    cdef np.ndarray[double, ndim=2] innovation = np.empty((N, N_ensemble), dtype=np.double)
    for i in range(N):
        for j in range(N_ensemble):
            innovation[i, j] = observations[i, t-1] - Z_tilde[i, j]

    # Allocate precomputed_term (N x N_ensemble)
    cdef np.ndarray[double, ndim=2] precomputed_term = np.empty((N, N_ensemble), dtype=np.double)
    # Multiply: precomputed_term = hat_Sigma_t_zz_inv (N x N) @ innovation (N x N_ensemble)
    # For CblasRowMajor arrays, lda for hat_Sigma_t_zz_inv is N, for innovation is N_ensemble.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N_ensemble, N,
                1.0, &hat_Sigma_t_zz_inv[0,0], N,
                     &innovation[0,0], N_ensemble,
                0.0, &precomputed_term[0,0], N_ensemble)

    # Compute row-wise means for Z_tilde and subtract to form Z_anom (N x N_ensemble)
    cdef np.ndarray[double, ndim=2] Z_anom = np.empty((N, N_ensemble), dtype=np.double)
    cdef double s
    for i in range(N):
        s = 0.0
        for j in range(N_ensemble):
            s += Z_tilde[i, j]
        s /= N_ensemble
        for j in range(N_ensemble):
            Z_anom[i, j] = Z_tilde[i, j] - s

    # Allocate term (N_ensemble x N_ensemble)
    cdef np.ndarray[double, ndim=2] term = np.empty((N_ensemble, N_ensemble), dtype=np.double)
    # Compute term = Z_anomᵀ @ precomputed_term
    # Z_anom is (N x N_ensemble); we want its transpose (N_ensemble x N)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                N_ensemble, N_ensemble, N,
                1.0, &Z_anom[0,0], N_ensemble,
                     &precomputed_term[0,0], N_ensemble,
                0.0, &term[0,0], N_ensemble)

    # For each lag l from max(0, t-lags) to t, update Y_analysis[:,:,l]
    l_start = t - lags if (t - lags) > 0 else 0
    cdef np.ndarray[double, ndim=2] Y_l, Y_anom, update
    cdef np.ndarray[double, ndim=2] m_Y_l = np.empty((N, 1), dtype=np.double)
    for l in range(l_start, t + 1):
        # Get Y_l = Y_analysis[:,:,l] as a contiguous array.
        Y_l = np.ascontiguousarray(Y_analysis[:, :, l])
        # Compute row-wise mean and Y_anom (N x N_ensemble)
        Y_anom = np.empty((N, N_ensemble), dtype=np.double)
        for i in range(N):
            s = 0.0
            for j in range(N_ensemble):
                s += Y_l[i, j]
            m_Y_l[i, 0] = s / N_ensemble
            for j in range(N_ensemble):
                Y_anom[i, j] = Y_l[i, j] - m_Y_l[i, 0]
        # Compute update = Y_anom @ term, where:
        # Y_anom is (N x N_ensemble), term is (N_ensemble x N_ensemble)
        update = np.empty((N, N_ensemble), dtype=np.double)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N_ensemble, N_ensemble,
                    1.0, &Y_anom[0,0], N_ensemble,
                         &term[0,0], N_ensemble,
                    0.0, &update[0,0], N_ensemble)
        # Scale update and add to Y_analysis[:,:,l]
        for i in range(N):
            for j in range(N_ensemble):
                update[i, j] /= (N_ensemble - 1)
                Y_analysis[i, j, l] += update[i, j]

    return 0  # Return value is not used; Y_analysis is updated in-place.