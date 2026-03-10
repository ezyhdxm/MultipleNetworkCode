import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import qr
from math import sqrt, log

##############################
# Low-Rank Signal Generation #
##############################

def generate_random_orthogonal_vectors(m, r):
    """
    Generate r orthonormal vectors in R^m using QR decomposition.

    Parameters:
    - m (int): Dimension of the ambient space.
    - r (int): Number of orthonormal vectors to generate (r <= m).

    Returns:
    - Q (ndarray): An m x r matrix with orthonormal columns.
    """
    if r > m:
        raise ValueError("r must be less than or equal to m")
    # Step 1: Generate a random m x r matrix with i.i.d. standard normal entries
    A = np.random.randn(m, r)
    # Step 2: QR decomposition
    Q, R = np.linalg.qr(A)
    # Step 3: Return the orthonormal columns
    return Q[:, :r]

    

def generate_coherent_orthogonal_vectors(n, r, mu):
    """
    Generate an r-dimensional subspace with coherence mu in an n-dimensional space.
    
    Parameters:
    n (int): Dimension of the space.
    r (int): Number of orthogonal vectors (rank).
    coh (float): incoherence parameter (1 to n/r).

    Returns:
    Q (ndarray): (n x k) matrix with sparse orthogonal columns.
    """
    assert 1 <= mu <= (n/r), "Coherence must be in [1,n/r]"
    assert r <= n, "Number of vectors must be ≤ dimension"
    
    Q = np.zeros((n, r))  # Initialize sparse matrix
    
    m = int(n/mu)  # High coherence portion
    Q[:m,:] = generate_random_orthogonal_vectors(m, r)  # Fill with orthogonal vectors
    Q[m:,:] = generate_random_orthogonal_vectors(n - m, r)  # Fill the rest with another set of orthogonal vectors
    Q = Q / np.linalg.norm(Q, axis=0)  # Normalize columns to unit length
        
    return Q

def generate_low_rank_coherent_signal(n, r, eigenvalues, mu=None):
    """
    Generate a symmetric low-rank matrix with sparse orthogonal eigenvectors.

    Parameters:
    n (int): Dimension of the matrix (n x n).
    r (int): Rank of the matrix.
    eigenvalues (list or array): List of r nonzero eigenvalues.
    sparsity (float): Fraction of zero entries in eigenvectors (0 to 1).

    Returns:
    A (ndarray): Symmetric low-rank matrix.
    Q (ndarray): Sparse orthogonal eigenvectors.
    """
    assert r <= n, "Rank must be ≤ n"
    assert len(eigenvalues) == r, "Eigenvalue list must match rank k"

    if mu is None:
        mu = sqrt(n)
        print(mu)
    
    # Step 1: Generate coherent subspace
    Q = generate_coherent_orthogonal_vectors(n, r, mu)

    # Step 2: Construct diagonal matrix with eigenvalues
    Lambda = np.diag(eigenvalues)

    # Step 3: Construct the low-rank symmetric matrix
    A = Q @ Lambda @ Q.T

    return A, Q


def generate_low_rank_symmetric(n, r, eigenvalues):
    """
    Generate an n x n symmetric matrix of rank r with given eigenvalues.
    
    Parameters:
    n (int): Dimension of the matrix.
    r (int): Rank of the matrix.
    eigenvalues (list or array): A list of r nonzero eigenvalues.
    
    Returns:
    np.ndarray: An n x n symmetric low-rank matrix.
    """
    assert r <= n, "Rank r must be <= matrix dimension n"
    assert len(eigenvalues) == r, "Number of eigenvalues must match rank k"

    # Step 1: Create an (n x n) random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))  # QR decomposition to ensure orthogonality

    # Step 2: Construct diagonal matrix with given eigenvalues
    Lambda = np.zeros((n, n))  # Full zero matrix
    Lambda[:r, :r] = np.diag(eigenvalues)  # Assign eigenvalues to top-left k×k block

    # Step 3: Compute the symmetric matrix
    A = Q @ Lambda @ Q.T
    return A, Q[:, :r]













#################################
# Node-Sparse Signal Generation #
#################################

def generate_node_sparse_signal_general(n, m, scale=None):
    """
    Generate an n x n matrix with exactly m nonzero rows.
    Each nonzero row is filled with standard normal entries.

    Parameters:
    - n (int): The dimension of the square matrix.
    - m (int): Number of nonzero rows (must satisfy 0 <= m <= n).

    Returns:
    - A (ndarray): n x n matrix with exactly m nonzero rows.
    """
    if not (0 <= m <= n):
        raise ValueError("m must be between 0 and n inclusive")
    if scale is None:
        scale = 2*n**(-0.25)*log(n)**(0.25)

    # Step 1: Randomly choose m distinct row indices to be nonzero
    Istar = np.random.choice(n, m, replace=False)

    # Step 2: Create the matrix
    B = np.zeros((n, n))

    # Step 3: Fill only the selected rows with random values
    B[Istar] = scale * np.random.randn(m, n)

    return (B.T + B), Istar


def generate_node_sparse_signal_sdp(n, m, scale=None):
    """
    Generate an n x n matrix with exactly m nonzero rows.
    Each nonzero row is filled with standard normal entries.

    Parameters:
    - n (int): The dimension of the square matrix.
    - m (int): Number of nonzero rows (must satisfy 0 <= m <= n).

    Returns:
    - A (ndarray): n x n matrix with exactly m nonzero rows.
    """
    if not (0 <= m <= n):
        raise ValueError("m must be between 0 and n inclusive")
    if scale is None:
        scale = 2*n**(-0.25)*log(n)**(0.25)

    # Step 1: Randomly choose m distinct row indices to be nonzero
    Istar = np.random.choice(n, m, replace=False)

    # Step 2: Create the matrix
    B = np.zeros((n, n))

    # Step 3: Fill only the selected rows with random values
    B[Istar] = scale * (2*np.random.binomial(n=m, p=0.5, size=n)-1)

    return (B.T + B), Istar
    
    


def generate_Bstar_failure(n, m, k, C=3):
    """
    Generate the matrix Bstar according to Proposition~\ref{prop:failure}.

    Parameters:
    - n (int): dimension of the matrix
    - m (int): number of signal nodes (size of I)
    - sigma (float): standard deviation of noise
    - C (float): constant used in setting beta_1 and beta_2

    Returns:
    - Bstar (np.ndarray): an (n x n) symmetric signal matrix
    """
    B = np.zeros((n, n))
    beta_2 = 2 * (n ** -0.25) * (np.log(n) ** 0.25)
    beta_1 = C * ((n * np.log(n)) ** 0.25) / np.sqrt(m)

    I = np.random.choice(n, m+k, replace=False)  # indices 0 to m-1
    Istar = I[:m]  # indices 0 to m-1
    J = I[m:]  # indices m to m+19

    for i in Istar:
        for j in J:
            B[i, j] = beta_1
            B[j, i] = beta_1

    # Set B_{i,j} = beta_2 for i in I, j > m+1 or vice versa
    for i in Istar:
        for j in range(n):
            if j not in I:
                B[i, j] = beta_2
                B[j, i] = beta_2

    return B, Istar










####################
# Noise Generation #
####################

def symmetric_gaussian_noise_homo(n, mean=0, var=1):
    """
    Generate an n x n symmetric matrix where all elements have the same Gaussian variance.
    
    Parameters:
    n (int): Size of the matrix (n x n).
    mean (float): Mean of the Gaussian distribution.
    var (float): Desired variance for all entries.
    
    Returns:
    np.ndarray: A symmetric matrix with Gaussian-distributed entries and uniform variance.
    """
    std = np.sqrt(var)

    # Step 1: Generate the diagonal entries
    A = np.zeros((n, n))
    np.fill_diagonal(A, np.random.normal(mean, std, size=n))

    # Step 2: Generate the upper-triangular part with variance var
    upper_triangular = np.random.normal(mean, std, size=(n, n))
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal

    # Step 3: Make the matrix symmetric
    A += upper_triangular + upper_triangular.T  # Reflect to lower triangular part

    return A

def asymmetric_gaussian_noise_homo(n, mean=0, var=1):
    A = np.random.normal(mean, sqrt(var), size=(n, n))
    return A


def symmetric_gaussian_noise_heter(n, mean=0, sdmax=sqrt(1.3), sdmin=sqrt(0.8)):
    Sigma = np.zeros((n, n))
    np.fill_diagonal(Sigma, sdmin + (sdmax-sdmin) * np.random.rand(n))
    upper_triangular = sdmin + (sdmax-sdmin) * np.random.rand(n,n)
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal
    Sigma += upper_triangular + upper_triangular.T 

    # Step 3: Make the matrix symmetric
    A = symmetric_gaussian_noise_homo(n)  # Reflect to lower triangular part

    A *= Sigma
    return A


def asymmetric_gaussian_noise_heter(n, mean=0, sdmax=sqrt(2), sdmin=sqrt(0.8)):
    Sigma = sdmin + (sdmax-sdmin) * np.random.rand(n,n)
    A = np.random.normal(size=(n, n))
    return A * Sigma


def symmetric_gaussian_noise_row_heter(n, mean=0, sdmax=1.3, sdmin=0.8):
    d = sdmin + (sdmax-sdmin) * np.random.rand(n)  # diagonal elements
    # Step 2: Create a matrix where each row is filled with its diagonal value
    Sigma = np.tile(d**2, (n, 1))

    # Step 3: Replace lower triangle with upper triangle
    i_lower = np.tril_indices(n, k=-1)
    Sigma[i_lower] = Sigma.T[i_lower]

    # Step 3: Make the matrix symmetric
    A = symmetric_gaussian_noise_homo(n)  # Reflect to lower triangular part

    A *= Sigma
    return A



def symmetric_t_noise_homo(n, df, var=1):
    if df <= 2:
        raise ValueError("t-distribution variance is infinite for df <= 2.")
    # Step 1: Generate the diagonal entries
    A = np.zeros((n, n))
    if var is not None:
        scale = np.sqrt(var * (df - 2) / df)
    else:
        scale = 1
    np.fill_diagonal(A, np.random.standard_t(df, size=n))

    # Step 2: Generate the upper-triangular part with variance var
    upper_triangular = np.random.standard_t(df, size=(n, n))
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal

    # Step 3: Make the matrix symmetric
    A += upper_triangular + upper_triangular.T  # Reflect to lower triangular part

    return A * scale


def symmetric_t_noise_homo_bound(n, df, mu, var=1):
    if df <= 2:
        raise ValueError("t-distribution variance is infinite for df <= 2.")
    # Step 1: Generate the diagonal entries
    A = np.zeros((n, n))
    if var is not None:
        scale = np.sqrt(var * (df - 2) / df)
    else:
        scale = 1
    np.fill_diagonal(A, np.random.standard_t(df, size=n))

    # Step 2: Generate the upper-triangular part with variance var
    upper_triangular = np.random.standard_t(df, size=(n, n))
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal

    # Step 3: Make the matrix symmetric
    A += upper_triangular + upper_triangular.T  # Reflect to lower triangular part
    A = A * scale
    L = sqrt(var) * sqrt(n/mu)
    A = np.minimum(A, L)
    A = np.maximum(A, -L)
    return A


def asymmetric_t_noise_homo(n, df, var=1):
    A = np.random.standard_t(df, size=(n, n))
    if var is not None:
        scale = np.sqrt(var * (df - 2) / df)
    else:
        scale = 1
    return A * scale


def asymmetric_t_noise_homo_bound(n, df, mu, var=1):
    A = np.random.standard_t(df, size=(n, n))
    if var is not None:
        scale = np.sqrt(var * (df - 2) / df)
    else:
        scale = 1
    A = A * scale
    L = sqrt(var) * sqrt(n/mu)
    A = np.minimum(A, L)
    A = np.maximum(A, -L)
    return A

def symmetric_t_noise_heter(n, df, sdmax=sqrt(2), sdmin=sqrt(0.8)):
    if df <= 2:
        raise ValueError("t-distribution variance is infinite for df <= 2.")
    Sigma = np.zeros((n, n))
    np.fill_diagonal(Sigma, sdmin + (sdmax-sdmin) * np.random.rand(n))
    upper_triangular = sdmin + (sdmax-sdmin) * np.random.rand(n,n)
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal
    Sigma += upper_triangular + upper_triangular.T 
    
    A = symmetric_t_noise_homo(n, df)

    return A * Sigma

def asymmetric_t_noise_heter(n, df, sdmax=sqrt(2), sdmin=sqrt(0.8)):
    Sigma = sdmin + (sdmax-sdmin) * np.random.rand(n,n)
    A = asymmetric_t_noise_homo(n, df)
    return A * Sigma