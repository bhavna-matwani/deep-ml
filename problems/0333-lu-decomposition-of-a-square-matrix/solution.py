import numpy as np

def lu_decomposition(A):
    """
    Decomposes a square matrix A into L and U using the Doolittle algorithm.
    
    Parameters:
    A (list of lists or np.ndarray): Square matrix to decompose.
    
    Returns:
    tuple: (L, U) as numpy arrays.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    # Initialize L as an identity matrix and U as a zero matrix
    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        # 1. Compute the Upper triangular matrix row (U[i, k])
        for k in range(i, n):
            s = sum(L[i][j] * U[j][k] for j in range(i))
            U[i, k] = A[i, k] - s
            
        # 2. Compute the Lower triangular matrix column (L[k, i])
        for k in range(i + 1, n):
            s = sum(L[k][j] * U[j][i] for j in range(i))
            if U[i, i] == 0:
                raise ValueError("Zero pivot encountered. Matrix requires pivoting.")
            L[k, i] = (A[k, i] - s) / U[i, i]
            
    return L, U