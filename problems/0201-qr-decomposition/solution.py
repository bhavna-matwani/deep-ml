import numpy as np

def qr_decomposition(A):
    """
    Performs QR decomposition of a matrix A using the Gram-Schmidt process.
    
    Parameters:
    A (list of lists or np.ndarray): Input matrix of shape (m, n)
    
    Returns:
    tuple: (Q, R) where Q is an orthogonal matrix and R is upper triangular.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        # Start with the j-th column of A
        v = A[:, j].copy()
        
        # Subtract the projections of v onto previous orthonormal vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
            
        # Compute the norm and normalize the vector to find the next Q column
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
        
    return Q, R
