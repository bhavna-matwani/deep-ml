import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    # Convert inputs to float arrays to avoid integer division
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    
    n = len(b)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find the row with the largest absolute value in the current column
        max_row = i + np.argmax(np.abs(A[i:n, i]))
        
        # Swap the current row with the row containing the maximum pivot element
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            
        # Check for a zero or near-zero pivot
        if np.isclose(A[i, i], 0.0):
            raise ValueError("Matrix is singular or nearly singular.")
            
        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
            
    # Back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        
    return x
