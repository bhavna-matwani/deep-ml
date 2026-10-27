import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    A = np.array(matrix, dtype=float)
    
    # Compute eigenvalues (eigvalsh returns them sorted in ascending order for symmetric matrices)
    evals = np.linalg.eigvals(A)
    
    # Convert eigenvalues to a sorted list of floats
    eigenvalues = sorted([round(float(e), 4) for e in evals])
    
    # Check if all eigenvalues are strictly greater than 0
    is_pd = all(e > 0 for e in eigenvalues)
    
    return {
        'is_positive_definite': is_pd,
        'eigenvalues': eigenvalues
    }