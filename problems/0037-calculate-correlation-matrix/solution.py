import numpy as np

def calculate_correlation_matrix(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Calculates the correlation matrix for a given dataset X, and optionally Y.
    Returns only the cross-correlation between X and Y when Y is provided.
    """
    if Y is None:
        return np.corrcoef(X, rowvar=False)
    
    # NumPy combines them: rows/cols 0 to num_features_X belong to X, 
    # and the remaining belong to Y.
    full_matrix = np.corrcoef(X, Y, rowvar=False)
    
    num_features_X = X.shape[1]
    
    # Extract only the top-right block: X vs Y correlations
    return full_matrix[:num_features_X, num_features_X:]
