import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	m = np.array(matrix)
	eigenvalues = np.linalg.eigvals(m).tolist()
	return eigenvalues