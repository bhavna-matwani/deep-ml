import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	v = np.array(vectors)
	cov = np.cov(v)
	return cov.tolist()