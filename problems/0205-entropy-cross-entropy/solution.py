import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    mask = P > 0

    entropy = -np.sum(P[mask] * np.log(P[mask]))

    if np.any(Q[mask] == 0):
        cross = float('inf')
    else:
        cross = -np.sum(P[mask] * np.log(Q[mask]))

    if entropy == 0:
        entropy = 0.0

    if cross == 0:
        cross = 0.0

    return float(entropy), float(cross)