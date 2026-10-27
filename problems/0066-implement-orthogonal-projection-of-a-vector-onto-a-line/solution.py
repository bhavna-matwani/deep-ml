import numpy as np

def orthogonal_projection(v: list[float], L: list[float]) -> list[float]:
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    if len(v) != len(L) or all(x == 0 for x in L):
        return []

    v_arr = np.array(v)
    L_arr = np.array(L)

    projection = np.dot(v_arr, L_arr) / np.dot(L_arr, L_arr) * L_arr

    return projection.tolist()
