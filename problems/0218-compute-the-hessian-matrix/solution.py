from typing import Callable

def compute_hessian(
    f: Callable[[list[float]], float],
    point: list[float],
    h: float = 1e-5
) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point
    using finite differences.
    """
    n = len(point)
    hessian = [[0.0] * n for _ in range(n)]
    f0 = f(point)

    for i in range(n):
        # Diagonal: second derivative with respect to x_i
        x_plus = point.copy()
        x_minus = point.copy()

        x_plus[i] += h
        x_minus[i] -= h

        hessian[i][i] = (f(x_plus) - 2 * f0 + f(x_minus)) / (h ** 2)

        # Off-diagonal: mixed second derivatives
        for j in range(i + 1, n):
            x_pp = point.copy()
            x_pm = point.copy()
            x_mp = point.copy()
            x_mm = point.copy()

            x_pp[i] += h
            x_pp[j] += h

            x_pm[i] += h
            x_pm[j] -= h

            x_mp[i] -= h
            x_mp[j] += h

            x_mm[i] -= h
            x_mm[j] -= h

            value = (
                f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)
            ) / (4 * h ** 2)

            hessian[i][j] = value
            hessian[j][i] = value

    return hessian