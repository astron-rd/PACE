import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def polyval(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Numba-compatible polynomial evaluation (equivalent to np.polyval).

    :param coeffs: Polynomial coefficients in descending order [a_n, a_{n-1}, ..., a_0]
    :param x: Value(s) at which to evaluate the polynomial
    :return Array of evaluated polynomial values, shape (len(x))
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        val = coefficients[0]
        for j in range(1, len(coefficients)):
            val = val * x[i] + coefficients[j]
        result[i] = val
    return result


@nb.njit(fastmath=True)
def evaluate_spheroidal(nu: np.ndarray) -> np.ndarray:
    """
    Evaluate the prolate spheroidal wave function.

    param: nu: parameters of the spheroidal wave function
    return: array with value of the spheroidal wave function, shape (len(nu))
    """
    P = np.array(
        [
            [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
            [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
        ]
    )
    Q = np.array(
        [
            [1.0000000e0, 8.212018e-1, 2.078043e-1],
            [1.0000000e0, 9.599102e-1, 2.918724e-1],
        ]
    )

    # Create result array
    result = np.zeros_like(nu)

    # Process each part
    for part, end in [(0, 0.75), (1, 1.00)]:
        mask = (nu >= (0.0 if part == 0 else 0.75)) & (nu <= end)
        if not np.any(mask):
            continue

        nu_part = nu[mask]
        nusq = nu_part**2
        delnusq = nusq - end**2

        # Calculate polynomial using Horner's method
        top = polyval(P[part][::-1], delnusq)
        bot = polyval(Q[part][::-1], delnusq)

        # Avoid division by zero
        valid = bot != 0
        result_part = np.zeros_like(nu_part)
        result_part[valid] = (1.0 - nusq[valid]) * (top[valid] / bot[valid])
        result[mask] = result_part

    return result
