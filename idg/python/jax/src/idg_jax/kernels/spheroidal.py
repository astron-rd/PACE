import jax
import jax.numpy as jnp


@jax.jit
def _polyval(coefficients, x):
    """Evaluate a polynomial (np.polyval equivalent) with JAX."""
    result = jnp.zeros_like(x)
    for c in coefficients:
        result = result * x + c
    return result


@jax.jit
def evaluate_spheroidal(nu):
    """Evaluate the prolate spheroidal wave function (JAX port of the numba
    version, which is a fast approximation of the PSWF used as taper)."""
    p = jnp.array(
        [
            [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
            [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
        ]
    )
    q = jnp.array(
        [
            [1.0000000e0, 8.212018e-1, 2.078043e-1],
            [1.0000000e0, 9.599102e-1, 2.918724e-1],
        ]
    )

    result = jnp.zeros_like(nu)

    for part, end in [(0, 0.75), (1, 1.00)]:
        lo = 0.0 if part == 0 else 0.75
        mask = (nu >= lo) & (nu <= end)
        nusq = nu**2
        delnusq = nusq - end**2

        top = _polyval(p[part][::-1], delnusq)
        bot = _polyval(q[part][::-1], delnusq)

        valid = bot != 0
        term = jnp.where(valid, (1.0 - nusq) * (top / bot), 0.0)
        result = jnp.where(mask, term, result)

    return result
