"""Kernel backend selection for IDG.

Each backend module exposes the same interface: ``evaluate_spheroidal``,
``get_taper``, ``visibilities_to_subgrids``, ``add_subgrid_to_grid`` and
``compute_phasor``.
"""

from . import jax, numba

_BACKENDS = {
    "numba": numba,
    "jax": jax,
}


def get_backend(name: str):
    """Return the backend module for ``name`` (``"numba"`` or ``"jax"``)."""
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backend {name!r}; choose from {sorted(_BACKENDS)}"
        ) from exc


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


__all__ = ["available_backends", "get_backend", "jax", "numba"]
