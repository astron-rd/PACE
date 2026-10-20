from .backends import get_backend


def get_taper(subgrid_size: int, backend_name: str = "numba"):
    """Construct the taper using the given backend's spheroidal evaluation."""
    backend = get_backend(backend_name)
    return backend.get_taper(subgrid_size)
