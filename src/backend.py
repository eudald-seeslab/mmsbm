from importlib import import_module


def load_backend(name: str = "auto"):
    """Return kernel functions (compute_omegas, update_coeffs, prod_dist).

    Parameters
    ----------
    name : str
        "auto"   – choose highest-performance backend available in the
                    order cupy → numba → numpy.
        "cupy"   – force GPU backend (raises ImportError if unavailable).
        "numba"  – force CPU/Numba backend.
        "numpy"  – reference implementation.
    """
    order = ["cupy", "numba", "numpy"] if name == "auto" else [name]

    last_error = None
    for backend in order:
        try:
            mod = import_module(f"kernels_{backend}")
            return mod.compute_omegas, mod.update_coefficients, mod.prod_dist, backend
        except ModuleNotFoundError as e:
            last_error = e
        except ImportError as e:  # e.g. cupy missing CUDA libs
            last_error = e
    raise ImportError(
        f"Could not load any backend. Last error: {last_error}") 