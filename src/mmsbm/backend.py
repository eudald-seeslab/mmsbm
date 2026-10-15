from importlib import import_module


def load_backend(name: str = "auto"):
    """Return kernel functions (compute_omegas, update_coeffs, prod_dist).

    Parameters
    ----------
    name : str
        ``"auto"`` chooses the highest-performance backend available, in the
        order cupy, numba, numpy; ``"cupy"`` forces the GPU backend (raises
        ImportError if unavailable); ``"numba"`` forces the CPU/Numba backend;
        ``"numpy"`` is the reference implementation.
    """
    order = ["cupy", "numba", "numpy"] if name == "auto" else [name]

    last_error = None
    for backend in order:
        try:
            mod = import_module(f"mmsbm.kernels_{backend}")
            return mod.compute_omegas, mod.update_coefficients, mod.prod_dist, backend
        except ModuleNotFoundError as e:
            last_error = e
        except ImportError as e:  # e.g. cupy missing CUDA libs
            last_error = e
    raise ImportError(
        f"Could not load any backend. Last error: {last_error}") 