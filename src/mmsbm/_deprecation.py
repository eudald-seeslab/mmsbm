"""Deprecation helpers of the 2.x series (see the deprecation policy in the changelog)."""
import os
import sys
import warnings
from importlib import import_module

REMOVAL = "mmsbm 3.0 (not before October 2027)"
_PACKAGE_DIR = os.path.dirname(__file__)


def _same_file(filename, candidates):
    """Whether ``filename`` is one of ``candidates``, also when one of them goes through a symbolic link."""
    return filename in candidates or os.path.realpath(filename) in {os.path.realpath(c) for c in candidates}


def _in_package(filename):
    return filename.startswith(_PACKAGE_DIR) or os.path.realpath(filename).startswith(os.path.realpath(_PACKAGE_DIR))


def _stacklevel(skip=()):
    """Stack level, for a ``warnings.warn`` issued by :func:`warn_deprecated`, of the first frame that belongs
    neither to this package, nor to the import system, nor to the files in ``skip``: the user's code."""
    level = 2
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if "importlib" in filename and "_bootstrap" in filename:
            # ``warnings.warn`` does not count the frames of the import machinery either
            frame = frame.f_back
            continue
        if not (_in_package(filename) or _same_file(filename, skip) or "importlib" in filename):
            break
        frame = frame.f_back
        level += 1
    return level


def warn_deprecated(what, instead, skip=()):
    """Emit the ``FutureWarning`` of a deprecated feature, attributed to the user's code.

    ``FutureWarning`` rather than ``DeprecationWarning`` because the latter is hidden by default outside
    ``__main__`` and test runners, and the users of the package are scripts and notebooks.
    """
    warnings.warn(
        f"{what} is deprecated and will be removed in {REMOVAL}; use {instead} instead.",
        FutureWarning,
        stacklevel=_stacklevel(skip),
    )


def alias_module(namespace, target):
    """Turn the module whose ``globals()`` is ``namespace`` into an alias of the module ``target``.

    Every name of ``target`` (except the dunder ones) is copied into ``namespace``, so ``from old import x``
    and ``old.x`` keep working and refer to the same objects, and a warning is emitted once per process.
    """
    module = import_module(target)
    namespace.update({k: v for k, v in vars(module).items() if not (k.startswith("__") and k.endswith("__"))})
    warn_deprecated(f"The module '{namespace['__name__']}'", target, skip=(namespace["__file__"],))
