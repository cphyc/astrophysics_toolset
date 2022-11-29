"""Useful decorators."""

import os
from functools import wraps
from typing import Callable

import numpy as np

from .exceptions import AstroToolsetNotSpatialError


class read_files:  # noqa: N801
    """Decorator for functions that take a file as input.

    Parameters:
    -----------
    N : int
        List of the name of the argument that are files

    Notes
    -----
    The wrapped function is expected to have its input files as last argument.
    """

    N_files = 0

    def __init__(self, N: int):
        self.N_files = N

    def __call__(self, fun: Callable) -> Callable:
        @wraps(fun)
        def wrapped(*args, **kwargs):
            nargs = len(args)
            for i in range(self.N_files):
                if not os.path.exists(args[nargs - i - 1]):
                    raise FileNotFoundError(args[nargs - i - 1])

            return fun(*args, **kwargs)

        return wrapped


def spatial(fun):
    """Asserts that the first argument passed to the function is of spatial type,
    i.e. its last dimension has three components.
    """

    @wraps(fun)
    def wrapped(spatial_array, *args, **kwargs):
        if not (
            isinstance(spatial_array, np.ndarray)
            and spatial_array.ndim >= 2
            and spatial_array.shape[-1] == 3
        ):
            raise AstroToolsetNotSpatialError(spatial_array)
        return fun(spatial_array, *args, **kwargs)

    return wrapped
