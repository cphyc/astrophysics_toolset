"""Useful decorators"""

from functools import wraps
from typing import Callable
import os

class read_files():
    """Decorator for functions that take a file as input.
    
    Parameters:
    -----------
    N : int
        List of the name of the argument that are files

    Notes
    -----
    The wrapped function is expected to have its input files as first arguments.
    """
    N_files = 0

    def __init__(self, N : int):
        self.N_files = N

    def __call__(self, fun : Callable) -> Callable:
        @wraps(fun)
        def wrapped(*args, **kwargs):
            for i in range(self.N_files):
                if not os.path.exists(args[i]):
                    raise FileNotFoundError(args[i])
            
            fun(*args, **kwargs)

        return wrapped