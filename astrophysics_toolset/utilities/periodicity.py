"""Handle periodicity."""
import numpy as np
from typing import Optional
from .types import FloatArrayType

def wrap_coordinates(x : FloatArrayType, w : Optional[float] = 1) -> FloatArrayType:
    """Wrap the position, taking into account periodicity.
    
    Parameters:
    -----------
    x : 3D array
        The position
    w : float, optional
        The width of the domain (default : 1)

    Returns
    -------
    x_wrapped : 3D array
        The positions, wrapped with periodic boundaries
    """
    x_wrapped = np.mod(x, w)
    return x_wrapped
