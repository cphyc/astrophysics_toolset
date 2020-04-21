"""Handle periodicity."""
import numpy as np
from typing import Optional
from .types import FloatArrayType
from .decorators import spatial


@spatial
def wrap_coordinates(x : FloatArrayType, w : float = 1) -> FloatArrayType:
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


@spatial
def distance_to_center(x : FloatArrayType, w : float = 1) -> FloatArrayType:
    """Compute the distance to the barycenter, taking into account periodicity.
    
    Parameters:
    -----------
    x : 3D array
        The position
    w : float, optional
        The width of the domain (default : 1)

    Returns
    -------
    dx : 3D array
        The distance to the center, wrapped with periodic boundaries
    """
    if np.any(x > w) or np.any(x < 0):
        x = wrap_coordinates(x, w)
    # Compute distances to first element
    x0 = x[..., 0, :]
    dx = x[..., 1:, :] - x0[..., None, :]
    dx[dx>(+w/2)] -= w
    dx[dx<(-w/2)] += w

    # Now rewrite w.r.t. center
    xcenter = dx.mean(axis=-2) + x0
    dx = x - xcenter[..., None, :]
    dx[dx>(+w/2)] -= w
    dx[dx<(-w/2)] += w
    return dx
