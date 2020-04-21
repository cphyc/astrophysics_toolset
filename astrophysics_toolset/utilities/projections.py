"""Projections."""
from numba import njit
import numpy as np

from .types import FloatArrayType
from .decorators import spatial


@spatial
@njit
def cic_projection(pos : FloatArrayType, N : int) -> FloatArrayType:
    """Compute a CIC projection of some position on a uniform grid.
    
    Parameters
    ----------
    pos : array-like (Npoint, 3)
        The position in the [0,1] periodic domain
    N : int
        The size of the grid to project onto
    
    Returns
    -------
    dens : array-like (N, N, N)
        The density.

    Notes
    -----
    The scheme conserves the volume, so that dens.sum() == Npoint.
    """
    posN = pos * N
    dens = np.zeros((N, N, N))
    for ipart in range(len(pos)):
        i = int(posN[ipart, 0]+0.5)
        j = int(posN[ipart, 1]+0.5)
        k = int(posN[ipart, 2]+0.5)

        vi0 = abs(i - posN[ipart, 0])
        vj0 = abs(j - posN[ipart, 1])
        vk0 = abs(k - posN[ipart, 2])

        vi1 = 1 - vi0
        vj1 = 1 - vj0
        vk1 = 1 - vk0

        for ii, vi in zip((i, i+1), (vi0, vi1)):
            for jj, vj in zip((j, j+1), (vj0, vj1)):
                for kk, vk in zip((k, k+1), (vk0, vk1)):
                    v = vi * vj * vk
                    dens[ii%N, jj%N, kk%N] += v

    return dens


@spatial
@njit
def nearest_projection(pos : FloatArrayType, N : int) -> FloatArrayType:
    """Compute a nearest projection of some position on a uniform grid.
    
    Parameters
    ----------
    pos : array-like (Npoint, 3)
        The position in the [0,1] periodic domain
    N : int
        The size of the grid to project onto
    
    Returns
    -------
    dens : array-like (N, N, N)
        The density.

    Notes
    -----
    The scheme conserves the volume, so that dens.sum() == Npoint.
    """
    posN = pos * N
    dens = np.zeros((N, N, N))
    for ipart in range(len(pos)):
        i = int(posN[ipart, 0]+0.5)
        j = int(posN[ipart, 1]+0.5)
        k = int(posN[ipart, 2]+0.5)

        dens[i%N, j%N, k%N] += 1

    return dens