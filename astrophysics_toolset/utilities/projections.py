"""Projections."""
import numpy as np
from numba import njit

from .decorators import spatial
from .types import FloatArrayType


@spatial
@njit
def cic_projection(pos: FloatArrayType, N: int) -> FloatArrayType:
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
    The density array follows the C convention, i.e. it is ordered as z, y, x

    Example
    -------
        X = np.array([
            [0.42, 0.125, 0.125],
            [0.51, 0.125, 0.125]])
        dens = cic_projection(X, 4)

        # Varying dimension
        #    z
        #    |  y
        #    |  |  x
        dens[0, 0, :] == [0, 1.28, 0.72, 0]
    """
    posN = pos * N
    dens = np.zeros((N, N, N))
    for ipart in range(len(pos)):
        x, y, z = posN[ipart, 0], posN[ipart, 1], posN[ipart, 2]
        i = int(np.floor(x - 0.5))
        j = int(np.floor(y - 0.5))
        k = int(np.floor(z - 0.5))

        vi0 = min(i + 1, x + 0.5) - max(i, x - 0.5)
        vj0 = min(j + 1, y + 0.5) - max(j, y - 0.5)
        vk0 = min(k + 1, z + 0.5) - max(k, z - 0.5)

        vi1 = 1 - vi0
        vj1 = 1 - vj0
        vk1 = 1 - vk0

        for kk, vk in zip((k, k + 1), (vk0, vk1)):
            for jj, vj in zip((j, j + 1), (vj0, vj1)):
                for ii, vi in zip((i, i + 1), (vi0, vi1)):
                    v = vi * vj * vk
                    dens[kk % N, jj % N, ii % N] += v

    return dens


@spatial
@njit
def nearest_projection(pos: FloatArrayType, N: int) -> FloatArrayType:
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
        x, y, z = posN[ipart, 0], posN[ipart, 1], posN[ipart, 2]
        i = int(np.floor(x - 0.5))
        j = int(np.floor(y - 0.5))
        k = int(np.floor(z - 0.5))

        dens[k % N, j % N, i % N] += 1

    return dens
