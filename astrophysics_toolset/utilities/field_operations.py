"""Perform operations on n-dimensional fields."""

import numpy as np

from .funcs import j1_over_x


def gaussian_filter(i, R):
    """Smooth the input on R pixels

    Parameters
    ----------
    i : np.ndarray
        The array to smooth
    R : float
        The sigma parameter of the Gaussian filter

    Returns
    -------
    i_smoothed : np.ndarray
        The input array, smoothed by a Gaussian filter
    """
    ik = np.fft.rfftn(i)
    d = 1 / (2 * np.pi)
    kgrid = np.stack(
        np.meshgrid(
            *[np.fft.fftfreq(_, d=d) for _ in i.shape[:-1]],
            np.fft.rfftfreq(i.shape[-1], d=d),
            indexing="ij",
        ),
        axis=0,
    )
    k2 = np.sum(kgrid**2, axis=0)

    return np.fft.irfftn(ik * np.exp(-k2 * (R**2 / 2)))


def top_hat_filter(i, R):
    """Smooth the input on R pixels using a Top-Hat filter in Fourier space

    Parameters
    ----------
    i : np.ndarray
        The array to smooth
    R : float
        The radius of the top hat filter

    Returns
    -------
    i_smoothed : np.ndarray
        The input array, smoothed by a Top-Hat
    """
    if i.ndim != 3:
        raise NotImplementedError(
            "Top-Hat filtering has not been implemented for dimensions ≠ 3, got %s."
            % i.ndim
        )
    ik = np.fft.rfftn(i)
    d = 1 / (2 * np.pi)
    kgrid = np.stack(
        np.meshgrid(
            *[np.fft.fftfreq(_, d=d) for _ in i.shape[:-1]],
            np.fft.rfftfreq(i.shape[-1], d=d),
            indexing="ij",
        ),
        axis=0,
    )
    kR = np.linalg.norm(kgrid, axis=0) * R

    W = 3 * j1_over_x(kR)

    return np.fft.irfftn(ik * W)
