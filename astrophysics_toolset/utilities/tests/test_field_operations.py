from ..field_operations import gaussian_filter, top_hat_filter

from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.optimize import curve_fit
import numpy as np
import pytest


N1 = 1000
R1 = 50

N3 = 64
R3 = 20


@pytest.fixture
def x_1D():
    x_1D = np.zeros(N1)
    x_1D[N1//2] = 10
    return x_1D


@pytest.fixture
def x_3D():
    x_3D = np.zeros((N3, N3, N3))
    x_3D[N3//2, N3//2, N3//2] = 10
    return x_3D


def test_gaussian_filter_1(x_1D):
    xref = np.arange(N1)
    ynew = gaussian_filter(x_1D, R1)

    def fit_fun(x_1D, C, sigma):
        return C*np.exp(-(x_1D-N1//2)**2/2/sigma**2)

    popt, pcov = curve_fit(fit_fun, xref, ynew, p0=[1, 2])
    Rmeasured = popt[1]
    dR = np.sqrt(pcov[1, 1])

    # Test width of curve is correct
    np.testing.assert_allclose(Rmeasured, R1, rtol=dR, atol=0)


def test_gaussian_filter_compare_to_scipy(x_1D, x_3D):
    for x, R in zip((x_1D, x_3D), (R1, R3)):
        yref = scipy_gaussian_filter(x, R, mode='wrap')
        ynew = gaussian_filter(x, R)

        # Compare result to the one using scipy's gaussian filter
        np.testing.assert_allclose(yref, ynew, atol=1e-4, rtol=0)


def test_gaussian_filter_volume(x_1D, x_3D):
    for x, R in zip((x_1D, x_3D), (R1, R3)):
        ynew = gaussian_filter(x, R)

        # Test volume conservation
        np.testing.assert_allclose(ynew.sum(), x.sum())


def test_gaussian_filter_3d(x_3D):
    yref = scipy_gaussian_filter(x_3D, R3, mode='wrap')
    ynew = gaussian_filter(x_3D, R3)

    # Compare result to the one using scipy's gaussian filter
    np.testing.assert_allclose(yref, ynew, atol=1e-4, rtol=0)


def test_top_hat_filter_exception(x_1D):
    with pytest.raises(NotImplementedError) as excinfo:
        top_hat_filter(x_1D, R1)

    assert "Top-Hat filtering has not been implemented for dimensions ≠ 3" \
        in str(excinfo.value)


def test_top_hat_filter_volume(x_3D):
    ynew = top_hat_filter(x_3D, R3)
    np.testing.assert_allclose(ynew.sum(), x_3D.sum())
