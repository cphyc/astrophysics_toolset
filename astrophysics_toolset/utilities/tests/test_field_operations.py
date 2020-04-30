from ..field_operations import gaussian_filter

from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.optimize import curve_fit
import numpy as np


def test_gaussian_filter():
    N = 1000
    R = 50
    x = np.zeros(N)
    x[N//2] = 10

    xref = np.arange(N)
    yref = scipy_gaussian_filter(x, R, mode='wrap')
    ynew = gaussian_filter(x, R)

    def fit_fun(x, C, sigma):
        return C*np.exp(-(x-N//2)**2/2/sigma**2)

    popt, pcov = curve_fit(fit_fun, xref, yref, p0=[1, 2])

    Rmeasured = popt[1]
    dR = np.diag(np.sqrt(pcov))[1]

    assert Rmeasured-dR < R < Rmeasured+dR
    np.testing.assert_allclose(yref, ynew, atol=1e-4, rtol=0)