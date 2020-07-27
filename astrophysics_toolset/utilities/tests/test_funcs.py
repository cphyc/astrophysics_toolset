import numpy as np
from mpmath import besselj, mpf, pi, sqrt

from ..funcs import j1_over_x


@np.vectorize
def mpmath_jn_over_x(i, x):
    xx = mpf(x)
    if x == 0:
        return float(1 / mpf(3))
    else:
        return float(sqrt(pi / mpf(2) / xx) * besselj(i + mpf("1/2"), xx) / xx)


def test_j1_over_x():
    x = np.concatenate(([0], np.geomspace(1e-8, 10, 1000)))

    yref = mpmath_jn_over_x(1, x)
    yval = j1_over_x(x)

    np.testing.assert_allclose(yref, yval, rtol=1e-14)
