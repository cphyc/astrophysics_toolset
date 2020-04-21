from astrophysics_toolset.utilities.periodicity import wrap_coordinates

import numpy as np

def test_wrap_coordinates():
    np.random.seed(16091992)
    x = np.random.rand(1000, 3) * 3 - 1

    xw = wrap_coordinates(x)
    assert np.all(xw >= 0)
    assert np.all(xw < 1)

    xw = wrap_coordinates(x, 0.5)
    assert np.all(xw >= 0)
    assert np.all(xw < 0.5)
