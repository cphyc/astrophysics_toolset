import numpy as np
import pytest

from ..periodicity import distance_to_center, wrap_coordinates


@pytest.fixture
def seed():
    return 16091992


def test_wrap_coordinates(seed):
    np.random.seed(seed)
    x = np.random.rand(1000, 3) * 3 - 1

    xw = wrap_coordinates(x)
    assert np.all(xw >= 0)
    assert np.all(xw < 1)

    xw = wrap_coordinates(x, 0.5)
    assert np.all(xw >= 0)
    assert np.all(xw < 0.5)


def test_distance_to_center(seed):
    np.random.seed(seed)
    x = np.random.rand(1000, 3) * 3 - 1

    dx = distance_to_center(x)
    assert np.all(dx >= -0.5)
    assert np.all(dx <= 0.5)

    dx = distance_to_center(x, 0.5)
    assert np.all(dx >= -0.25)
    assert np.all(dx <= 0.25)

    x = np.random.normal(loc=0.5, scale=0.001, size=(10_000, 3))
    dx_ref = x - x.mean(axis=0)
    dx = distance_to_center(x)

    np.testing.assert_allclose(dx, dx_ref, atol=1e-7)
