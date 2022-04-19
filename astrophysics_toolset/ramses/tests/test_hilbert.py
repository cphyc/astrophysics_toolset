import numpy as np
import pytest
from yt.frontends.ramses.hilbert import hilbert3d as hilbert3d_yt

from astrophysics_toolset.ramses.hilbert import hilbert3d


@pytest.fixture
def seed():
    np.random.seed(16091992)


@pytest.mark.parametrize("bit_length", range(5, 40))
def test_hilbert(seed, bit_length):
    N = 10

    ipos = np.random.randint(2**bit_length, size=(N, 3))

    ref = hilbert3d_yt(ipos, bit_length)
    new = hilbert3d(ipos, bit_length).astype(np.float64)

    assert np.any(ref > 0)
    np.testing.assert_allclose(ref, new)
