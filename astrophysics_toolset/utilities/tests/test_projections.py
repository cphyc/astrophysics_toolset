import numpy as np

from ..projections import cic_projection, nearest_projection


def test_projection_conservation():
    # Check that the CIC projection conserves the volume
    X = np.random.rand(1000, 3)

    for proj in (cic_projection, nearest_projection):
        dens = proj(X, 16)
        np.testing.assert_allclose(dens.sum(), len(X))

def test_cic():
    # Corner case
    X = np.array([[0, 0, 0]])
    dens = cic_projection(X, 4)
    dens_real = np.zeros((4, 4, 4))
    for i in (-1, 0):
        for j in (-1, 0):
            for k in (-1, 0):
                dens_real[i, j, k] = 1/8

    np.testing.assert_allclose(dens, dens_real)

    # More complicated case -- checked manually
    np.random.seed(16091992)
    X = np.random.rand(10, 3)
    dens = cic_projection(X, 4)
    np.testing.assert_allclose(
        [dens.ptp(), dens.std(), dens.min(), dens.max(), dens[0, 3, 1], dens[1, 2, 0]],
        [0.800747538955831, 0.18253974078654206, 0.0, 0.800747538955831, 0.0, 0.3539705205842787])