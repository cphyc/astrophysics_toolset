from numba import vectorize, float64, float32
import numpy as np


@vectorize([float64(float64), float32(float32)])
def j1_over_x(x):
    """Compute j_1(x)/x.

    Parameter
    ---------
    x : float, or array-like

    Return
    ------
    j_1(x) / x : float, or array-like
    """
    x2 = x*x
    if x < 1e-2:
        return (1 - x2*0.1*(1 - x2/28))/3
    # elif x < 8e-2:
    #     return (1 - x2*0.1*(1 - x2/28*(1 - x2/54.)))/3
    elif x < 2e-1:
        return (1 - x2*0.1*(1 - x2/28*(1 - x2/54. * (1 - x2/88.))))/3
    else:
        return (np.sin(x) / x - np.cos(x))/x2