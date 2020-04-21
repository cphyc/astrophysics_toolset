"""Support Grafic I/O."""

import os
from scipy.io import FortranFile as FF
import pathlib
from typing import Union
import numpy as np

from ..utilities.decorators import read_files
from ..utilities.types import PathType
from .common import IOResult

@read_files(1)
def read(fname : PathType) -> IOResult:
    """Read a grafic file.

    Parameters
    ----------
    fname : str, filename
    
    """
    aexp = float(fname.split('-')[-1].replace('.dat', ''))
    with FF(fname, 'r') as f:
        N1, N2, N3, qq, L1, L2, L3, Omega_m, Omega_c, Omega_l, H = f.read_record(*['i']*3, *['f']*8)
        Lbox = qq * L1 * H / 100
        Lbox /= 1000  # Gpc
        Lbox = float(Lbox)

        N1, N2, N3 = (int(_) for _ in (N1, N2, N3))

        dens = np.zeros((N1, N2, N3), dtype='f')
        for i in range(N1):
            dens[i] = f.read_reals('f').reshape(N2, N3)

    return IOResult(data=dict(rho=dens), metadata=dict(Lbox=Lbox, Omega_m=Omega_m, Omega_c=Omega_c, Omega_l=Omega_l, aexp=aexp, N=N1))