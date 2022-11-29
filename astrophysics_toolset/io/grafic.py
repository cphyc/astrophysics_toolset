"""Support Grafic I/O."""

import numpy as np
from scipy.io import FortranFile as FF  # noqa: N817

from ..utilities.decorators import read_files
from ..utilities.types import PathType
from .common import IOResult


@read_files(1)
def read(fname: PathType) -> IOResult:
    """Read a grafic file.

    Parameters
    ----------
    fname : str, filename

    """
    try:
        aexp = float(fname.split("-")[-1].replace(".dat", ""))
    except ValueError:
        aexp = 1

    with FF(fname, "r") as f:
        N1, N2, N3, qq, Xc1, Xc2, Xc3, Omega_m, Omega_c, Omega_l, H = f.read_record(
            *["i"] * 3, *["f"] * 8
        )
        Lbox = qq * Xc1 * H / 100
        Lbox /= 1000  # Gpc
        Lbox = float(Lbox)

        N1, N2, N3 = (int(_) for _ in (N1, N2, N3))
        Xc = np.stack((Xc1, Xc2, Xc3)).flatten()

        dens = np.zeros((N1, N2, N3), dtype="f")
        for i in range(N1):
            dens[i] = f.read_reals("f").reshape(N2, N3)

    return IOResult(
        data={"rho": dens},
        metadata={
            "Lbox": Lbox,
            "Omega_m": Omega_m,
            "Omega_c": Omega_c,
            "Omega_l": Omega_l,
            "Xc": Xc,
            "aexp": aexp,
            "N": N1,
        },
    )
