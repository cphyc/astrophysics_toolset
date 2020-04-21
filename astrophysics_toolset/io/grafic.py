"""Support Grafic I/O"""

import os
from astrophysics_toolset.utilities.decorators import reads_file

Grafic = namedtuple('Grafic', ['rho', 'Lbox', 'Omega_m', 'Omega_c', 'Omega_l', 'aexp', 'N'])

@read_files(1)
def read(fname):
    if not os.path.exists()
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
    return Grafic(rho=dens, Lbox=Lbox, Omega_m=Omega_m, Omega_c=Omega_c, Omega_l=Omega_l, aexp=aexp, N=N1)