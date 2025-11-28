from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import numpy as np
import pandas as pd
import pyneb as pn
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, interp1d
import yt


# Constants
c = 2.99792458e10  # speed of light cm/s
h = 6.6261e-27  # erg s
k = 1.380649e-16  # erg / K


"""
First deal with pyneb
"""
# Initialize pyneb atoms
H1 = pn.RecAtom("H", 1)
O3 = pn.Atom("O", 3)
S3 = pn.Atom("S", 3)
S4 = pn.Atom("S", 4)
N3 = pn.Atom("N", 3)
Ne3 = pn.Atom("Ne", 3)
Ne5 = pn.Atom("Ne", 5)

# Make a Te/ne grid
n_vals = 500
temperatures = np.logspace(1, 9, n_vals)
e_densities = np.logspace(-7, 6, n_vals)

line_dict = {
    "N3-57": {
        "ion": "NIII",
        "atom": N3,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 14.0067,
    },
    "Ne3-36": {
        "ion": "NeIII",
        "atom": Ne3,
        "lev_u": 3,
        "lev_d": 2,
        "e_weight": 20.1797,
    },
    "Ne3-15": {
        "ion": "NeIII",
        "atom": Ne3,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 20.1797,
    },
    "Ne5-24": {
        "ion": "NeV",
        "atom": Ne5,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 20.1797,
    },
    "Ne5-14": {
        "ion": "NeV",
        "atom": Ne5,
        "lev_u": 3,
        "lev_d": 2,
        "e_weight": 20.1797,
    },
    "O3-88": {
        "ion": "OIII",
        "atom": O3,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 15.9994,
    },
    "O3-52": {
        "ion": "OIII",
        "atom": O3,
        "lev_u": 3,
        "lev_d": 2,
        "e_weight": 15.9994,
    },
    "O4-26": {
        "ion": "OIII",
        "atom": O3,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 15.9994,
    },
    "S3-18": {
        "ion": "SIII",
        "atom": S3,
        "lev_u": 3,
        "lev_d": 2,
        "e_weight": 32.065,
    },
    "S3-33": {
        "ion": "SIII",
        "atom": S3,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 32.065,
    },
    "S4-10": {
        "ion": "SIV",
        "atom": S4,
        "lev_u": 2,
        "lev_d": 1,
        "e_weight": 32.065,
    },
    "Hua": {  # Humphreys Alpha
        "ion": "HI",
        "atom": H1,
        "lev_u": 7,
        "lev_d": 6,
        "e_weight": 1.0,
    },
}


@dataclass
class Interpolator:
    atom: pn.Atom
    lev_u: int
    lev_d: int

    @cached_property
    def emissivity(self):
        """
        Calculate the emissivity for the given atom and levels.
        """
        yt.mylog.debug("Calculating emissivity for %s %s→%s", self.atom.name, self.lev_u, self.lev_d)
        # Get the emissivity grid
        em_grid = self.atom.getEmissivity(
            tem=temperatures, den=e_densities, lev_i=self.lev_u, lev_j=self.lev_d
        )
        # Create the interpolator
        interp = RegularGridInterpolator(
            (np.log10(temperatures), np.log10(e_densities)),
            em_grid,
            bounds_error=False,
            fill_value=0.0,
        )
        return interp

    @staticmethod
    def from_line_dict(line_dict):
        """
        Create an Interpolator from a line dictionary.
        """
        atom = line_dict["atom"]
        lev_u = line_dict["lev_u"]
        lev_d = line_dict["lev_d"]
        return Interpolator(atom, lev_u, lev_d)

    def __call__(self, log_t_ne_grid):
        return self.emissivity(log_t_ne_grid)


class HuaInterpolator(Interpolator):
    """
    Special case for the H-alpha line due to NaN values in the emissivity grid.
    """
    @cached_property
    def emissivity(self):
        """
        Calculate the emissivity for the given atom and levels.
        """
        yt.mylog.debug("Calculating emissivity for %s %s→%s", self.atom.name, self.lev_u, self.lev_d)
        em_grid = np.zeros((len(temperatures), len(e_densities)))
        for i, ne in enumerate(e_densities):
            em = self.atom.getEmissivity(
                temperatures, ne, lev_i=self.lev_u, lev_j=self.lev_d
            )
            filt_loc = np.isnan(em)
            mf = interp1d(
                np.log10(temperatures[~filt_loc]),
                np.log10(em[~filt_loc]),
                bounds_error=False,
                fill_value="extrapolate",
            )
            em_grid[:, i] = 10.0 ** mf(np.log10(temperatures))

        interp = RegularGridInterpolator(
            (np.log10(temperatures), np.log10(e_densities)),
            em_grid,
            bounds_error=False,
            fill_value=0.0,
        )
        return interp



for line_dict_data in line_dict.values():
    line_dict_data["emis_grid"] = Interpolator.from_line_dict(line_dict_data)

line_dict["Hua"]["emis_grid"] = HuaInterpolator.from_line_dict(line_dict["Hua"])

"""
Next deal with CLOUDY
"""
data_dir = Path(__file__).parent / "IR_LINE_DATA"

# Set up the temperature grid for cloudy
temp_grid = np.arange(1.0, 5.0, 0.025)

# Load in the collision data that we interpolate from cloudy and set
# up the interpolators
OI_63 = pd.read_csv(data_dir / "O_I" / "OI_63um_rates.dat", index_col=0)
OI_44 = pd.read_csv(data_dir / "O_I" / "OI_44um_rates.dat", index_col=0)
OI_145 = pd.read_csv(data_dir / "O_I" / "OI_145um_rates.dat", index_col=0)

OI_63_interp_dict = {
    key: PchipInterpolator(temp_grid, OI_63[key]) for key in OI_63.keys()
}
OI_44_interp_dict = {
    key: PchipInterpolator(temp_grid, OI_44[key]) for key in OI_44.keys()
}
OI_145_interp_dict = {
    key: PchipInterpolator(temp_grid, OI_145[key]) for key in OI_145.keys()
}

CI_609 = pd.read_csv(data_dir / "C_I" / "CI_609um_rates.dat", index_col=0)
CI_230 = pd.read_csv(data_dir / "C_I" / "CI_230um_rates.dat", index_col=0)
CI_370 = pd.read_csv(data_dir / "C_I" / "CI_370um_rates.dat", index_col=0)

CI_609_interp_dict = {
    key: PchipInterpolator(temp_grid, CI_609[key]) for key in CI_609.keys()
}
CI_230_interp_dict = {
    key: PchipInterpolator(temp_grid, CI_230[key]) for key in CI_230.keys()
}
CI_370_interp_dict = {
    key: PchipInterpolator(temp_grid, CI_370[key]) for key in CI_370.keys()
}

CII_158 = pd.read_csv(data_dir / "C_II" / "CII_158um_rates.dat", index_col=0)

CII_158_interp_dict = {
    key: PchipInterpolator(temp_grid, CII_158[key]) for key in CII_158.keys()
}

NII_205 = pd.read_csv(data_dir / "N_II" / "NII_205um_rates.dat", index_col=0)
NII_76 = pd.read_csv(data_dir / "N_II" / "NII_76um_rates.dat", index_col=0)
NII_122 = pd.read_csv(data_dir / "N_II" / "NII_122um_rates.dat", index_col=0)

NII_205_interp_dict = {
    key: PchipInterpolator(temp_grid, NII_205[key]) for key in NII_205.keys()
}
NII_76_interp_dict = {
    key: PchipInterpolator(temp_grid, NII_76[key]) for key in NII_76.keys()
}
NII_122_interp_dict = {
    key: PchipInterpolator(temp_grid, NII_122[key]) for key in NII_122.keys()
}

SiII_35 = pd.read_csv(data_dir / "Si_II" / "SiII_35um_rates.dat", index_col=0)

SiII_35_interp_dict = {
    key: PchipInterpolator(temp_grid, SiII_35[key]) for key in SiII_35.keys()
}

NeII_13 = pd.read_csv(data_dir / "Ne_II" / "NeII_13um_rates.dat", index_col=0)

NeII_13_interp_dict = {
    key: PchipInterpolator(temp_grid, NeII_13[key]) for key in NeII_13.keys()
}
