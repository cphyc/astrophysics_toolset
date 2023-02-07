"""Define useful functions for working with stars."""

from pathlib import Path
from typing import List, Literal, Optional, Tuple

import joblib
import numpy as np
import yt
from scipy.interpolate import RegularGridInterpolator
from yt.utilities.on_demand_imports import NotAModule

location = Path("~/.cache/fsps").expanduser()
location.mkdir(parents=True, exist_ok=True)
mem = joblib.Memory(location=location)


class FspsImports:
    _name = "fsps"
    _module = None

    def __init__(self):
        try:
            import fsps

            self._module = fsps
        except ImportError:
            self._module = NotAModule(self._name)

    def __getattr__(self, attr):
        return getattr(self._module, attr)


fsps = FspsImports()


@yt.particle_filter(requires=["particle_family"], filtered_type="star")
def young_stars(pfilter, data):
    age = data[pfilter.filtered_type, "star_age"].to("Myr")

    return age < 10


IMFS = {"Salpeter": 0, "Chabrier": 1, "Kroupa": 2, "von Dokkum": 3, "Dave": 4}


@mem.cache
def __helper(fsps_params, bands, metallicities):
    sp = fsps.StellarPopulation(
        **fsps_params,
    )

    magnitudes = {}
    for z in metallicities:
        sp.params["logzsol"] = np.log10(z)
        magnitudes[z] = sp.get_mags(bands=bands)

    return sp.ssp_ages, magnitudes


@mem.cache(ignore=["n_jobs"])
def _compute_table(
    fsps_params: dict,
    *,
    metallicities: np.ndarray,
    bands: list,
    n_jobs=-1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the luminosity table for a given stellar population.

    Parameters
    ----------
    fsps_params : dict
        The parameters to build the stellar population.
    age : np.ndarray, shape (Nage, )
        Stellar ages in Gyr
    metallicity : np.ndarray, shape (Nmet, )
        Stellar metallicity in solar units
    bands : list
        The bands to compute the luminosity for.

    Returns
    -------
    magnitudes : np.ndarray, shape (Nage, Nmet, Nbands)
        Stellar luminosity in solar units
    """
    magnitudes = {}

    if n_jobs == -1:
        n_jobs = 1

    magnitudes = None
    z2ind = {z: i for i, z in enumerate(metallicities)}

    prev_log_ages = None
    for log_ages, mags in joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(__helper)(fsps_params, bands, [met]) for met in metallicities
    ):
        if magnitudes is None:
            magnitudes = np.zeros(
                (
                    log_ages.size,
                    metallicities.size,
                    len(bands),
                )
            )

        imags = [z2ind[m] for m in mags.keys()]
        values = np.stack(list(mags.values()), axis=1)

        magnitudes[:, imags, :] = values

        if prev_log_ages is not None and not np.allclose(log_ages, prev_log_ages):
            raise ValueError("The ages are not the same for all metallicities")
        prev_log_ages = log_ages

    return RegularGridInterpolator(
        (log_ages, metallicities), magnitudes, bounds_error=False, fill_value=np.nan
    )


IMF_T = Optional[Literal["Salpeter", "Chabrier", "Kroupa", "von Dokkum", "Dave"]]


def add_luminosities(
    ds,
    imf: IMF_T = None,
    fsps_params: Optional[dict] = None,
    bands: Optional[List] = None,
    ptype: str = "star",
):
    if imf is None:
        imf = "Chabrier"
    if imf not in IMFS:
        raise ValueError(f"{imf} is not a valid IMF. Choose one of {IMFS.keys()}")

    if fsps_params is None:
        fsps_params = {}

    if bands is None:
        bands = fsps.find_filter("jwst")

    fsps_params["imf_type"] = IMFS[imf]
    fsps_params["zcontinuous"] = 1
    fsps_params["sfh"] = 0
    fsps_params["logzsol"] = 0.0

    # Pre-compute the magnitude
    #  per metallicity
    #  per stellar age
    #  per band
    all_metallicities = np.geomspace(1e-10, 10, 100)
    zmin = ds.r["star", "particle_metallicity"].min()
    zmax = ds.r["star", "particle_metallicity"].max()

    imin = max(np.argmin((all_metallicities - zmin) ** 2), 1)
    imax = min(np.argmin((all_metallicities - zmax) ** 2), len(all_metallicities) - 1)

    metallicities = all_metallicities[imin - 1 : imax + 1]
    magnitude_interpolator = _compute_table(
        fsps_params, metallicities=metallicities, bands=bands, n_jobs=8
    )

    def band_helper(iband):
        def magnitude(field, data):
            met = data[field.name[0], "particle_metallicity"]
            age = np.log10(data[field.name[0], "star_age"].to("yr"))

            xi = np.stack((age, met), axis=-1)

            return magnitude_interpolator(xi)[:, iband]

        def luminosity(field, data):
            magnitude = data[field.name[0], f"{band}_magnitude"]
            return 10 ** (-magnitude / 2.5)

        return magnitude, luminosity

    # Create the derived quantities
    for iband, band in enumerate(bands):
        mag, lum = band_helper(iband)
        ds.add_field(
            name=(ptype, f"{band}_magnitude"),
            function=mag,
            sampling_type="particle",
            units="",
        )

        ds.add_field(
            name=(ptype, f"{band}_luminosity"),
            function=lum,
            sampling_type="particle",
            units="",
        )
