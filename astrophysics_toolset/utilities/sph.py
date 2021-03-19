import yt
from yt.data_objects.selection_objects.data_selection_objects import (
    YTSelectionContainer,
)
from yt.data_objects.static_output import Dataset

from ..utilities.logging import logger


def create_sph_fields(
    data_source: YTSelectionContainer, ptype: str = "all", *args, **kwargs
) -> Dataset:
    """
    Return a dataset with sph interpolation

    Parameters:
    -----------
    data_source : YTSelectionContainer
        The data source from which particles will be extracted
    ptype : str
        The type of particle to extract
    args, kwargs : extra arguments will be passed to `add_sph_fields`

    Returns
    -------
    ds : Dataset
        A dataset with sph interpolation.
    """
    ds = data_source.ds

    if hasattr(ds, "add_sph_fields"):
        ds.add_sph_fields(*args, **kwargs)
        return ds

    logger.debug("Loading data from data source")
    data = {
        (ptype, fname): data_source[ftype, fname]
        for (ftype, fname) in ds.field_list
        if (ftype == ptype) and (data_source[ftype, fname].ndim == 1)
    }

    # Make sure position and masses are copied
    for k in "xyz":
        data[ptype, f"particle_position_{k}"] = data_source[
            ptype, f"particle_position_{k}"
        ]
    data[ptype, "particle_mass"] = data_source[ptype, "particle_mass"]
    data[ptype, "particle_identity"] = data_source[ptype, "particle_identity"]
    data[ptype, "particle_family"] = data_source[ptype, "particle_family"]

    periodicity = (True, True, True)
    # No support for aperiodicity for now
    # L, R = data_source.get_bbox()
    # periodicity = (
    #     np.asarray(ds.periodicity)
    #     & (ds.domain_left_edge == L)
    #     & (ds.domain_right_edge == R)
    # )

    logger.debug("Create particle dataset")
    sph_ds = yt.load_particles(data, data_source=data_source, periodicity=periodicity)
    sph_ds.current_redshift = ds.current_redshift  # Copy current redshift
    sph_ds.add_sph_fields(*args, **kwargs, sph_ptype=ptype)

    return sph_ds
