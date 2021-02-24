import numpy as np
import yt
from yt.data_objects.selection_objects.data_selection_objects import (
    YTSelectionContainer,
)
from yt.data_objects.static_output import Dataset


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

    data = {
        (ftype, fname): data_source[ftype, fname]
        for (ftype, fname) in ds.derived_field_list
        if ftype == ptype
    }
    periodicity = (
        np.asarray(ds.periodicity)
        & (ds.domain_left_edge == data_source.left_edge)
        & (ds.domain_right_edge == data_source.right_edge)
    )

    sph_ds = yt.load_particles(data, data_source=data_source, periodicity=periodicity)
    sph_ds.add_sph_fields()
