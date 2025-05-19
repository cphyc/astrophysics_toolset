from typing import Any
from collections.abc import Callable

from tqdm import tqdm
import yt

from yt.data_objects.data_containers import YTDataContainer
from scipy.spatial import KDTree
import numpy as np
import networkx as nx


def find_clumps(
    data_source: YTDataContainer,
    callbacks: list[Callable[[YTDataContainer, np.ndarray], Any]] = None,
    filter_callbacks: list[Callable[[YTDataContainer, np.ndarray], bool]] = None,
) -> list[dict]:
    """
    Find clumps in the data source. Works only if the
    input data is from an AMR dataset.

    Parameters
    -----------
    data_source :
        The data source to find clumps in.
    callbacks : list[Callable]
        A list of callback functions to be called with the clumps found.
        Each callback takes two arguments: the data source and the indices of the
        cells forming a clump. It returns a list of properties.
    filter_callbacks : list[Callable]
        A list of callback that returns True or False for each clump.

    Returns
    --------
    list:
        A list of clumps found in the data source.
    """

    if callbacks is None:
        callbacks = []
    if filter_callbacks is None:
        filter_callbacks = []

    fields = [*(("gas", axis) for axis in "xyz"), ("gas", "dx")]

    # Query data from the data source
    data_source.get_data(fields)

    # Get cell positions
    xc = np.stack(
        [data_source["gas", axis].to("unitary").value for axis in "xyz"],
        axis=-1,
    )
    dx = data_source["gas", "dx"].to("unitary").value

    yt.mylog.info("Building KDTree")
    tree = KDTree(xc, boxsize=data_source.ds.domain_width.to("unitary").value[0])

    yt.mylog.info("Finding neighbours")
    # Use Manhattan distance since we only want ±xyz
    dists, inds = tree.query(xc, 24, p=1, workers=-1)

    yt.mylog.info("Discarding non-connected cells")
    mask = dists <= (1.5 * dx[..., None])

    yt.mylog.info("Building graph")
    graph = nx.from_edgelist(
        tqdm(
            ((int(i), int(j)) for i in range(len(xc)) for j in inds[i][mask[i]]),
            total=mask.sum(),
        )
    )

    yt.mylog.info("Finding connected components")
    components = [
        list(c)
        for c in tqdm(nx.connected_components(graph))
        if all(cb(data_source, c) for cb in filter_callbacks)
    ]

    return [
        {
            "indices": c,
        }
        | {cb.__name__: cb(data_source, c) for cb in callbacks}
        for c in components
    ]


def filter_size(min_size):
    def _filter(data, inds):
        return len(inds) > min_size

    return _filter


def total_mass(data, inds):
    return data["gas", "cell_mass"][inds].sum().to("Msun")


def average_temperature(data, inds):
    return np.average(
        data["gas", "temperature"][inds],
        weights=data["gas", "cell_mass"][inds],
    )
