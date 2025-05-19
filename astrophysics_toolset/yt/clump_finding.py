from typing import Any
from collections.abc import Callable

from tqdm import tqdm
import yt

from yt.data_objects.data_containers import YTDataContainer
from scipy.spatial import KDTree
import numpy as np
import networkx as nx


def watershed_split(G: nx.Graph, value_attr="density", min_value=0):
    """
    Split the graph G into subgraphs based on density maxima.

    Each output subgraph corresponds to a region of locally maximal density.
    """
    visited = set()
    regions = []

    # Sort nodes by descending density
    nodes_by_density = sorted(
        (n for n in G.nodes if G.nodes[n][value_attr] > min_value),
        key=lambda n: G.nodes[n][value_attr],
        reverse=True,
    )

    for node in tqdm(nodes_by_density):
        # Already added to a region, no need to do it again
        if node in visited:
            continue

        # Start a region
        region = set()
        threshold = G.nodes[node][value_attr]

        stack = [(node, threshold)]

        while stack:
            current, value = stack.pop()
            visited.add(current)
            region.add(current)

            # Decrease threshold if necessary
            threshold = min(threshold, value)
            # print(f"{i:3d}: added {current} ({threshold=:.2e})")

            # Add neighbours that have a density lower than the threshold
            # Note: we keep on searching as long as the threshold is above
            # some critical value
            for neighbor in G.neighbors(current):
                neighbor_value = G.nodes[neighbor][value_attr]
                if neighbor not in visited and (
                    neighbor_value <= threshold or min_value <= threshold
                ):
                    stack.append((neighbor, neighbor_value))

        if region:
            regions.append((threshold, region))

    return regions


def find_clumps(
    data_source: YTDataContainer,
    field: tuple[str, str],
    min_value: float,
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
    field : tuple
        The field to consider when creating the clumps.
    min_value : float
        Minimum value to consider a clump a clump.
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

    fields = [*(("gas", axis) for axis in "xyz"), ("gas", "dx"), field]

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

    qty = data_source[field]
    if hasattr(min_value, "units"):
        min_value = min_value.to(qty.units).d
    qty = qty.d

    yt.mylog.info("Building graph")
    graph = nx.Graph()
    graph.add_nodes_from(
        (i, {"value": qty}) for i, qty in enumerate(data_source[field].value)
    )
    graph.add_edges_from(
        tqdm(
            ((int(i), int(j)) for i in range(len(xc)) for j in inds[i][mask[i]]),
            total=mask.sum(),
        )
    )

    yt.mylog.info("Finding connected components")
    components = [
        (threshold, list(inds))
        for threshold, inds in watershed_split(
            graph, value_attr="value", min_value=min_value
        )
        if all(cb(data_source, inds) for cb in filter_callbacks)
    ]

    return [
        {"indices": inds, "threshold": threshold}
        | {cb.__name__: cb(data_source, inds) for cb in callbacks}
        for threshold, inds in components
    ]


def filter_size(min_size):
    def _filter(data, inds):
        return len(inds) >= min_size

    return _filter


def total_mass(data, inds):
    return data["gas", "cell_mass"][inds].sum().to("Msun")


def average_temperature(data, inds):
    return np.average(
        data["gas", "temperature"][inds],
        weights=data["gas", "cell_mass"][inds],
    )


if __name__ == "__main__":
    import yt

    ds = yt.load(
        "/home/cphyc/Documents/prog/yt-data/DICEGalaxyDisk_nonCosmological/output_00002/"
    )
    ad = ds.all_data()
    ad2 = ad.include_inside(("gas", "density"), 0.5, 2, units="mp/cm**3")
    clumps = find_clumps(
        ad,
        ("gas", "density"),
        ds.quan(0.1, "mp/cm**3"),
        filter_callbacks=[filter_size(10)],
        callbacks=[total_mass, average_temperature],
    )
