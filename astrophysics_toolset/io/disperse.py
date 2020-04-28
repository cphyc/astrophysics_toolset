"""Support Disperse I/O.

The reader is based on a description of the structure using Kaitai (https://kaitai.io/).
"""

import pandas as pd

from ..utilities.decorators import read_files
from ..utilities.types import PathType
from .disperse_reader import DisperseReader
from .common import IOResult


@read_files(1)
def read(fname: PathType) -> IOResult:
    """Read a disperse NDskl file.

    Parameters
    ----------
    fname : str, filename

    Returns
    -------
    IOResult :
        .data :
            nodes, segments : pandas.DataFrame
                Node and segment data as dataframes
            raw : .disperse_reader.DisperseReader.data
                Raw data block of the NDskl file
        .header : .disperse_reader.DisperseReader.header
            Header block of the NDskl file
    """
    ds = DisperseReader(fname)

    #  # Gather information about nodes
    _v1 = pd.DataFrame(
        [(*ds.data.nodepos[s0.pos_index], s0.index, s0.flags, s0.type)
         for s0 in ds.data.nodesdata_int],
        columns='x y z index flags type'.split()).set_index('index')

    _v2 = pd.DataFrame(
        ds.data.nodesdata,
        columns=[_.replace('\x00', '') for _ in ds.header.nodedata_info],
        index=_v1.index)
    # Parse what's integer as integer
    int_columns = ['parent_index', 'persistence_pair']
    for c in int_columns:
        _v2[c] = _v2[c].astype(int)

    node_ds = pd.concat((_v1, _v2), axis=1)
    node_ds['type_s'] = [('void', 'wall', 'filament', 'peak')[_] for _ in node_ds.type]

    #  # Gather information about segments
    _v1 = pd.DataFrame(
        [(s0.index, s0.nodes[0], s0.nodes[1], s0.prev_seg, s0.next_seg)
         for s0 in ds.data.segdata_int],
        columns=['index', 'node_start', 'node_end', 'seg_prev', 'seg_next'])
    _v1 = _v1.set_index('index')
    _v2 = pd.DataFrame(
        ds.data.segdata,
        columns=[_.replace('\x00', '') for _ in ds.header.segdata_info],
        index=_v1.index)
    # Parse what's integer as integer
    int_columns = ['type', 'orientation']
    for c in int_columns:
        _v2[c] = _v2[c].astype(int)
    # Replace off-bound values by dummy values
    _v1.loc[_v1['seg_prev'] > ds.header.nsegs, 'seg_prev'] = ds.header.nsegs
    _v1.loc[_v1['seg_next'] > ds.header.nsegs, 'seg_next'] = ds.header.nsegs

    seg_ds = pd.concat((_v1, _v2), axis=1)

    seg_ds['node_start_type'] = node_ds['type'].iloc[seg_ds['node_start']].values
    seg_ds['node_end_type'] = node_ds['type'].iloc[seg_ds['node_end']].values

    return IOResult(
        data=dict(nodes=node_ds, segments=seg_ds, raw=ds.data),
        metadata=ds.header
    )
