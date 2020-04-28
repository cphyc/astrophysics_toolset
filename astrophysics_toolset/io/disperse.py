"""Support Disperse I/O.

The reader is based on a description of the structure using Kaitai (https://kaitai.io/).
"""

import pandas as pd
import numpy as np

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
    ds = DisperseReader.from_file(fname)

    # Gather data
    nnode, nnode_data, ndim = ds.header.nnode, ds.header.nnode_data, ds.header.ndim
    node_pos = np.asarray(ds.data.node_pos).reshape((nnode, ndim))
    node_data = np.asarray(ds.data.node_data).reshape((nnode, nnode_data))

    nseg, nseg_data = ds.header.nseg, ds.header.nseg_data
    seg_data = np.asarray(ds.data.seg_data).reshape((nseg, nseg_data))

    #  # Gather information about nodes
    _v1 = pd.DataFrame(
        [(*node_pos[s0.pos_index], s0.index, s0.flags, s0.type)
         for s0 in ds.data.node_data_int],
        columns=['x', 'y', 'z', 'index', 'flags', 'type']).set_index('index')

    _v2 = pd.DataFrame(
        node_data,
        columns=[_.replace('\x00', '') for _ in ds.header.node_data_info],
        index=_v1.index)
    # Parse what's integer as integer
    int_columns = ['parent_index', 'persistence_pair']
    for c in int_columns:
        _v2[c] = _v2[c].astype(int)

    node_ds = pd.concat((_v1, _v2), axis=1)
    node_ds['type_s'] = [('void', 'wall', 'filament', 'peak')[_] for _ in node_ds.type]

    #  # Gather information about segments
    _v1 = pd.DataFrame(
        [(s0.index, s0.node_ids[0], s0.node_ids[1], s0.prev_seg, s0.next_seg)
         for s0 in ds.data.seg_data_int],
        columns=['index', 'node_start', 'node_end', 'seg_prev', 'seg_next'])
    _v1 = _v1.set_index('index')
    _v2 = pd.DataFrame(
        seg_data,
        columns=[_.replace('\x00', '') for _ in ds.header.seg_data_info],
        index=_v1.index)
    # Parse what's integer as integer
    int_columns = ['type', 'orientation']
    for c in int_columns:
        _v2[c] = _v2[c].astype(int)
    # Replace off-bound values by dummy values
    _v1.loc[_v1['seg_prev'] > ds.header.nseg, 'seg_prev'] = ds.header.nseg
    _v1.loc[_v1['seg_next'] > ds.header.nseg, 'seg_next'] = ds.header.nseg

    seg_ds = pd.concat((_v1, _v2), axis=1)

    seg_ds['node_start_type'] = node_ds['type'].iloc[seg_ds['node_start']].values
    seg_ds['node_end_type'] = node_ds['type'].iloc[seg_ds['node_end']].values

    return IOResult(
        data=dict(nodes=node_ds, segments=seg_ds, raw=ds.data),
        metadata=ds.header
    )
