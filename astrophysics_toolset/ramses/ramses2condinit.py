# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Writing myself the files

# %%
from glob import glob
import os
# from yt.utilities.cython_fortran_utils import FortranFile as FF
from cython_fortran_file import FortranFile as FF

from yt.frontends.ramses.definitions import ramses_header
import numpy as np
import yt

from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from astrophysics_toolset.ramses.hilbert import hilbert3d
from astrophysics_toolset.ramses.oct_handler import Octree

# %%
from yt.frontends.ramses.hilbert import hilbert3d as hilbert3d_yt

ipos = np.random.randint(0, 2**9, size=(2000, 3))

print(ipos)
a = hilbert3d(ipos, 10)
b = hilbert3d_yt(ipos, 10)
np.testing.assert_allclose(a, b)

# %%
ds = yt.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt')


# %%
def read_amr(amr_file, longint=False, quadhilbert=False):
    i8b = 'l' if longint else 'i'
    qdp = 'float128' if quadhilbert else 'float64'
    dp = 'float64'
    with FF(amr_file) as f:
        headers = {}
        for h in ramses_header(headers):
            headers.update(f.read_attrs(h))

        # Read level variables
        shape = headers['nlevelmax'], headers['ncpu']
        headl = f.read_vector('i').reshape(shape)
        taill = f.read_vector('i').reshape(shape)
        numbl = f.read_vector('i').reshape(shape)
        numbtot = f.read_vector(i8b).reshape((10, headers['nlevelmax']))

        # Free memory
        headf, tailf, numbf, used_mem, used_mem_tot = f.read_vector('i')

        # Ordering
        ordering = ''.join(f.read_vector('c').astype(str)).strip()
        ndomain = headers['ncpu'] * 1
        if ordering != 'hilbert':
            raise NotImplementedError
        bound_keys = f.read_vector(qdp).reshape(ndomain + 1)

        # Coarse levels  # should be of length 1 unless there are non-periodic boundaries
        ncoarse = 1
        son = f.read_vector('i').reshape(ncoarse)
        refmap = f.read_vector('i').reshape(ncoarse)
        cpu_map = f.read_vector('i').reshape(ncoarse)

        ret = {'headers': headers}
        for k in ('headl taill numbl numbtot bound_keys son refmap cpu_map').split():
            v = eval(k)
            ret[k] = v


        # Fine levels
        nlevelmax, nboundary, ncpu, ndim, ngridmax = (headers[k] for k in ('nlevelmax', 'nboundary', 'ncpu', 'ndim', 'ngridmax'))

        _level, _ndom, ind_grid, next, prev, parent = (np.zeros(ngridmax, dtype='i') for _ in range(6))
        _level_cell = np.zeros((ngridmax, 8), dtype='i')
        xc = np.zeros((ngridmax, ndim), dtype='d')
        nbor = np.zeros((ngridmax, 2*ndim), dtype='i')
        son, cpu_map, refmap = (np.zeros((ngridmax, 2**ndim), dtype='i') for _ in range(3))

        i = 0
        for ilevel in range(nlevelmax):
            for ibound in range(nboundary+ncpu):
                if ibound <= ncpu:
                    ncache = numbl[ilevel, ibound]  # NOTE: C-order vs. F-order
                    istart = headl[ilevel, ibound]
                else:
                    ncache = numbb[ilevel, ibound-ncpu]
                    istart = headb[ilevel, ibound-ncpu]

                def read(kind):
                    tmp = f.read_vector(kind)
                    return tmp.reshape(ncache)

                if ncache > 0:
                    ind_grid[i:i+ncache]= read('i')
                    next[i:i+ncache]    = read('i')
                    prev[i:i+ncache]    = read('i')
                    xc[i:i+ncache]      = np.stack([read(dp) for i in range(ndim)], axis=-1)
                    parent[i:i+ncache]  = read('i')
                    nbor[i:i+ncache]    = np.stack([read('i') for idim in range(2*ndim)], axis=-1)
                    son[i:i+ncache]     = np.stack([read('i') for idim in range(2**ndim)], axis=-1)
                    cpu_map[i:i+ncache] = np.stack([read('i') for idim in range(2**ndim)], axis=-1)
                    refmap[i:i+ncache]   = np.stack([read('i') for idim in range(2**ndim)], axis=-1)

                    _level[i:i+ncache]  = ilevel + 1  # Fortran is 1-indexed
                    _level_cell[i:i+ncache, :] = ilevel + 2
                    _ndom[i:i+ncache]   = ibound + 1  # Fortran is 1-indexed
                    i += ncache

        ind_grid, next, prev, xc, parent, nbor, son, cpu_map, refmap, _level, _level_cell, _ndom = \
            (_[:i] for _ in (ind_grid, next, prev, xc, parent, nbor, son, cpu_map, refmap, _level, _level_cell, _ndom))

        ret.update(dict(ind_grid=ind_grid, next=next, prev=prev, xc=xc, parent=parent, nbor=nbor, son=son,
                        cpu_map=cpu_map, refmap=refmap, _level=_level, _level_cell=_level_cell, _ndom=_ndom))

        return ret

def convert_ncpus(dirname : str, out_dirname : str):
    pattern = os.path.join(dirname, 'amr_?????.out?????')
    amr_files = sorted(glob(pattern))

    data = {}
    for i, amr_file in enumerate(amr_files):
        print(f'Reading {amr_file}')
        icpu = int(amr_file.split('.out')[1])
        data[icpu] = read_amr(amr_file)  # CPU are indexed by one
    return data


data = convert_ncpus('/home/ccc/Documents/prog/yt-data/output_00080/', None)

# %%
data[1]['refmap'].shape

# %%
nlevelmin = ds.parameters['levelmin']
nlevelmax = data[1]['headers']['nlevelmax']
boxlen = data[1]['headers']['boxlen']
nx_loc = 1
scale = boxlen / nx_loc
ndim = data[1]['headers']['ndim']
assert ndim == 3

# %%
bscale = 2**(nlevelmax+1)
ncode = nx_loc*int(bscale)
bscale = int(bscale/scale)

temp = ncode
for bit_length in range(1, 32+1):
    ncode = ncode//2
    if ncode <= 1:
        break
ncode, bit_length, bscale

# %% [markdown]
# At this stage, we have loaded the AMR structure and we make sure we are able to recompute the CPU map. This will be useful in the future.

# %%
dd = np.array([(i, j, k) for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)])  # note the ordering here (for Fortran/C ordering)
dt = data[1]
ngridmax = dt['headers']['ngridmax']
ncoarse = np.product(dt['headers']['nx'])
ii = np.array([i*ngridmax + ncoarse for i in range(8)])

Ncell_tot = 0
for icpu, dt in data.items():
    dx = 1/2/2**dt['_level']
    ixcell = ((dt['xc'][:, None, :] + dx[:, None, None] * dd[None, :, :])*bscale).astype(int)
    dt['_ixgrid'] = (dt['xc'][:, None, :] * bscale).astype(int)
    dt['_ixcell'] = ixcell
    dt['_hilbert_key'] = hilbert3d(ixcell.reshape(-1, 3), bit_length).astype(np.uint64)
    dt['_hilbert_key_grid'] = hilbert3d(dt['_ixgrid'].reshape(-1, 3), bit_length).astype(np.uint64)

    cpu_map = dt['cpu_map'].reshape(-1)
    cpu_map_with_keys = np.digitize(dt['_hilbert_key'], dt['bound_keys'])
    assert np.all(cpu_map == cpu_map_with_keys)

    assert np.allclose(dt['_hilbert_key'], hilbert3d(ixcell.reshape(-1, 3), bit_length))

    dt['_ind_cell'] = dt['ind_grid'][:, None] + ii[None, :]

    Ncell_tot += (dt['cpu_map'] == icpu).sum()

# %% [markdown]
# Store global data

# %%
ind_glob = np.empty((Ncell_tot, 3), dtype=np.int64, order='F')  # first dim: local cell index, second: icpu, third: global index
ixcell_glob = np.empty((Ncell_tot, 3), dtype=np.int32, order='F')
cpu_map_glob, lvl_glob, son_glob = (np.empty((Ncell_tot), dtype=np.int32) for _ in range(3))
hilbert_keys_glob = np.empty(Ncell_tot, dtype=np.float64)
i = 0
for icpu, dt in data.items():
    mask = dt['cpu_map'] == icpu
    N = mask.sum()
    ind_glob[i:i+N, 0] = dt['_ind_cell'][mask]
    ind_glob[i:i+N, 1] = icpu
    ind_glob[i:i+N, 2] = np.arange(i, i+N)

    ixcell_glob[i:i+N, :] = dt['_ixcell'][mask, :]
    hilbert_keys_glob[i:i+N] = dt['_hilbert_key'][mask.flatten()].astype(np.uint64)
    cpu_map_glob[i:i+N] = dt['cpu_map'][mask]
    itmp = np.array([[1]*8])
    lvl_glob[i:i+N] = (dt['_level'][:, None]+itmp)[mask]
    son_glob[i:i+N] = dt['son'][mask]
    i += N

# %% [markdown]
# Recompute CPU map using new keys

# %%
bound_key_orig = interp1d(np.arange(dt['headers']['ncpu']+1), dt['bound_keys'])

old_ncpu = dt['headers']['ncpu']
new_ncpu = 16
new_bound_keys = bound_key_orig(np.linspace(0, dt['headers']['ncpu'], new_ncpu+1))

cpu_map_new = np.digitize(hilbert_keys_glob, new_bound_keys)

# %%
# # Add cells to oct
# oct = Octree(nlevelmax)

# N2 = 8
# for icpu, dt in tqdm(data.items()):
#     ipos = dt['_ixcell'].reshape(-1, 3)
#     file_ind = (dt['ind_grid'].astype(np.int64)[:, None] + np.arange(8)[None, :]*ngridmax + ncoarse).reshape(-1)
#     domain_ind = dt['cpu_map'].flatten().astype(np.int64)
#     new_domain_ind = np.digitize(dt['_hilbert_key'], new_bound_keys)
#     lvl_ind = (dt['_level'].astype(np.int64)[:, None] + np.zeros(8, dtype=np.int64)[None, :]).reshape(-1)

#     N2 += oct.add(ipos, file_ind, domain_ind, new_domain_ind, dt['_hilbert_key'], lvl_ind)

# %%
oct = Octree(nlevelmin, nlevelmax, old_ncpu=old_ncpu, new_ncpu=new_ncpu)
N2 = 1
for icpu, dt in tqdm(data.items()):
    mask = dt['_level'] <= 99999
    ipos = dt['_ixgrid'][mask].reshape(-1, 3)
    file_ind = dt['ind_grid'][mask].astype(np.int64)
    domain_ind = np.digitize(dt['_hilbert_key_grid'][mask], dt['bound_keys'])
    new_domain_ind = np.digitize(dt['_hilbert_key_grid'][mask], new_bound_keys)
    dt['_new_cpu_map'] = np.digitize(dt['_hilbert_key'], new_bound_keys).astype(np.int32).reshape(-1, 8)

    lvl_ind = dt['_level'][mask].astype(np.int64)

    original_domain = np.full_like(new_domain_ind, icpu)
    N2 += oct.add(ipos, file_ind, domain_ind, original_domain, new_domain_ind, dt['_hilbert_key_grid'], lvl_ind)

print(f'Inserted {N2} octs')

# %%
oct.print_tree(3, print_neighbours=True)

# %%
dx_neigh0 = np.array([[-1, 1,  0, 0,  0, 0],
                      [ 0, 0, -1, 1,  0, 0],
                      [ 0, 0,  0, 0, -1, 1]]).T[None, :, :]


for icpu, dt in tqdm(data.items()):
    lvl = dt['_level'].astype(int)
    xc = dt['xc']
    dx = 1/2**(lvl-1)
    xc_neigh = (xc[:, None, :] + dx_neigh0 * dx[:, None, None]) % 1

    ixc = (xc * bscale).astype(int).copy()
    ixc_neigh = (xc_neigh * bscale).astype(int).copy()

    # Set neighbours for all octs that do have a child
    oct.set_neighbours(ixc, ixc_neigh, lvl)

# %%
oct.print_tree(3, print_neighbours=True)

# %%
dt['headers']

# %%
LONGINT = False
QUADHILBERT = False

QUADHILBERT = 'float128' if QUADHILBERT else 'float64'
LONGINT = 'int64' if LONGINT else 'int32'

_HEADERS = (
    ('ncpu', 'i'),
    ('ndim', 'i'),
    ('nx', 'i'),
    ('nlevelmax', 'i'),
    ('ngridmax', 'i'),
    ('nboundary', 'i'),
    ('ngrid_current', 'i'),
    ('boxlen', 'd'),
    ('nout', 'i'),
    ('tout', 'd'),
    ('aout', 'd'),
    ('t', 'd'),
    ('dtold', 'd'),
    ('dtnew', 'd'),
    ('nstep', 'i'),
    ('stat', 'd'),
    ('cosm', 'd'),
    ('timing', 'd'),
    ('mass_sph', 'd')
)

_AMR_STRUCT = (
    ('headl', 'i'),
    ('taill', 'i'),
    ('numbl', 'i'),
    ('numbtot', LONGINT),
    # BOUNDARIES: not supported
    (('headf', 'tailf', 'numbf', 'used_mem', 'used_mem_tot'), 'i'),
    ('ordering', 128),  # 128-long char array
    ('bound_keys', QUADHILBERT),
)


def write_amr_file(headers, amr_struct, amr_file, original_files, original_offsets):
    """This write the new amr files

    Parameters
    ----------
    """

    # Make sure arrays have the right types
    def convert(key, dtype):
        amr_struct[key] = amr_struct[key].astype(dtype)

    for key in ('headl', 'numbl', 'ind_grid', 'next', 'prev',
                'parent', 'nbor', 'son', 'cpu_map', 'refmap'):
        convert(key, np.int32)
    for key in ('xc', ):
        convert(key, np.float64)
    # Write headers
    f = FF(amr_file, mode='w')
    print('Headers')
    for k, t in _HEADERS:
        tmp = np.atleast_1d(headers[k]).astype(t)
        print(k, end='...')
        f.write_vector(tmp)
    print()

    # Write global AMR structure
    print('Gobal AMR structure')
    for k, t in _AMR_STRUCT:
        if not isinstance(t, int):
            tmp = np.atleast_1d(amr_struct[k]).astype(t)
        else:
            tmp = np.char.asarray(amr_struct[k].ljust(128).encode(), 1).astype('c')
        print(f'\tWriting {k}')
        f.write_vector(tmp)

    # Coarse level
    headl = amr_struct['headl']
    numbl = amr_struct['numbl']
    son = amr_struct['son']
    refmap = amr_struct['refmap']
    cpu_map = amr_struct['cpu_map']
    print('Coarse level')
    ncoarse = np.product(headers['nx'])
    f.write_vector(son[:ncoarse])
    f.write_vector(refmap[:ncoarse])
    f.write_vector(cpu_map[:ncoarse])

    print('Fine levels')
    nlevelmax = headers['nlevelmax']
    ncpu = headers['ncpu']
    nboundary = headers['nboundary']
    if nboundary > 0:
        raise NotImplementedError

    ii = ncoarse
    ncache = 0
    def write_chunk(key, extra_slice=...):
        f.write_vector(amr_struct[key][ii:ii+ncache, extra_slice])

    for ilvl in range(nlevelmax):
        for ibound in range(ncpu+nboundary):
            if ibound < ncpu:
                ncache = numbl[ilvl, ibound]
            else:
                ncache = numbb[ilvl, ibound]
            if ncache == 0:
                continue
            write_chunk('ind_grid')
            write_chunk('next')
            write_chunk('prev')
            for idim in range(3):
                write_chunk('xc', idim)
            write_chunk('parent')
            for idim in range(2*3):
                write_chunk('nbor', idim)
            for idim in range(2**3):
                write_chunk('son', idim)
            for idim in range(2**3):
                write_chunk('cpu_map', idim)
            for idim in range(2**3):
                write_chunk('refmap', idim)


# %%
# new_data = {}

# for new_icpu in range(1, new_ncpu+1):
#     bk_low = new_bound_keys[new_icpu-1]
#     bk_up = new_bound_keys[new_icpu]

#     print(f'Selecting grid intersecting with new cpu #{new_icpu}')

#     counts = []
#     oct.clear_paint()
#     amr_struct = oct.domain_info(new_icpu, bk_low, bk_up)
#     new_data[new_icpu] = amr_struct

# %%
# for icpu, dt in data.items():
#     print(icpu, '—'*140)
#     for ilvl, l in enumerate(dt['numbl']):
#         print(ilvl+1, end='\t|\t')
#         for c in l:
#             print(c, end='\t')
#         print('|', l.sum())
# print(dt['numbl'].sum())

# %%
# for icpu, dt in new_data.items():
#     print(icpu, '—'*140)
#     for ilvl, l in enumerate(dt['numbl']):
#         print(ilvl+1, end='\t|\t')
#         for c in l:
#             print(c, end='\t')
#         print('|', l.sum())
# print(dt['numbl'].sum())

# %%
new_data = {}
for new_icpu in range(1, new_ncpu+1):
    bk_low = new_bound_keys[new_icpu-1]
    bk_up = new_bound_keys[new_icpu]

    print(f'Selecting grid intersecting with new cpu #{new_icpu}')

    counts = []

    oct.clear_paint()

    # Select octs that intersect with domain
    # NOTE: possible discrepancy, in RAMSES, we selects octs that contain
    #       at least one cell that intersects. Is it the same?
#     for icpu, dt in data.items():
#         lvl = dt['_level'].astype(np.uint8)
#         hkg = dt['_hilbert_key_grid'].astype(np.uint64)

#         ishift = 3*(bit_length-lvl+1)
#         order_min = (hkg >> ishift)
#         order_max = (order_min + 1) << ishift
#         order_min <<= ishift

#         mask = (order_max > bk_low) & (order_min < bk_up)
#         counts.append(mask.sum())

#         oct.select(dt['_ixgrid'][mask].reshape(-1, 3),
#                    dt['_level'][mask].astype(np.int64))
#     for icpu, dt in data.items():
#         lvl = dt['_level_cell'].astype(np.uint8).flatten()
#         hkg = dt['_hilbert_key'].astype(np.uint64)

#         ishift = 3*(bit_length-lvl+1)
#         order_min = (hkg >> ishift)
#         order_max = (order_min + 1) << ishift
#         order_min <<= ishift

#         mask = (order_max > bk_low) & (order_min < bk_up)
#         counts.append(mask.sum())

#         oct.select(dt['_ixcell'].reshape(-1, 3)[mask],
#                    lvl[mask].astype(np.int64))

    ###########################################################
    # Headers
    dt = data[1]
    headers = dt['headers'].copy()
    headers['ncpu'] = new_ncpu

    ###########################################################
    # Amr structure
    amr_struct = oct.domain_info(new_icpu, bk_low, bk_up)
    file_inds = amr_struct['file_ind']
    nocts = len(file_inds)

    amr_struct['bound_keys'] = new_bound_keys
    amr_struct['numbtot'] = dt['numbtot']
    amr_struct[('headf', 'tailf', 'numbf', 'used_mem', 'used_mem_tot')] = 0, 0, 0, 0, 0
    amr_struct['ordering'] = 'hilbert'

    def pair2icell(v):
        # Compute cell indices from parent oct + icell
        return v[..., 0] + ncoarse + v[..., 1] * ngridmax

    amr_struct['parent'] = pair2icell(amr_struct['parent'])
    amr_struct['nbor'] = pair2icell(amr_struct['nbor'])
    amr_struct['cpu_map'] = amr_struct['new_domain_ind']
    amr_struct['ind_grid'] = np.arange(1, nocts+1, dtype=np.int32)
    new_data[new_icpu] = amr_struct

    # Compute refmap, etc.
    fields = (('refmap', 'refmap'), ('xc', 'xc'), ('cpu_map', '_new_cpu_map'))

    data_out = {}

    dt = data[1]
    for f_new, f_old in fields:
        data_out[f_new] = np.zeros(
            [nocts] + list(dt[f_old].shape[1:]),
            dtype=dt[f_old].dtype
        )

    for icpu, dt in data.items():
        mask = amr_struct['old_domain_ind'] == icpu
        find = file_inds[mask] - 1
        for f_new, f_old in fields:
            data_out[f_new][mask] = dt[f_old][find]

    for k, v in data_out.items():
        amr_struct[k] = v

    # Write AMR file
    base = 'output_00080'
    os.makedirs(base, exist_ok=True)

    amr_file = os.path.join(base, f'amr_00001.out{new_icpu:05d}')

    write_amr_file(headers, amr_struct, amr_file, None, None)

# %%
for l in data[1]['numbl']:
    for c in l:
        print('%7d' % c, end='')
    print()

# %%
for l in new_data[1]['numbl']:
    for c in l:
        print('%7d' % c, end='')
    print()

# %%
dt, dt0 = new_data[1], data[1]

# %%
dt['xc'][:10]


# %%
def inspect(dt, imin=0, imax=8):
    sl = slice(imin, imax)
    bb = 2**17
    i,j,k=(dt['xc'][sl]*bb).T
    ijk=i+2*j+4*k
    order =np.argsort(ijk)
    print('xc')
    print(dt['xc'][sl][order])
    print('')
    print('son')
    print(dt['son'][sl][order])
    
    print('')
    print('parent')
    print(dt['parent'][sl][order])


# %%
inspect(dt)

# %%
inspect(dt0)

# %%
