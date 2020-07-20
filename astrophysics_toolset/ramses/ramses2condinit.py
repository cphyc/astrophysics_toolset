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
from scipy.io import FortranFile as FF
from glob import glob
import os
from yt.utilities.cython_fortran_utils import FortranFile as FF

from yt.frontends.ramses.definitions import ramses_header
import numpy as np
import yt

from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from astrophysics_toolset.ramses.hilbert import hilbert3d
from astrophysics_toolset.ramses.oct_handler import Octree

# %%
ds = yt.load('output_00080/info_00080.txt')


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
        flag1 = f.read_vector('i').reshape(ncoarse)
        cpu_map = f.read_vector('i').reshape(ncoarse)
        
        ret = {'headers': headers}
        for k in ('headl taill numbl numbtot bound_keys son flag1 cpu_map').split():
            v = eval(k)
            ret[k] = v
        
        
        # Fine levels
        nlevelmax, nboundary, ncpu, ndim, ngridmax = (headers[k] for k in ('nlevelmax', 'nboundary', 'ncpu', 'ndim', 'ngridmax'))
        
        _level, _ndom, ind_grid, next, prev, father = (np.zeros(ngridmax, dtype='i') for _ in range(6))
        xc = np.zeros((ngridmax, ndim), dtype='d')
        nbor = np.zeros((ngridmax, 2*ndim), dtype='i')
        son, cpu_map, flag1 = (np.zeros((ngridmax, 2**ndim), dtype='i') for _ in range(3))
        
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
                    father[i:i+ncache]  = read('i')
                    nbor[i:i+ncache]    = np.stack([read('i') for idim in range(2*ndim)], axis=-1)
                    son[i:i+ncache]     = np.stack([read('i') for idim in range(2**ndim)], axis=-1)
                    cpu_map[i:i+ncache] = np.stack([read('i') for idim in range(2**ndim)], axis=-1)
                    flag1[i:i+ncache]   = np.stack([read('i') for idim in range(2**ndim)], axis=-1)
                    
                    _level[i:i+ncache]  = ilevel + 1  # Fortran is 1-indexed
                    _ndom[i:i+ncache]   = ibound + 1  # Fortran is 1-indexed
                    i += ncache
                    
        ind_grid, next, prev, xc, father, nbor, son, cpu_map, flag1, _level, _ndom = \
            (_[:i] for _ in (ind_grid, next, prev, xc, father, nbor, son, cpu_map, flag1, _level, _ndom))

        ret.update(dict(ind_grid=ind_grid, next=next, prev=prev, xc=xc, father=father, nbor=nbor, son=son,
                        cpu_map=cpu_map, flag1=flag1, _level=_level, _ndom=_ndom))
        
        return ret

def convert_ncpus(dirname:str, out_dirname:str):
    pattern = os.path.join(dirname, 'amr_?????.out?????')
    amr_files = sorted(glob(pattern))
    
    data = {}
    for i, amr_file in enumerate(amr_files):
        print(f'Reading {amr_file}')
        icpu = int(amr_file.split('.out')[1])
        data[icpu] = read_amr(amr_file)  # CPU are indexed by one
    return data

data = convert_ncpus('output_00080/', None) 

# %%
nlevelmax = data[1]['headers']['nlevelmax']
boxlen = data[1]['headers']['boxlen']
nx_loc = 1
scale = boxlen / nx_loc
ndim = data[1]['headers']['ndim']
assert ndim == 3

# %%
bscale=2**(nlevelmax+1)
ncode=nx_loc*int(bscale)
bscale=int(bscale/scale)

temp=ncode
for bit_length in range(1, 32+1):
    ncode=ncode//2
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
new_ncpu = 8
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
oct = Octree(nlevelmax, old_ncpu=old_ncpu, new_ncpu=new_ncpu)
N2 = 1
for icpu, dt in tqdm(data.items()):
    mask = dt['_level'] <= 99999
    ipos = dt['_ixgrid'][mask].reshape(-1, 3)
    file_ind = dt['ind_grid'][mask].astype(np.int64)
    domain_ind = np.digitize(dt['_hilbert_key_grid'][mask], dt['bound_keys'])
    new_domain_ind = np.digitize(dt['_hilbert_key_grid'][mask], new_bound_keys)

    lvl_ind = dt['_level'][mask].astype(np.int64)
    
    N2 += oct.add(ipos, file_ind, domain_ind, new_domain_ind, dt['_hilbert_key_grid'], lvl_ind)
    
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
def write_new_amr(file_ind, domain_ind, new_domain_ind, lvl_ind, nbor_ind):
    """This write the new amr files
    
    Parameters
    ----------
    file_ind, domain_ind : np.ndarray
        The position and domain in the _original_ mapping
    TODO
    """
    # Build the AMR struct
    headl = np.zeros((ncpu, nlevelmax))
    taill = np.zeros((ncpu, nlevelmax))
    numbl = np.zeros((ncpu, nlevelmax))
    numbtot = np.zeros((10, nlevelmax))
    headf, tailf, numbf, used_mem, used_mem_tot = 0
    ordering = 'hilbert'
    
    
    # Write it out    

# %%
new_data = {}
for new_icpu in range(1, new_ncpu+1):
    bk_low = new_bound_keys[new_icpu-1]
    bk_up = new_bound_keys[new_icpu]
    
    print(f'Selecting grid intersecting with new cpu #{new_icpu}')
    
    counts = []
    
    oct.clear_paint()

    for icpu, dt in data.items():
        lvl = dt['_level'].astype(np.uint8)
        hkg = dt['_hilbert_key_grid'].astype(np.uint64)

        ishift = 3*(bit_length-lvl+1)
        order_min = (hkg >> ishift)
        order_max = (order_min + 1) << ishift
        order_min <<= ishift
        
        mask = (order_max > bk_low) & (order_min < bk_up)
        counts.append(mask.sum())
        
        oct.select(dt['_ixgrid'][mask].reshape(-1, 3), dt['_level'][mask].astype(np.int64))

    # Extract the file inds
    print('| ', end='')
    for i, c in enumerate(counts):
        print(f'{i+1:7d}', end=' | ')
    print('\n| ', end='')
    for i, c in enumerate(counts):
        print(f'{c:7d}', end=' | ')
    print()
    tmp = oct.domain_info()
    # Compute cell indices from parent oct + icell
    tmp['parent'] = tmp['parent'][:, 0] + ncoarse + tmp['parent'][:, 1] * ngridmax
    tmp['nbor'] = tmp['nbor'][:, :, 0] + ncoarse + tmp['nbor'][:, :, 1] * ngridmax
    new_data[new_icpu] = tmp

# %%
# for icpu, dt in data.items():
#     print(icpu, '—'*140)
#     for ilvl, l in enumerate(dt['headl']):
#         print(ilvl+1, end='\t|\t')
#         for c in l:
#             print(c, end='\t')
#         print()

# %%
(dt['parent'] - ncoarse) % ngridmax

# %%
# for icpu, dt in new_data.items():
#     print(icpu, '—'*140)
#     for ilvl, l in enumerate(dt['headl']):
#         print(ilvl+1, end='\t|\t')
#         for c in l:
#             print(c, end='\t')
#         print()

# %%
