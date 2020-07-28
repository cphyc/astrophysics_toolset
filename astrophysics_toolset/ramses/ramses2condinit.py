#!/home/ccc/anaconda3/bin/python
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
# ert# Writing myself the files

# %%
from glob import glob
import os
import shutil
from cython_fortran_file import FortranFile as FF
import sys

from yt.frontends.ramses.definitions import ramses_header
import numpy as np
import yt

from yt.frontends.ramses.field_handlers import (
    FieldFileHandler,
    GravFieldFileHandler,
    HydroFieldFileHandler
)
from yt.frontends.ramses.particle_handlers import (
    DefaultParticleFileHandler,
    ParticleFileHandler
)
from yt.frontends.ramses.hilbert import hilbert3d as hilbert3d_yt

from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from astrophysics_toolset.ramses.hilbert import hilbert3d
from astrophysics_toolset.ramses.oct_handler import Octree

import argparse

import sys

import matplotlib as mpl


# %%
def in_notebook():
    return sys.argv[0].endswith('ipykernel_launcher.py')


# %%
parser = argparse.ArgumentParser()
# parser.add_argument('--input', type=str, default='/home/ccc/Documents/Postdoc/genetIC-angular-momentum-constrain/simulations/data/DM_256_planck/resim/00140/halo_189/relative_change_lx_0.8/output_00001/info_00001.txt',
#     help='Input file'
# )
parser.add_argument('--input', default='output_00002/info_00002.txt')
parser.add_argument('--output-dir', type=str, default='change_ncpu/new_output')
parser.add_argument('--ncpu', type=int, default=4)
parser.add_argument('--longint', action='store_true')
parser.add_argument('--quadhilbert', action='store_true')
parser.add_argument('--nexpand', default=1, type=int, help='Number of times to expand boundaries (default: %(default)s)')
parser.add_argument('--test', action='store_true')

try:   # in notebook
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('rm -rf /home/ccc/Documents/prog/genetIC/genetIC/tests/test_hydro/change_ncpu/output_00005/')
    args = parser.parse_args([
        '--input',
        '/home/ccc/Documents/prog/genetIC/genetIC/tests/test_hydro/output_00005/info_00005.txt',
        '--output-dir',
        '/home/ccc/Documents/prog/genetIC/genetIC/tests/test_hydro/change_ncpu/output_00005/',
        '--ncpu', '4',
        '--longint', '--quadhilbert',
        '--nexpand', '1'
    ])
    print('='*200)
    print('# Running in notebook mode!')
    print('='*200)
except AttributeError:
    args = parser.parse_args()

input_dir = os.path.abspath(os.path.split(args.input)[0])
output_dir = os.path.abspath(args.output_dir)
if input_dir == output_dir or os.path.exists(args.output_dir):
    raise Exception(
        ('Either the destination folder already exists (%s) or '
         'you are trying to write new files in the same folder as the old one (%s).\n'
         'Either way, I am Refusing to do that.\n\n\n'
         'Sorry.') % (output_dir, input_dir)
    )

# Making output dir
_iout = int(os.path.split(args.input)[1].split('.txt')[0].split('_')[1])
CONFIG = dict(
    new_ncpu=args.ncpu,
    iout=_iout
)

# %%
ipos = np.random.randint(0, 2**9, size=(2000, 3))

a = hilbert3d(ipos, 10)
b = hilbert3d_yt(ipos, 10)
np.testing.assert_allclose(a, b)

# %%
ds = yt.load(args.input)
default_headers = {
    'ramses': {},
    'gravity': {'nvar': 4}
}

# ds = yt.load_sample('output_00080', 'info_00080.txt')
# LONGINT, QUADHILBERT = False, False
# default_headers = {
#     'ramses': {},
#     'gravity': {'nvar': 3}
# }


# %%
def read_amr(amr_file, longint=args.longint, quadhilbert=args.quadhilbert):
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
        try:
            numbtot = f.read_vector(i8b).reshape((10, headers['nlevelmax']))
        except ValueError as e:
            raise Exception(
                'Caught an exception while reading numbtot. This is likely due '
                'to you forgetting to (un)set the longint flag!\n'
                'Try calling the script with/out `--longint`.')

        # Free memory
        headf, tailf, numbf, used_mem, used_mem_tot = f.read_vector('i')

        # Ordering
        ordering = ''.join(f.read_vector('c').astype(str)).strip()
        ndomain = headers['ncpu'] * 1
        if ordering != 'hilbert':
            raise NotImplementedError
        try:
            bound_keys = f.read_vector(qdp).reshape(ndomain + 1)
        except ValueError as e:
            raise Exception(
                'Caught an exception while reading hilbert keys. This is likely due '
                'to you forgetting to (un)set the quadhibert flag!\n'
                'Try calling the script with/out `--quadhilbert`.')

        nlevelmax, nboundary, ncpu, ndim, ngridmax = (
            headers[k] for k in ('nlevelmax', 'nboundary', 'ncpu', 'ndim', 'ngridmax')
        )

        # Allocate memory
        _level, _ndom, ind_grid, next, prev, parent = (np.zeros(ngridmax, dtype='i') for _ in range(6))
        _level_cell = np.zeros((ngridmax, 8), dtype='i')
        xc = np.zeros((ngridmax, ndim), dtype='d')
        nbor = np.zeros((ngridmax, 2*ndim), dtype='i')
        son, cpu_map, refmap = (np.zeros((ngridmax, 2**ndim), dtype='i') for _ in range(3))

        # Coarse levels
        # should be of length 1 unless there are non-periodic boundaries
        ncoarse = 1
        coarse_son = f.read_vector('i').reshape(ncoarse)
        coarse_refmap = f.read_vector('i').reshape(ncoarse)
        coarse_cpu_map = f.read_vector('i').reshape(ncoarse)

        ret = {'headers': headers}
        for k in ('headl taill numbl numbtot bound_keys son refmap cpu_map').split():
            v = eval(k)
            ret[k] = v

        # Fine levels
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

        (ind_grid, next, prev, xc, parent,
         nbor, son, cpu_map, refmap,
         _level, _level_cell, _ndom
        ) = (_[:i] for _ in
         (ind_grid, next, prev, xc, parent,
          nbor, son, cpu_map, refmap,
          _level, _level_cell, _ndom)
        )

        ret.update(dict(
            ind_grid=ind_grid,
            next=next,
            prev=prev,
            xc=xc,
            parent=parent,
            nbor=nbor,
            son=son,
            cpu_map=cpu_map,
            refmap=refmap,
            coarse_refmap=coarse_refmap,
            coarse_cpu_map=coarse_cpu_map,
            coarse_son=coarse_son,
            _level=_level,
            _level_cell=_level_cell,
            _ndom=_ndom)
        )

        return ret


def read_all_amr_files(dirname: str):
    pattern = os.path.join(dirname, 'amr_?????.out?????')
    amr_files = sorted(glob(pattern))

    data = {}
    for i, amr_file in enumerate(tqdm(amr_files, desc='Reading AMR')):
        icpu = int(amr_file.split('.out')[1])
        data[icpu] = read_amr(amr_file)  # CPU are indexed by one
    return data

path = os.path.split(ds.parameter_filename)[0]

data = read_all_amr_files(path)

# %%
if args.quadhilbert:
    # Load keys from disk
    with open(glob(os.path.join(path, 'info_?????.txt'))[0], 'r') as f:
        lines = f.readlines()
        
    for i, l in enumerate(lines):
        if l.strip().startswith('DOMAIN'):
            break
    lines = lines[i+1:]
    bound_keys = np.zeros(len(data)+1, dtype=np.float128)
    for i, l in enumerate(lines):
        klow = float(l[11:32])
        kup = float(l[35:-1])
        bound_keys[i] = klow
        bound_keys[i+1] = kup
    for dt in data.values():
        dt['bound_keys'] = bound_keys

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
    cpu_map_with_keys = np.searchsorted(dt['bound_keys'], dt['_hilbert_key'], side='left')
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
new_bound_keys = bound_key_orig(np.linspace(
    0,
    dt['headers']['ncpu'],
    CONFIG['new_ncpu']+1)
)

cpu_map_new = np.searchsorted(new_bound_keys, hilbert_keys_glob, side='left')

# %%
oct = Octree(nlevelmin, nlevelmax, old_ncpu=old_ncpu, new_ncpu=CONFIG['new_ncpu'])
N2 = 1
for icpu, dt in tqdm(data.items()):
    mask = dt['_level'] <= 99999
    ipos = dt['_ixgrid'][mask].reshape(-1, 3)
    file_ind = dt['ind_grid'][mask].astype(np.int64)
    domain_ind = np.searchsorted( dt['bound_keys'], dt['_hilbert_key_grid'][mask], side='left')
    new_domain_ind = np.searchsorted(new_bound_keys, dt['_hilbert_key_grid'][mask], side='left')
    dt['_new_cpu_map'] = np.searchsorted(new_bound_keys, dt['_hilbert_key'], side='left').astype(np.int32).reshape(-1, 8)

    lvl_ind = dt['_level'][mask].astype(np.int64)

    original_domain = np.full_like(new_domain_ind, icpu)
    N2 += oct.add(ipos, file_ind, domain_ind=domain_ind, new_domain_ind=new_domain_ind, owning_cpu=original_domain, hilbert_key=dt['_hilbert_key_grid'], refmap=dt['refmap'], lvl=lvl_ind)

print(f'Inserted {N2} octs')

# %%
oct.print_tree(3, print_neighbours=True)

# %%
# def set_neighbours(icpu, dt):
dt = data[1]
igrid0 = dt['ind_grid'][1:]
icell, igrid = np.unravel_index(dt['parent'][igrid0-1]-ncoarse, (8, ngridmax))

np.testing.assert_allclose((((dt['xc'][igrid-1] - dt['xc'][1:]) * 2**dt['_level'][1:, None])**2).sum(axis=1), 3)

# %%
icell, igrid = np.unravel_index(dt['nbor'][1:]-ncoarse, (8, ngridmax))
np.round(dt['xc'][igrid-1]*bscale).astype(np.int64)

def set_neighbours(icpu, dt):
    lvl = dt['_level'].astype(int)[1:]
    igrid0 = dt['ind_grid'][1:]

    icell_neigh, igrid_neigh = np.unravel_index(dt['nbor'][igrid0-1]-ncoarse, (8, ngridmax))
    dd = 1/2**dt['_level'][igrid_neigh-1][..., None]/2
    dx = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
    dy = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    dz = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    xc = dt['xc']
    xc_neigh = xc[igrid_neigh-1] + dd * np.stack([dx[icell_neigh], dy[icell_neigh], dz[icell_neigh]], axis=-1)
    
    ixc = (xc[igrid0-1] * bscale).astype(int).copy()
    ixc_neigh = (xc_neigh * bscale).astype(int).copy()

    # Set neighbours for all octs that do have a child
    oct.set_neighbours(ixc, ixc_neigh, lvl)

# dx_neigh0 = np.array([[-1, 1,  0, 0,  0, 0],
#                       [ 0, 0, -1, 1,  0, 0],
#                       [ 0, 0,  0, 0, -1, 1]]).T[None, :, :]


# def set_neighbours(icpu, dt):
#     lvl = dt['_level'].astype(int)[1:]
#     xc = dt['xc'][1:]
#     dx = 1/2**(lvl-1)
#     xc_neigh = (xc[1:, None, :] + dx_neigh0 * dx[1:, None, None]) % 1

#     ixc = (xc * bscale).astype(int).copy()
#     ixc_neigh = (xc_neigh * bscale).astype(int).copy()

#     # Set neighbours for all octs that do have a child
#     oct.set_neighbours(ixc, ixc_neigh, lvl)

for icpu, dt in tqdm(data.items(), desc='Setting neighbours'):
    set_neighbours(icpu, dt)

# %%
oct.print_tree(3, print_neighbours=True)

# %%
QUADHILBERT = 'float128' if args.quadhilbert else 'float64'
LONGINT = 'int64' if args.longint else 'int32'

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


def write_amr_file(headers, amr_struct, amr_file):
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
    print('Writing headers. ', end='')
    for k, t in _HEADERS:
        tmp = np.atleast_1d(headers[k]).astype(t)
        print(k, end='...')
        f.write_vector(tmp)
    print()

    # Write global AMR structure
    print('Writing global AMR structure. ', end='')
    for k, t in _AMR_STRUCT:
        if not isinstance(t, int):
            tmp = np.ascontiguousarray(np.atleast_1d(amr_struct[k]), dtype=t)
        else:
            tmp = np.char.asarray(amr_struct[k].ljust(128).encode(), 1).astype('c')
        print(f'{k}', end='...')
        f.write_vector(tmp)

    # Coarse level
    headl = amr_struct['headl']
    numbl = amr_struct['numbl']
    son = amr_struct['son']
    refmap = amr_struct['refmap']
    cpu_map = amr_struct['cpu_map']
    print('Coarse level')
    ncoarse = np.product(headers['nx'])
    # NOTE: since son has shape (ncoarse + ngridmax*8, we only need to write cell 0
    #       of coarse level (i.e. root oct)
    f.write_vector(amr_struct['coarse_son'].astype(np.int32))
    f.write_vector(amr_struct['coarse_refmap'].astype(np.int32))
    cpu_map_coarse = np.argwhere(numbl[0] == 1).astype(np.int32).flatten()+1
    assert cpu_map_coarse.size == 1
    f.write_vector(cpu_map_coarse)

    print('Fine levels')
    nlevelmax = headers['nlevelmax']
    ncpu = headers['ncpu']
    nboundary = headers['nboundary']
    if nboundary > 0:
        raise NotImplementedError

    ii = 0
    ncache = 0
    def write_chunk(key, extra_slice=...):
        f.write_vector(np.ascontiguousarray(amr_struct[key][ii:ii+ncache, extra_slice]))

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
                assert np.all(amr_struct['cpu_map'][ii:ii+ncache, idim] > 0)
                write_chunk('cpu_map', idim)
            for idim in range(2**3):
                write_chunk('refmap', idim)

            ii += ncache


# %%
new_data = {}
for new_icpu in range(1, CONFIG['new_ncpu']+1):
    bk_low = new_bound_keys[new_icpu-1]
    bk_up = new_bound_keys[new_icpu]

    print(f'Selecting grid intersecting with new cpu #{new_icpu}')

    oct.clear_paint()

    # Select cells that intersect with domain
    for icpu, dt in data.items():
        lvl = dt['_level_cell'].astype(np.uint8).flatten()[ncoarse*8:]
        hkg = dt['_hilbert_key'].astype(np.uint64)[ncoarse*8:]

        ishift = 3*(bit_length-lvl+1)
        order_min = (hkg >> ishift)
        order_max = (order_min + 1) << ishift
        order_min <<= ishift

        mask = (order_max > bk_low) & (order_min < bk_up)

        n = oct.select(dt['_ixcell'][1:].reshape(-1, 3)[mask],
                       lvl[mask].astype(np.int64))
        print(f'Selected {n} cells')

    ###########################################################
    # Headers
    dt = data[1]
    headers = dt['headers'].copy()
    headers['ncpu'] = CONFIG['new_ncpu']

    ###########################################################
    # Amr structure
    amr_struct = oct.domain_info(new_icpu, nexpand=args.nexpand)
    file_inds = amr_struct['file_ind']
    nocts = len(file_inds)

    amr_struct['bound_keys'] = new_bound_keys
    amr_struct['numbtot'] = dt['numbtot']
    amr_struct[('headf', 'tailf', 'numbf', 'used_mem', 'used_mem_tot')] = 0, 0, 0, 0, 0
    amr_struct['ordering'] = 'hilbert'
    amr_struct['coarse_son'] = dt['coarse_son']
    amr_struct['coarse_refmap'] = dt['coarse_refmap']

    def pair2icell(v):
        # Compute cell indices from parent oct + icell
        return v[..., 0] + ncoarse + v[..., 1] * ngridmax

    amr_struct['parent'] = pair2icell(amr_struct['parent'])
    amr_struct['nbor'] = pair2icell(amr_struct['nbor'])
    amr_struct['cpu_map'] = amr_struct['new_domain_ind']
    amr_struct['ind_grid'] = np.arange(1, nocts+1, dtype=np.int32)
    amr_struct['headers'] = headers
    new_data[new_icpu] = amr_struct

    # Compute refmap, etc.
    # fields = (('refmap', 'refmap'), ('xc', 'xc'), ('cpu_map', '_new_cpu_map'))
    fields = (('xc', 'xc'), ('cpu_map', '_new_cpu_map'))

    data_out = {}

    dt = data[1]
    for f_new, f_old in fields:
        data_out[f_new] = np.zeros(
            [nocts] + list(dt[f_old].shape[1:]),
            dtype=dt[f_old].dtype
        )

    for icpu, dt in data.items():
        mask = amr_struct['owning_cpu'] == icpu
        find = file_inds[mask] - 1
        for f_new, f_old in fields:
            data_out[f_new][mask] = dt[f_old][find]

    for k, v in data_out.items():
        amr_struct[k] = v

    # Make sure ngridmax is large enough
    if nocts > headers['ngridmax']:
        raise RuntimeError(
            'ERROR: you need to increase ngridmax to at least %s!' % nocts
        )

    # Write AMR file
    base = args.output_dir
    os.makedirs(base, exist_ok=True)
    iout = CONFIG['iout']
    amr_file = os.path.join(base, f'amr_{iout:05d}.out{new_icpu:05d}')

    tmp = amr_struct
    write_amr_file(headers, tmp, amr_file)

# %%
print(' old structure '.center(120, '='))
for l in data[1]['numbl']:
    for c in l:
        print('%7d' % c, end='')
    print('| %s' % l.sum())

# %%
print(' new structure '.center(120, '='))

for l in new_data[1]['numbl']:
    for c in l:
        print('%7d' % c, end='')
    print('| %s' % l.sum())

# %%
oct.check_tree(999)

# %%
dt = data[1]

# %%
ds.min_level

# %%
dt['_level_cell'].max()


# %% [markdown]
# Now write hydro

# %%
def fluid_file_reader(field_handler: FieldFileHandler, headers: dict = {}):
    with FF(field_handler.fname, 'r') as fin:
        headers.update(fin.read_attrs(field_handler.attrs))
        nvar = headers['nvar']
        nboundaries = headers['nboundary']
        nlevelmax = headers['nlevelmax']
        ncpu = headers['ncpu']

        data_out = np.full((nvar, 8, field_handler.domain.amr_header['ngrid_current']), 0, dtype='d')

        ii = 0
        for ilvl in range(nlevelmax):
            for icpu in range(ncpu+nboundaries):
                fin.read_int()  # ilvl2
                ncache = fin.read_int()
                if ncache > 0:
                    for icell in range(8):
                        for ivar in range(nvar):
                            data_out[ivar, icell, ii:ii+ncache] = fin.read_vector('d')
                ii += ncache
    return headers, data_out


def fluid_file_writer(fname, headers, data, numbl):
    with FF(fname, mode='w') as fout:
        for key, _len, dtype in headers['_structure']:
            tmp = np.atleast_1d(headers[key]).astype(dtype)
            assert len(tmp) == _len
            fout.write_vector(tmp)
        nvar = headers['nvar']
        nboundaries = headers['nboundary']
        nlevelmax = headers['nlevelmax']
        ncpu = headers['ncpu']

        ii = 0

        for ilvl in range(1, 1+nlevelmax):
            for icpu in range(1, ncpu+nboundaries+1):
                ncache = numbl[ilvl-1, icpu-1]
                fout.write_vector(np.asarray([ilvl], dtype=np.int32))
                fout.write_vector(np.asarray([ncache], dtype=np.int32))
                if ncache > 0:
                    for icell in range(8):
                        for ivar in range(nvar):
                            tmp = data[ivar, icell, ii:ii+ncache]
                            fout.write_vector(np.ascontiguousarray(tmp))
                ii += ncache


def write_fluid_file(filename, amr_structure, headers, data_old):
    ncpu_old = len(data_old)

    # Loop over AMR structure, and read relevant files
    nvar = headers['nvar']
    nocts = amr_structure['owning_cpu'].size
    data_new = np.full((nvar, 8, nocts), np.nan, dtype='d')

    for icpu_old in range(1, 1+ncpu_old):
        mask = amr_structure['owning_cpu'] == icpu_old   # select octs in this file
        n_to_read = mask.sum()
        if n_to_read == 0:
            continue

        # Original positions
        file_ind = amr_structure['file_ind'][mask] - 1

        # Target position
        new_file_ind = amr_structure['ind_grid'][mask] - 1
        data_new[..., new_file_ind] = data_old[icpu_old][..., file_ind]

    # Write file
    fluid_file_writer(filename, headers, data_new, amr_structure['numbl'])


class FluidFileAttrs:
    fname_pattern = None


class GravityFluidFileAttrs(FluidFileAttrs, GravFieldFileHandler):
    fname_pattern = 'grav_{iout:05d}.out{icpu:05d}'
    ftype = None  # prevent yt from using this to detect files


class HydroFluidFileAttrs(FluidFileAttrs, HydroFieldFileHandler):
    fname_pattern = 'hydro_{iout:05d}.out{icpu:05d}'
    ftype = None  # prevent yt from using this to detect files


fluid_descs = {
    'gravity': GravityFluidFileAttrs,
    'ramses': HydroFluidFileAttrs
}


def rewrite_fluid_files(amr_structure, domains, output_dir, iout):
    nkind = len(domains[0].field_handlers)
    ncpu_new = len(amr_structure)
    all_fdescs = [fluid_descs[fh.ftype] for fh in domains[0].field_handlers]

    progress = tqdm(total=nkind)
    for i, fdesc in enumerate(all_fdescs):
        ftype = domains[0].field_handlers[i].ftype
        progress.set_description(f'{ftype}: R')
        data_orig = {}
        headers = {}
        for icpu, dom in enumerate(tqdm(domains, desc='Reading files', leave=False)):
            fh = dom.field_handlers[i]
            ret = fluid_file_reader(fh, default_headers[ftype].copy())
            headers, data_orig[icpu+1] = ret

        # Need to update the number of cpus in the headers
        headers['ncpu'] = ncpu_new

        headers['_structure'] = fdesc.attrs

        progress.set_description(f'{ftype}: W')
        for icpu in tqdm(amr_structure.keys(), desc='Writing files', leave=False):
            fname = os.path.join(
                output_dir, fdesc.fname_pattern.format(iout=iout, icpu=icpu)
            )
            write_fluid_file(fname, amr_structure[icpu], headers, data_old=data_orig)
        progress.update(1)

rewrite_fluid_files(
    new_data, ds.index.domains,
    output_dir=args.output_dir, iout=CONFIG['iout']
)


# %%
class ParticleFileAttrs(DefaultParticleFileHandler):
    fname_pattern = 'part_{iout:05d}.out{icpu:05d}'
    ptype = None  # prevent yt from using this to detect files


particle_descs = {
    'io': ParticleFileAttrs,
}


particle_filename_mapping = {
    'io': 'part_{iout:05d}.out{icpu:05d}',
}


def particle_file_reader(
        particle_handler: ParticleFileHandler,
        headers: dict = {}
    ) -> dict:

    data_out = {}
    with FF(particle_handler.fname, 'r') as fin:
        headers.update(fin.read_attrs(particle_handler.attrs))
        npart = headers['npart']

        for k, dtype in particle_handler.field_types.items():
            data_out[k] = fin.read_vector(dtype)
            assert data_out[k].size == npart

    return headers, data_out


def particle_file_writer(fname, particle_new_domain, headers, data_old, new_icpu):
    # Count number of particles
    npart = 0
    masks = {}
    for icpu_old, part_dom in particle_new_domain.items():
        mask = (part_dom == new_icpu)
        masks[icpu_old] = mask

        npart += mask.sum()

    # Extract fields
    fields = data_old[1].keys()
    print()

    with FF(fname, mode='w') as fout:
        h = headers.copy()
        h['npart'] = npart
        # TODO: nstar, nstar_tot, etc.

        # Write headers
        for key, _len, dtype in h['_structure']:
            print(f'Writing {key}->{h[key]}')
            tmp = np.atleast_1d(h[key]).astype(dtype)
            assert (len(tmp) == _len) or (_len == -1)
            fout.write_vector(tmp)

        # Write fields
        for field in fields:
            vals = np.empty(npart, dtype=data_old[1][field].dtype)

            i0 = 0
            for mask, dt in zip(masks.values(), data_old.values()):
                count = mask.sum()
                vals[i0:i0+count] = dt[field][mask]
                i0 += count

            print(f'\tWriting {field} with dtype {vals.dtype} and size {vals.size}/{npart}')

            fout.write_vector(vals)


def rewrite_particle_files(amr_structure, domains, output_dir, iout):
    nkind = len(domains[0].particle_handlers)
    ncpu_new = len(amr_structure)
    all_pdescs = [particle_descs[fh.ptype] for fh in domains[0].particle_handlers]

    progress = tqdm(total=nkind)
    for i, pdesc in enumerate(all_pdescs):
        ptype = domains[0].particle_handlers[i].ptype
        progress.set_description(f'{ptype}: R')
        data_orig = {}
        particle_new_domain = {}
        headers = {}
        for icpu, dom in enumerate(tqdm(domains, desc='Reading files', leave=False)):
            fh = dom.particle_handlers[i]
            ret = particle_file_reader(fh, {})
            headers, data_orig[icpu+1] = ret
            pos = np.stack(
                [ret[1]['io', 'particle_position_%s' % k] for k in 'xyz'],
                axis=-1
            )
            ipos = np.round(pos * bscale).astype(np.int64)
            particle_new_domain[icpu+1] = np.searchsorted(
                amr_structure[1]['bound_keys'],
                hilbert3d(ipos, bit_length),
                side='left'
            )

        # Need to update the number of cpus in the headers
        headers['ncpu'] = ncpu_new
        headers['_structure'] = pdesc.attrs

        progress.set_description(f'{ptype}: W')
        for icpu in tqdm(amr_structure.keys(), desc='Writing files', leave=False):
            fname = os.path.join(
                output_dir,
                pdesc.fname_pattern.format(iout=iout, icpu=icpu)
            )
            particle_file_writer(
                fname,
                particle_new_domain,
                headers,
                data_old=data_orig,
                new_icpu=icpu
            )
        progress.update(1)


rewrite_particle_files(
    new_data, ds.index.domains,
    output_dir=args.output_dir, iout=CONFIG['iout']
)


# %%
# Final touch: copy namelist, file descriptors and info
def copy_meta(input_info_file, output_dir, bound_key, new_ncpu):
    input_dir, input_info_fname = os.path.split(input_info_file)
    output_info_file = os.path.join(output_dir, input_info_fname)

    # Rewrite "info_XXXXX.txt"
    with open(input_info_file, 'r') as fin:
        lines = fin.readlines()
    with open(output_info_file, 'w') as fout:
        line = lines.pop(0)
        while not line.strip().startswith('DOMAIN   ind_min'):
            if 'ncpu' in line:
                line = 'ncpu        = %10d\n' % new_ncpu
            fout.write(line)
            line = lines.pop(0)

        # Write hilbert keys
        fout.write('   DOMAIN   ind_min                 ind_max\n')
        for icpu in range(1, new_ncpu+1):
            s = f'{icpu:8d}   {bound_key[icpu-1]:.15E}   {bound_key[icpu]:.15E}\n'
            fout.write(s)

    # Copy file descriptors, header & namelist
    tgt_files = glob(os.path.join(input_dir, '*file_descriptor.txt'))
    tgt_files += glob(os.path.join(input_dir, 'header_?????.txt'))
    tgt_files += glob(os.path.join(input_dir, 'namelist.txt'))
    for f_in in tgt_files:
        f_out = os.path.join(output_dir, os.path.split(f_in)[1])
        shutil.copy(f_in, f_out)


copy_meta(args.input, args.output_dir, new_bound_keys, CONFIG['new_ncpu'])
# %%
if not args.test:
    import sys; sys.exit(0)
else:
    print('Testing using yt.'.center(200, '='))
# %%
yt.funcs.mylog.setLevel(40)
# Test reading with yt as a weak test
ds_original = yt.load(args.input)
ds_new = yt.load(os.path.join(args.output_dir, 'info_%05d.txt' % CONFIG['iout']))

images = []
os.makedirs('tmp/frames/', exist_ok=True)
for ds, prefix in reversed(list(zip((ds_new, ds_original, ds_new), ('new', 'ref')))):
    print(f'Loading = {ds}')
    for d in 'xyz':
        p = yt.ProjectionPlot(ds, d, 'density')
        images.extend(p.save(f'tmp/frames/{d}_{prefix}'))

for ds, prefix in zip((ds_original, ds_new), ('ref', 'new')):
    for d in 'xyz':
        p = yt.ProjectionPlot(ds, d, 'DM_cic')
        images.extend(p.save(f'tmp/frames/{d}_{prefix}'))
print('You can compare the images look the same using the images from %s' % images)

# %%
images_with_diff = list(images)

def raw(i):
    return mpl.image.imread(images[i])
for i in range(3):
    new_fname = images[i].replace('_ref', '_diff')
    diff = (raw(3+i) - raw(i) + 1) / 2
    mpl.image.imsave(new_fname, diff)
    images_with_diff.append(new_fname)
    
for i in range(6, 6+3):
    new_fname = images[i].replace('_ref', '_diff')
    diff = (raw(3+i) - raw(i) + 1) / 2
    mpl.image.imsave(new_fname, diff)
    images_with_diff.append(new_fname)

# %%
import ipyplot
print('Note: we expect artifacts with particle CIC deposition due to yt internal deposition (does not depose on other domains)')
ipyplot.plot_class_tabs(np.asarray(images_with_diff), np.asarray(['_'.join(_.split('_')[-3:]) for _ in images_with_diff]), img_width=400)

# %%
# Compare gas cells
order = {}

ad_new = ds_new.all_data()
ad_ref = ds_original.all_data()

for ad, prefix in reversed(list(zip((ad_new, ad_ref), ('new', 'ref')))):
    order[prefix] = np.argsort(ad['index', 'morton_index'])
    
def extract_field(ad, field_name, order):
    return ad[field_name][order]

for field in 'Density Pressure Metallicity'.split() + [f'{k}-velocity' for k in 'xyz'] + [_ for _ in 'xyz']:
    print('Checking %s' % field, end='... ')
    vnew = extract_field(ad_new, field, order['new'])
    vref = extract_field(ad_ref, field, order['ref'])

    np.testing.assert_allclose(vnew, vref)
    print('ok!')

# %%
# Compare gas cells
order = {}

ad_new = ds_new.all_data()
ad_ref = ds_original.all_data()

for ad, prefix in reversed(list(zip((ad_new, ad_ref), ('new', 'ref')))):
    order[prefix] = np.argsort(ad['io', 'particle_identity'])
    
def extract_field(ad, field_name, order):
    return ad[field_name][order]

for field in (('DM', f'particle_{suffix}') for suffix in 'mass position velocity family tag'.split()):
    print('Checking %s, %s' % field, end='... ')
    vnew = extract_field(ad_new, field, order['new'])
    vref = extract_field(ad_ref, field, order['ref'])

    np.testing.assert_allclose(vnew, vref)
    print('ok!')

# %% [markdown]
# ## Debugging the grid

# %%
from collections import defaultdict

def print_tree(ioct, dt):
    header='     ioct  xc                                     cpu_map                    son                                                cpu_neigh            neigh_ind                            '
    print(header)
    print('—'*len(header))
    while ioct >= 1:
        __ = lambda e: e[e>0]
        nbor = dt['nbor'][ioct-1]
        mask = nbor > 0
        neigh_icell, neigh_igrid = (np.full(6, -1, dtype=int) for _ in range(2))
        neigh_icell[mask], neigh_igrid[mask] = np.unravel_index(nbor[mask]-ncoarse, (8, ngridmax))
        cpu_neigh = dt['cpu_map'][neigh_igrid-1, neigh_icell]
        cpu_neigh[~mask] = -1
        def __(x, n):
            return str(x).ljust(n)
        neigh_ind = dt['son'][neigh_igrid-1, neigh_icell]
        neigh_ind[~mask] = -1
        print('', f'{ioct:>8d}', __(dt['xc'][ioct-1], 29), '\t', __(dt['cpu_map'][ioct-1], 26), __(dt['son'][ioct-1], 50), __(cpu_neigh, 20), __(neigh_ind, 20))
        ioct = (dt['parent'][ioct-1] - ncoarse) % ngridmax

DIRECTIONS = [0, 1, 2]
def get_cube(ioct, dt, ret=None, prev_directions=[], depth=0, path=''):
    if ret is None:
        ret = {}
        ret['iocts'] = set((ioct, ))
        ret['path'] = defaultdict(list)
        ret['path'][ioct] = ['']

    for idir in (_ for _ in DIRECTIONS if _ not in prev_directions):
        __ = lambda e: e[e>0]
        neigh_icell, neigh_igrid = np.unravel_index(__(dt['nbor'][ioct-1])-ncoarse, (8, ngridmax))
        iocts_neigh = dt['son'][neigh_igrid-1, neigh_icell][2*idir:2*idir+2]
        for ii, ioct_neigh in enumerate(iocts_neigh):
            if ioct_neigh == 0:
                continue
            new_directions = prev_directions + [idir]
            ret['iocts'].add(ioct_neigh)
            new_path = str(path)
            new_path += '-+'[ii] + 'xyz'[idir]
            ret['path'][ioct_neigh].append(new_path)
            get_cube(ioct_neigh, dt, ret=ret, prev_directions=new_directions, depth=depth+1, path=new_path)

    paths = ret['path']
    iocts = sorted(ret['iocts'], key=lambda k: (len(paths[k][0]), paths[k][0]))
    return iocts, {ioct: paths[ioct] for ioct in iocts}

def match_tree(ioct, dt, new_dt):
    '''Find an oct in the other tree.'''
    # Walk the tree up
    icell_list = []
    while ioct > 1:
        __ = lambda e: e[e>0]
        neigh_icell, neigh_igrid = np.unravel_index(__(dt['nbor'][ioct-1])-ncoarse, (8, ngridmax))
        ioct_parent = (dt['parent'][ioct-1] - ncoarse) % ngridmax
        my_index = np.argwhere(dt['son'][ioct_parent-1] == ioct).flatten()[0]
        icell_list.append(my_index)
        ioct = ioct_parent

    ioct_old = 1
    ioct = 1
    ioct_prev = -1
    for icell in reversed(icell_list):
        ioct_prev = ioct
        ioct = new_dt['son'][ioct-1][icell]
        ioct_old = dt['son'][ioct_old-1][icell]
        print(f'{ioct_old:10d} {ioct:10d}')
    return ioct_prev

ngridmax = dt['headers']['ngridmax']
icell, iparent = np.unravel_index(dt['parent']-ncoarse, (8, ngridmax))
iparent += 1

assert np.all(iparent[1:] > 0)
def get_this_neighbour(i, j, k, dt, ioct):
    ioct0 = ioct

    icell, igrid = np.unravel_index(dt['nbor'][ioct-1, i]-ncoarse, (8, ngridmax))
    ioct = dt['son'][igrid, icell]

    icell, igrid = np.unravel_index(dt['nbor'][ioct-1, j]-ncoarse, (8, ngridmax))
    ioct = dt['son'][igrid, icell]

    icell, igrid = np.unravel_index(dt['nbor'][ioct-1, k]-ncoarse, (8, ngridmax))
    ioct = dt['son'][igrid, icell]
    
    return ioct

ioct = dt['ind_grid'][9:]
for i in range(6):
    for j in range(i+1, 6):
        if (i//2 == j//2): continue
        for k in range(j+1, 6):
            if (k//2 == j//2): continue
            ineigh = get_this_neighbour(i, j, k, dt, iparent)
            break
        break
    break

def check_domain(dt):
    ngridmax = dt['headers']['ngridmax']
    print('\t\tChecking neighbours')
    icell, igrid = np.unravel_index(dt['nbor']-ncoarse, (8, ngridmax))
    icell, igrid

    # Make sure all neighbours exist
    assert np.all(igrid[1:] > 0)
    
    # Make sure all neighbours' neighbour exist
    all_ix = np.array([0, 2, 4, 6])
    all_iy = np.array([0, 1, 4, 5])
    all_iz = np.array([0, 1, 2, 3])

    for ioct in dt['ind_grid'][9:]:
        son_mask = dt['son'][ioct-1]>0
        if son_mask.all():
            cube = get_cube(ioct, dt)[0]
            if len(cube) != 27:
                raise Exception(f'{ioct} is missing {27-len(cube)} neighbours')

def check_tree(data):
    # Check that each grid has its parent neighbour grid
    for icpu, dt in data.items():
        print(f'\tChecking domain {icpu}')
        check_domain(dt)


# %%
print('Original')
check_tree(data)
print('Rewritten')
check_tree(new_data)

# %%
import sys; sys.exit(0)

# %%
new_data[2]['nbor'].min()

# %%
new_data[2]['nbor'][2405]

# %%
new_data[2]['xc'][2405] * bscale

# %%
print_tree(2406, new_data[2])

# %%
igrid = 51
ndt = new_data[5]

ndt['nbor'][igrid-1]

# %%
ndt['xc'][igrid-1]


# %%
def inspect(dt, imin=0, imax=8, reorder=True):
    sl = slice(imin, imax)
    bb = 2**17
    i,j,k=(dt['xc'][sl]*bb).T
    ijk=i+2*j+4*k
    if reorder:
        order = np.argsort(ijk)
    else:
        order = np.arange(len(ijk))
    print('xc')
    print(dt['xc'][sl][order])
    print('')
    print('son')
    print(dt['son'][sl][order])

    print('')
    print('nbor')
    print(dt['nbor'][sl][order])


# %%
inspect(new_data[1])

# %%
ii = 0

# %%
ii += 1

b, d = None, None
for dt in (data[ii], new_data[ii]):
    sl = slice(0, 65)
    order = np.argsort(dt['xc'][sl, 0]*2**20 + dt['xc'][sl, 1]*2**10 + dt['xc'][sl, 2])
    a = dt['parent'][sl][order]
    b = a.copy() if b is None else b

    c = dt['ind_grid'][sl][order]
    d = c.copy() if d is None else d

def order(xc):
    i, j, k = (xc.T*2**20).astype(np.uint64)
    return np.argsort(i<<40 + j<<20 + k)

# Find elements in new mapping that aren't in the old one

ixc_orig = np.round(data[1]['xc']*2**17).astype(np.uint64)
ixc_new = np.round(new_data[1]['xc']*2**17).astype(np.uint64)

_ijk = lambda i, j, k: ((i<<20) + j<<20) + k
ijk_orig = set(_ijk(*ixc_orig.T))
ijk_new = set(_ijk(*ixc_new.T))

ijk_diff = np.asarray(list(ijk_orig - ijk_new))
if ijk_diff.size == 0:
    print(ii, 'YOLO! FOUND ALL OCTS')
else:
    i = ijk_diff >> 40
    j = (ijk_diff >> 20) & ((1<<21) - 1)
    k = (ijk_diff) & ((1<<21) - 1)

    ixc_diff = np.stack((i, j, k), axis=-1)

# %%
data[1]['ind_grid'][np.argwhere(np.all(ixc_diff[-1] == ixc_orig, axis=1)).flatten()]

# %%
ioct_in_new_data = match_tree(28233, data[1], new_data[1])

# %%
iocts, path = get_cube(24994, data[1])
for ioct in iocts:
    print(ioct, path[ioct])
    print_tree(ioct, data[1])

# %%
iocts, path = get_cube(ioct_in_new_data, new_data[1])
for ioct in iocts:
    print(ioct, path[ioct])
    print_tree(ioct, new_data[1])

# %%
for _ in (data[1], new_data[1]):
    inspect(_, imin=50, imax=60, reorder=True)
