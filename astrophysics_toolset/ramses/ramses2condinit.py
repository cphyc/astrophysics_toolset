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
ds = yt.load('/home/ccc/Documents/Postdoc/genetIC-angular-momentum-constrain/simulations/data/DM_256_planck/resim/00140/halo_189/relative_change_lx_0.8/output_00001/info_00001.txt')
LONGINT, QUADHILBERT = True, False
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

new_ncpu = 4

# %%
dsn = yt.load('output_00080/info_00001.txt')


# %%
def read_amr(amr_file, longint=LONGINT, quadhilbert=QUADHILBERT):
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

def convert_ncpus(dirname : str):
    pattern = os.path.join(dirname, 'amr_?????.out?????')
    amr_files = sorted(glob(pattern))

    data = {}
    for i, amr_file in enumerate(amr_files):
        print(f'Reading {amr_file}')
        icpu = int(amr_file.split('.out')[1])
        data[icpu] = read_amr(amr_file)  # CPU are indexed by one
    return data

path = os.path.split(ds.parameter_filename)[0]

data = convert_ncpus(path)

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
    N2 += oct.add(ipos, file_ind, domain_ind=domain_ind, new_domain_ind=new_domain_ind, owning_cpu=original_domain, hilbert_key=dt['_hilbert_key_grid'], lvl=lvl_ind)

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
    f.write_vector(son[:ncoarse])
    f.write_vector(refmap[:ncoarse])
    f.write_vector(cpu_map[:ncoarse])

    print('Fine levels')
    nlevelmax = headers['nlevelmax']
    ncpu = headers['ncpu']
    nboundary = headers['nboundary']
    if nboundary > 0:
        raise NotImplementedError

    ii = 0 # ncoarse
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
                write_chunk('cpu_map', idim)
            for idim in range(2**3):
                write_chunk('refmap', idim)

            ii += ncache


# %%
new_data = {}
for new_icpu in range(1, new_ncpu+1):
    bk_low = new_bound_keys[new_icpu-1]
    bk_up = new_bound_keys[new_icpu]

    print(f'Selecting grid intersecting with new cpu #{new_icpu}')

    oct.clear_paint()

    # Select cells that intersect with domain
    for icpu, dt in data.items():
        lvl = dt['_level_cell'].astype(np.uint8).flatten()
        hkg = dt['_hilbert_key'].astype(np.uint64)

        ishift = 3*(bit_length-lvl+1)
        order_min = (hkg >> ishift)
        order_max = (order_min + 1) << ishift
        order_min <<= ishift

        mask = (order_max > bk_low) & (order_min < bk_up)

        oct.select(dt['_ixcell'].reshape(-1, 3)[mask],
                  lvl[mask].astype(np.int64))

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
        mask = amr_struct['owning_cpu'] == icpu
        find = file_inds[mask] - 1
        for f_new, f_old in fields:
            data_out[f_new][mask] = dt[f_old][find]

    for k, v in data_out.items():
        amr_struct[k] = v

    # Write AMR file
    base = 'output_00080'
    os.makedirs(base, exist_ok=True)
    amr_file = os.path.join(base, f'amr_00001.out{new_icpu:05d}')

    tmp = amr_struct
    write_amr_file(headers, tmp, amr_file, None, None)

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


# %% [markdown]
# Now write hydro

# %%
# # %%cython -a

# from yt.utilities.cython_fortran_utils cimport FortranFile as FF
# cimport numpy as np
# import numpy as np
# cimport cython

#@cython.wraparound(False)
#@cython.boundscheck(False)
# def read_entire_file(tuple headers_desc, np.float64_t[:, :, ::1] data_out, str fname):
def fluid_file_reader(field_handler, headers={}):
    #cdef FF fin
    #cdef int nvar, nboundaries, nlevelmax, ncpu, ilvl, icpu, ilvl2, ncache, icell, ivar, ii

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
                ilvl2 = fin.read_int()
                ncache = fin.read_int()
                if ncache > 0:
                    for icell in range(8):
                        for ivar in range(nvar):
                            data_out[ivar, icell, ii:ii+ncache] = fin.read_vector('d')
                ii += ncache
    return headers, data_out


def fluid_file_writer(fname, headers, data, numbl):
    with FF(fname, mode='w') as fout:
        for v in headers.values():
            fout.write_vector(np.asarray(v))
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


fluid_filename_mapping = {
    'gravity': 'grav_{iout:05d}.out{icpu:05d}',
    'ramses': 'hydro_{iout:05d}.out{icpu:05d}'
}

def rewrite_fluid_files(amr_structure, domains, base_out='output_00080/'):
    nkind = len(domains[0].field_handlers)
    ncpu_new = len(amr_structure)
    filenames = [fluid_filename_mapping[fh.ftype] for fh in domains[0].field_handlers]

    progress = tqdm(total=nkind)
    for i in range(nkind):
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

        progress.set_description(f'{ftype}: W')
        for icpu in amr_structure.keys():
            fname = os.path.join(base_out, filenames[i].format(iout=1, icpu=icpu))
            write_fluid_file(fname, amr_structure[icpu], headers, data_old=data_orig)
        progress.update(1)


rewrite_fluid_files(new_data, ds.index.domains)


# %%
from yt.frontends.ramses.particle_handlers import ParticleFileHandler

particle_filename_mapping = {
    'io': 'particle_{iout:05d}.out{icpu:05d}',
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
        
    with FF(fname, mode='w') as fout:
        h = headers.copy()
        h['npart'] = npart
        # TODO: nstar, nstar_tot, etc.
        
        # Write headers
        for v in h.values():
            fout.write_vector(np.asarray(v))

        # Write fields
        for field in fields:
            vals = np.empty(npart, dtype=data_old[1][field].dtype)

            i0 = 0
            for mask, dt in zip(masks.values(), data_old.values()):
                count = mask.sum()
                vals[i0:i0+count] = dt[field][mask]
                i0 += count

            fout.write_vector(vals)
        

def rewrite_particle_files(amr_structure, domains, base_out='output_00080/'):
    nkind = len(domains[0].particle_handlers)
    ncpu_new = len(amr_structure)
    filenames = [particle_filename_mapping[fh.ptype] for fh in domains[0].particle_handlers]

    progress = tqdm(total=nkind)
    for i in range(nkind):
        ptype = domains[0].particle_handlers[i].ptype
        progress.set_description(f'{ptype}: R')
        data_orig = {}
        particle_new_domain = {}
        headers = {}
        for icpu, dom in enumerate(tqdm(domains, desc='Reading files', leave=False)):
            fh = dom.particle_handlers[i]
            ret = particle_file_reader(fh, {})
            headers, data_orig[icpu+1] = ret
            pos =  np.stack([ret[1]['io', 'particle_position_%s' % k] for k in 'xyz'], axis=-1)
            ipos = np.round(pos * bscale).astype(np.int64)
            particle_new_domain[icpu+1] = np.digitize(hilbert3d(ipos, bit_length), amr_structure[1]['bound_keys'])

        # Need to update the number of cpus in the headers
        headers['ncpu'] = ncpu_new
        
        progress.set_description(f'{ptype}: W')
        for icpu in amr_structure.keys():
            fname = os.path.join(base_out, filenames[i].format(iout=1, icpu=icpu))
            particle_file_writer(fname, particle_new_domain, headers, data_old=data_orig, new_icpu=icpu)
        progress.update(1)
    return particle_new_domain
toto = rewrite_particle_files(new_data, ds.index.domains)

# %%
import sys; sys.exit(0)


# %% [markdown]
# ## Debugging the grid

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
from collections import defaultdict

def print_tree(ioct, dt):
    header='     ioct  xc                                     cpu_map                    son                                                cpu_neigh            neigh_ind                            '
    print(header)
    print('—'*len(header))
    while ioct >= 1:
        neigh_icell, neigh_igrid = np.unravel_index(dt['nbor'][ioct-1]-ncoarse, (8, ngridmax))
        cpu_neigh = dt['cpu_map'][neigh_igrid-1, neigh_icell]
        def __(x, n):
            return str(x).ljust(n)
        print('', f'{ioct:>8d}', __(dt['xc'][ioct-1], 29), '\t', __(dt['cpu_map'][ioct-1], 26), __(dt['son'][ioct-1], 50), __(cpu_neigh, 20), __(dt['son'][neigh_igrid-1, neigh_icell], 20))
        ioct = (dt['parent'][ioct-1] - ncoarse) % ngridmax

DIRECTIONS = [0, 1, 2]
def get_cube(ioct, dt, ret=None, prev_directions=[], depth=0, path=''):
    if ret is None:
        ret = {}
        ret['iocts'] = set((ioct, ))
        ret['path'] = defaultdict(list)
        ret['path'][ioct] = ['']

    for idir in (_ for _ in DIRECTIONS if _ not in prev_directions):
        neigh_icell, neigh_igrid = np.unravel_index(dt['nbor'][ioct-1]-ncoarse, (8, ngridmax))
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
        neigh_icell, neigh_igrid = np.unravel_index(dt['nbor'][ioct-1]-ncoarse, (8, ngridmax))
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
