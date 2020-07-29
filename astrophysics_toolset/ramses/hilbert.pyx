cimport numpy as np

import numpy as np

cimport cython


cdef np.uint64_t[:, :, ::1] state_diagram = np.array([
    1, 2, 3, 2, 4, 5, 3, 5,
    0, 1, 3, 2, 7, 6, 4, 5,
    2, 6, 0, 7, 8, 8, 0, 7,
    0, 7, 1, 6, 3, 4, 2, 5,
    0, 9,10, 9, 1, 1,11,11,
    0, 3, 7, 4, 1, 2, 6, 5,
    6, 0, 6,11, 9, 0, 9, 8,
    2, 3, 1, 0, 5, 4, 6, 7,
    11,11, 0, 7, 5, 9, 0, 7,
    4, 3, 5, 2, 7, 0, 6, 1,
    4, 4, 8, 8, 0, 6,10, 6,
    6, 5, 1, 2, 7, 4, 0, 3,
    5, 7, 5, 3, 1, 1,11,11,
    4, 7, 3, 0, 5, 6, 2, 1,
    6, 1, 6,10, 9, 4, 9,10,
    6, 7, 5, 4, 1, 0, 2, 3,
    10, 3, 1, 1,10, 3, 5, 9,
    2, 5, 3, 4, 1, 6, 0, 7,
    4, 4, 8, 8, 2, 7, 2, 3,
    2, 1, 5, 6, 3, 0, 4, 7,
    7, 2,11, 2, 7, 5, 8, 5,
    4, 5, 7, 6, 3, 2, 0, 1,
    10, 3, 2, 6,10, 3, 4, 4,
    6, 1, 7, 0, 5, 2, 4, 3], dtype=np.uint64).reshape(12, 2, 8).T.copy()

cdef np.int64_t magics[6]
magics[0] = 0x1249249249249249
magics[1] = 0x10c30c30c30c30c3
magics[2] = 0x100f00f00f00f00f
magics[3] = 0x001f0000ff0000ff
magics[4] = 0x001f00000000ffff
magics[5] = 0x1fffff

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef inline np.int64_t interleave_3bits_64(np.int64_t x) nogil:
    # From https://stackoverflow.com/a/18528775/2601223
    x &= magics[5]
    x = (x | x << 32) & magics[4]
    x = (x | x << 16) & magics[3]
    x = (x | x <<  8) & magics[2]
    x = (x | x <<  4) & magics[1]
    x = (x | x <<  2) & magics[0]
    return x

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef np.uint64_t hilbert3d_single(const np.int64_t X[3], const int bit_length) nogil:
    global state_diagram
    cdef np.uint64_t i_bit_mask, i_bit_mask_out, sdigit, hdigit, cstate, nstate
    cdef int i
    i_bit_mask = ( (interleave_3bits_64(X[0]) << 2)
                 | (interleave_3bits_64(X[1]) << 1)
                 | (interleave_3bits_64(X[2])))
    i_bit_mask_out = 0

    # Build Hilbert ordering using state diagram
    cstate = 0
    for i in range(bit_length-1, -1, -1):
        sdigit = (i_bit_mask >> (3*i)) & 0b111
        nstate = state_diagram[sdigit, 0, cstate]
        hdigit = state_diagram[sdigit, 1, cstate]

        # Set the three bits
        i_bit_mask_out |= hdigit << (3*i)
        cstate = nstate
    return i_bit_mask_out

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef np.uint64_t[:] hilbert3d_int(np.int64_t[:, ::1] X, int bit_length):
    global state_diagram
    cdef int npoint, ip, i
    cdef np.uint64_t[::1] order
    cdef np.uint64_t i_bit_mask, i_bit_mask_out, sdigit, hdigit, cstate, nstate
    cdef np.int64_t[:, ::1] X_view = X

    npoint = len(X_view)
    order = np.zeros(npoint, dtype=np.uint64)

    # Convert positions to binary
    for ip in range(npoint):
        order[ip] = hilbert3d_single(&X_view[ip, 0], bit_length)

    return order

def hilbert3d(np.ndarray[np.int64_t, ndim=2] X, int bit_length):
    '''Compute the order using Hilbert indexing.

    Arguments
    ---------
    * X: (N, ndim) int ndarray
      The positions
    * bit_length: integer
      The bit_length for the indexing.
    '''
    return np.asarray(hilbert3d_int(X, bit_length), dtype=np.float64)
