cimport cython

cimport numpy as np
import numpy as np


cdef np.int64_t magics[6]
magics[0] = 0x1249249249249249
magics[1] = 0x10c30c30c30c30c3
magics[2] = 0x100f00f00f00f00f
magics[3] = 0x001f0000ff0000ff
magics[4] = 0x001f00000000ffff
magics[5] = 0x1fffff


@cython.cdivision(True)
cdef inline np.int64_t interleave_3bits_64(np.int64_t x) nogil:
    # From https://stackoverflow.com/a/18528775/2601223
    x &= magics[5]
    x = (x | x << 32) & magics[4]
    x = (x | x << 16) & magics[3]
    x = (x | x <<  8) & magics[2]
    x = (x | x <<  4) & magics[1]
    x = (x | x <<  2) & magics[0]
    return x


cdef np.uint64_t hilbert3d_single(const np.int64_t X[3], const int bit_length)
cdef np.uint64_t[:] hilbert3d_int(np.int64_t[:, ::1] X, int bit_length)
cpdef np.ndarray[np.float64_t, ndim=1] hilbert3d(np.ndarray[np.int64_t, ndim=2] X, int bit_length)
