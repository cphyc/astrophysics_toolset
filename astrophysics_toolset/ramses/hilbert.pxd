cimport cython

cimport numpy as np

cdef np.int64_t interleave_3bits_64(np.int64_t x) nogil
cdef np.uint64_t hilbert3d_single(const np.int64_t X[3], const int bit_length) nogil
cdef np.uint64_t[:] hilbert3d_int(np.int64_t[:, ::1] X, int bit_length)
