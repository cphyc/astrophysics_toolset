cimport cython
cimport numpy as np


cdef extern from *:
    ctypedef int int128_t "__int128_t"
    ctypedef float float128_t "__float128"

cdef np.int64_t interleave_3bits_64(np.int64_t x) nogil
cdef np.uint64_t hilbert3d_single(const np.int64_t X[3], const int bit_length) nogil
cdef np.uint64_t[:] hilbert3d_int(np.int64_t[:, ::1] X, int bit_length)

cdef int128_t interleave_3bits_128(int128_t x) nogil
cdef int128_t hilbert3d_single_quad(const np.int64_t X[3], const int bit_length) nogil
