from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from libc.stddef cimport size_t
from libcpp.pair cimport pair
cdef extern from "<tsidx.h>" namespace "qlibc" nogil:

    cdef cppclass TSIdx:
        TSIdx()

        int build(uint32_t *ts, size_t len) except +
        pair[uint32_t, uint32_t] index(uint32_t start, uint32_t stop) except +
        uint32_t stop(uint32_t stop) except +