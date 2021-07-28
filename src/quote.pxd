from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
cdef extern from "<quote.h>" namespace "qlibc" nogil:

    cdef cppclass Quote:
        Quote()

        float sum(float* value, size_t beg, size_t end) except+