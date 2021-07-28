from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
cdef extern from "<order.h>" namespace "qlibc" nogil:

    cdef cppclass Order:
        Order()

