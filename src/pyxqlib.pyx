# cython: profile=False, embedsignature=True, language_level=3

from libc.stdint cimport uint32_t
from libc.stdint cimport int32_t
import zlib
import sys
import numpy as np
from libcpp.vector cimport vector

cimport tsidx

from cython import boundscheck, wraparound

cdef class _Tsidx:
    cdef tsidx.TSIdx* _idx
    cdef int _id
    cdef uint32_t _check_sum
    cdef uint32_t _start
    cdef uint32_t _end
    cdef uint32_t _size
    cdef list _check_sums_
    cdef vector[uint32_t] days_buffer
    cdef vector[uint32_t] range_buffer

    property id:
        def __get__(self):
            return self._id
        def __set__(self, int i):
            self._id = i
            
    property start:
        def __get__(self):
            return self._start

    property stop:
        def __get__(self):
            return self._end

    property days:
        def __get__(self):
            self.days_buffer.resize(self._idx.dlen())
            self.days_buffer = self._idx.didx()
            return np.asarray(<uint32_t[:self.days_buffer.size()]>self.days_buffer.data(), dtype=np.uint32)

    property drange:
        def __get__(self):
            self.range_buffer.resize(self._idx.dlen())
            self.range_buffer = self._idx.drange()
            return np.asarray(<uint32_t[:self.range_buffer.size()]>self.range_buffer.data(), dtype=np.uint32)

    property dlen:
        def __get__(self):
            return self._idx.dlen()

    property dcount:
        def __get__(self):
            return  np.diff(np.append(self.drange, self._size))

    property dmask:
       def __get__(self):
           return np.where(self.dcount != 240)

    def __len__(self):
        return self._size

    def __hash__(self):
        return self._check_sum
    
    def __cinit__(self, list check_sum_list):
        if self._idx:
            return
        self._idx = new tsidx.TSIdx()
        self._check_sums_ = check_sum_list

    def _(self, x):
        if x.start is None:
            return self._idx.dstop(x.stop) - 1
        elif x.stop is None:
            return self._idx.dstart(x.start)
        else:
            start_ = self._idx.dstart(x.start)
            stop_ = self._idx.dstop(x.stop)
            return slice(start_, stop_)
        
    def __getitem__(self, x):
        if x.start is None:
            return self._idx.stop(x.stop) - 1
        elif x.stop is None:
            return self._idx.start(x.start)
        else:
            start_, stop_ = self._idx.index(x.start, x.stop)
            return slice(start_, stop_)
        
    def load(self, uint32_t[::1] ts):
        self._check_sum = zlib.crc32(ts)
        if (self._check_sum in self._check_sums_):
            return 0
        else:
            self._size = ts.shape[0]
            self._start = ts[0]
            self._end = ts[-1]
            res = self._idx.build(&ts[0], ts.shape[0])
            if res:
                return 0
            else:
                return self._check_sum

    def __dealloc__(self):
        if self._idx:
            del self._idx

cdef class Tsidx:
    cdef list idxes
    cdef list check_sum_list
    cdef dict ids
    cdef dict sums

    def __init__(self):
        self.check_sum_list = []
        self.idxes = []
        self.ids = {}
        self.sums = {}
        
    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, uint32_t id):
        return self.idxes[self.ids[id]]
        
    def __setitem__(self, uint32_t id, uint32_t[::1] ts):
        idx = _Tsidx.__new__(_Tsidx, self.check_sum_list)
        res = idx.load(ts)
        if res:
            idx.id = id
            self.ids[id] = len(self.idxes)
            self.sums[res] = len(self.check_sum_list)
            self.idxes.append(idx)
            self.check_sum_list.append(res)
        else:
            idx.id = id
            self.ids[id] = self.sums[hash(idx)]
            del idx