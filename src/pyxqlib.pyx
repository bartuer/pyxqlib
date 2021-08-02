# cython: profile=False, embedsignature=True, language_level=3

from libc.stdint cimport uint32_t
from libc.stdint cimport int32_t
from libcpp.vector cimport vector
from libc.string cimport memset
from libc.stdio cimport printf
import zlib
import numpy as np
import json
import sys

cimport tsidx

from cython import boundscheck, wraparound

cdef class _Tsidx:
    cdef tsidx.TSIdx* _idx
    cdef int _id
    cdef int _check_sum
    cdef uint32_t _start
    cdef uint32_t _end
    cdef uint32_t _size
    cdef list _check_sums_
    
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

    def __len__(self):
        return self._size

    def __hash__(self):
        return self._check_sum
    
    def __init__(self, list check_sum_list):
        if self._idx:
            return
        self._idx = new tsidx.TSIdx()
        self._check_sums_ = check_sum_list

    def __getitem__(self, x):
        if x.start is None:
            stop = min(self.stop, x.stop)
            return self._idx.stop(stop)
        else:
            start = max(self.start,  x.start)
            stop = min(self.stop, x.stop)
            start_, stop_ = self._idx.index(start, stop)
            return slice(start_, stop_)
        
    @boundscheck(False)
    @wraparound(False)
    cdef load(self, uint32_t[::1] ts):
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
                return self.check_sum

    def __dealloc__(self):
        if self._idx:
            del self._idx

cdef class Tsidx:
    cdef list idxes
    cdef list check_sum_list
    cdef dict ids
    cdef dict sums
    
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