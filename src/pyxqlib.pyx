# cython: profile=False, embedsignature=True, language_level=3

from libc.stdint cimport uint32_t
from libc.stdint cimport uint8_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
import zlib
import sys
import numpy as np
import pandas as pd
from libcpp.vector cimport vector

cimport tsidx
cimport numpy as np

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
            return  np.diff(np.append(self.drange, self._size)).astype(np.int32)

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


cdef class MustelasQuote:
    cdef _Tsidx i
    cdef dict d
    cdef dict c
    cdef list _s
    cdef list indicators 
    cdef tuple shape
    cdef tuple ushape
    cdef np.uint32_t mod
    cdef np.uint32_t[:] midx
    cdef np.uint32_t[:] didx
    cdef np.uint32_t[:] drange
    cdef np.uint32_t[:] dcount
    cdef np.uint8_t[:] mask
    cdef np.float32_t[:,:] unit
    cdef np.int16_t[:,:] unum
    cdef np.int16_t[:,:] useq
    cdef np.int16_t[:,:] ucount
    cdef np.int16_t[:,:] utrade
    cdef np.int16_t[:,:] c_s
    cdef np.int16_t[:,:] amount
    cdef np.float32_t[:,:,:] q
    cdef np.float32_t[:,:,:] m
    cdef np.float32_t[:] price
    cdef int dir

    @boundscheck(False)
    @wraparound(False)
    def __init__(self, df):
        cdef int j = 0
        for s, v in df.groupby(level="instrument"):             # loops : stocks
            d = v.droplevel(level="instrument")
            if (j == 0):
                self.d = dict()
                self._s = list()
                self.c = dict((c,i) for i, c in enumerate(d.columns))
                self.midx = d.index.values.astype('datetime64[s]').astype('uint32')    # minutes time index ts(min)
                self.i = _Tsidx.__new__(_Tsidx, [])
                self.i.load(self.midx)
            self.d[s] = d.values.T                                    # feature in numpy
            self._s.append(np.asarray(np.where(self.d[s][self.c['limit_sell']]==True)[0],dtype=np.int32))
            j += 1

        self.indicators = ["ffr", "pa", "pos", "deal_amount", "value", "count"]
        self.didx = self.i.days.astype('datetime64[s]').astype('datetime64[D]').astype('uint32')
        self.drange = self.i.drange
        self.dcount = self.i.dcount.astype(np.uint32)
        self.mod = np.max(np.unique(self.dcount)).astype(np.uint32)
        self.mask = np.ones(self.i.dlen * self.mod, dtype=np.uint8)
        cdef int i
        for i in np.where(np.not_equal(self.dcount, self.mod))[0]:
            b = self.drange[i] + self.dcount[i]
            e = self.drange[i] + self.mod
            self.mask[b:e] = 0

        pick = [self.c[f] for f in ['$factor', '$close']]             # picks for map.reduce, loops : 2
        self.q = np.stack([v.astype(np.float32)
                           for v in self.d.values()])[:, pick, :]    # loops : columns

        self.m = np.zeros((len(self.indicators), self.q.shape[0], self.q.shape[2]), dtype=np.float32)
        self.shape = tuple([self.q.shape[0], self.q.shape[2]])
        self.ushape = tuple([self.shape[0], 1])
        np.add(np.asarray(self.m[5]), np.ones(self.shape), out=np.asarray(self.m[5]))
        np.true_divide(np.asarray(self.m[0]), self.shape[0], out=np.asarray(self.m[0]))
        self.unit = np.zeros(self.shape, dtype=np.float32)
        self.unum = np.zeros(self.shape, dtype=np.int16)
        self.utrade = np.zeros(self.shape, dtype=np.int16)
        self.ucount = np.repeat(np.tile(np.asarray(self.dcount).astype(np.int16), self.ushape), self.dcount, axis=1) 
        self.useq = np.tile(np.repeat(np.arange(self.mod, dtype=np.int16), self.dcount.size)[np.asarray(self.mask, dtype=bool)], self.ushape)   
        self.c_s = np.subtract(self.ucount, self.useq)
        self.amount = np.zeros(self.shape, dtype=np.int16)

    @boundscheck(False)
    @wraparound(False)
    cdef inline _map(self, config):
        shape = self.shape
        ushape = self.ushape
        stocks = self.shape[0]

        # Price Chain
        price = np.asarray(self.q[:,1,:])               # (stock, min)
        
        # Order Unit
        factor = np.asarray(self.q[:,0,:])              # (stock, min)
        np.true_divide(config['trade_unit'], factor, dtype=np.float32, out=np.asarray(self.unit))
        limit = False
        cdef int i
        for i in range(shape[0]):            # loops : stocks 
            if (len(self._s[i]) > 0):
                limit = True
                self.unit[i, self._s[i]] = np.NaN

        # Amount Chain, simulate 2 segment linear trading amount split
        vidx = config['volume'].columns.values.astype('datetime64[s]').astype('datetime64[D]').astype('uint32')
        _index = dict((d,i) for i,d in enumerate(vidx))                          # loops : days
        drop = np.ones(vidx.size, dtype=bool)
        m = np.asarray(self.m)
        if (vidx.shape != self.didx.shape):
            drop[[_index[i] for i in np.setdiff1d(vidx, self.didx)]] = False     # loops : 1 or less than days
        v = config['volume'].values[:,drop] * config['volume_ratio']                                                # int   (stock, day)
        utotal = np.repeat(np.divide(v, np.asarray(self.unit)[:,np.asarray(self.drange)],
                                     dtype=np.float32).astype(np.int16), self.dcount, axis=1)                       # int   (stock, dcount -> min) 
        np.add(np.divide(utotal, self.ucount, dtype=np.float32), 1, casting='unsafe', out=np.asarray(self.unum))    # int   /((stock, min)) -> (stock, min)
        np.add(np.divide(np.subtract(utotal, np.multiply(self.useq, self.unum)), self.c_s, dtype=np.float32),
               1, casting='unsafe', out=np.asarray(self.utrade))                                                    # int   L1((stock, min)) -> (stock, min)
        self.amount = np.where(np.equal(self.utrade, self.unum), self.utrade, np.subtract(self.unum, 1))            # int   where((stock, min)) -> (stock, min)
        np.multiply(self.amount, self.unit, dtype=np.float32, out=m[3])                                             # float L2((stock, min)) -> (stock, min)
        if (config['round_amount']):
            tu = config['trade_unit']
            m[3] = ((m[3] * factor) + 0.1) // tu * tu / factor
        np.multiply(price, m[3], dtype=np.float32, out=m[4])
        if limit:
            m[0][np.where(np.equal(m[3], np.NaN))] = 0

        self.price = np.sum(price, axis=0, dtype=np.float32)       # REALLY UGLY PA CALCULATION BUG
        self.dir = config['order_dir']
        return self

    @boundscheck(False)
    @wraparound(False)
    cdef inline _reduce(self):
        m = np.sum(self.m, axis=1, dtype=np.float32)                     # aggregate by stock
        d = np.add.reduceat(m, self.drange, axis=1, dtype=np.float32)    # aggregate one day

        base_price = np.true_divide(
            np.add.reduceat(np.true_divide(self.price, self.m.shape[0]), self.drange, dtype=np.float32),
            self.dcount)
        
        d[1] = ((d[4] / d[3]) / base_price - 1) * self.dir
        d[2] = (d[1] > 0).astype(np.float32) 
        d[[0, 5]] = np.true_divide(d[[0,5]],  self.dcount)                 # avgerage on ffr, count

        return (np.asarray(self.didx).astype('datetime64[D]'),
                np.asarray(self.midx).astype('datetime64[s]'),
                d,
                m,
                self.indicators)

    def map(self, config):
        return self._map(config)

    def reduce(self):
        return self._reduce()