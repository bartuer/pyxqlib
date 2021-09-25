# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import cProfile
from threadpoolctl import threadpool_limits
from typing import List, Tuple, Union, Callable, Iterable, Dict
import pandas as pd
import numpy as np
import json,codecs
from pyxqlib import Tsidx
from pandas.testing import assert_index_equal
import ipdb
from datetime import datetime
import ctypes
import numexpr as ne

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 16)

openblas_lib = ctypes.cdll.LoadLibrary('/home/bazhou/local/src/pyxqlib/py37env/lib/python3.7/site-packages/numpy.libs/libopenblasp-r0-2d23e62b.3.17.so')

class BaseQuote:

    def __init__(self, quote_df: pd.DataFrame):
        pass

    def get_all_stock(self) -> Iterable:
        raise NotImplementedError(f"Please implement the `get_all_stock` method")

    def get_data(self, stock_id: str, start_time: Union[pd.Timestamp, str], end_time: Union[pd.Timestamp, str], field: str, method: str,) -> Union[None, float, pd.Series]:
        raise NotImplementedError(f"Please implement the `get_data` method")

def drop_volume_data_absent_in_quote(volume, days):
    vidx = volume.columns.values.astype('datetime64[s]').astype('uint32')
    _index = dict((d,i) for i,d in enumerate(vidx))                      # loops : days
    mask = np.ones(vidx.size, dtype=bool)
    if (vidx.shape != days.shape):
        mask[[_index[i] for i in np.setdiff1d(vidx, days)]] = False      # loops : 1 or less than days
    return mask
    
class MustelasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame):
        quote_df = pd.read_pickle(quote_df)
        super().__init__(quote_df=quote_df)

        self.i = Tsidx()        # ts index
        self.d = {}             # cache in numpy
        self.p = {}             # cache in pandas
        self.n = {}             # name -> id
        self.b = {}             # buy limit
        self.s = {}             # sell limit
        self._n = {}            # id -> name
        self._s = []            # sell limit list

        j = 0
        # for quote query (READONLY)
        for s, v in quote_df.groupby(level="instrument"):             # loops : stocks
            self.n[s] = j                                             # stock name index
            self._n[j] = s                                            # stock name reverse index
            d = v.droplevel(level="instrument")
            if not hasattr(self, 'c'):
                self.c = dict((c,i) for i, c in enumerate(d.columns)) # column(feature) name, loops : columns
                self.midx = d.index.values.astype('datetime64[s]')    # minutes time index ts(min)
            self.i[j] = self.midx.astype('uint32')                    # ts index build 
            self.d[s] = d.values.T                                    # feature in numpy
            self.p[s] = [d[[f]] for f in self.c.keys()]               # feature in pandas (not for map.reduce ), loops : columns
            self.b[s] = np.asarray(np.where(self.d[s][self.c['limit_buy']]==True)[0], dtype=np.int32) # buy limitation
            self.s[s] = np.asarray(np.where(self.d[s][self.c['limit_sell']]==True)[0],dtype=np.int32) # sell limitation
            self._s.append(self.s[s])
            j += 1

        # for indicators map.reduce (WRITE self.m)
        self.indicators = ["ffr", "pa", "pos", "deal_amount", "value", "count"]
        self.keys = self.n.keys()                                     # cache get_all_stock
        self.ii = self.i[0]                                           # shared time series index (CRC cache)
        self.days = self.ii.days                                      # valid trading days ts(day)
        self.drange = self.ii.drange                                  # valid trading days pos index
        self.dcount = self.ii.dcount                                  # valid trading days interval
        self.mod = np.max(np.unique(self.dcount)).astype(np.int16)    # trading minutes of day
        self.mask = np.ones(self.ii.dlen * self.mod, dtype=bool)
        for i in np.where(self.dcount != self.mod)[0]:                # loops : 1 or less than days
            b = self.drange[i] + self.dcount[i]
            e = self.drange[i] + self.mod
            self.mask[b:e] = False
        self.didx = self.days.astype('datetime64[s]').astype('datetime64[D]')
        pick = [self.c[f] for f in ['$factor', '$close']]             # picks for map.reduce, loops : 2
        self.q = np.stack([v.astype(float)
                           for v in self.d.values()])[:, pick, :]    # loops : columns
        self.m = np.zeros((len(self.indicators), self.q.shape[0], self.q.shape[2]), dtype=np.float32)
        self.shape = tuple([self.q.shape[0], self.q.shape[2]])
        self.ushape = tuple([self.shape[0], 1])
        self.m[5] += 1
        self.m[0] /= self.shape[0]
        self.unit = np.zeros(self.shape, dtype=np.float32)
        self.unum = np.zeros(self.shape, dtype=np.int16)
        self.utrade = np.zeros(self.shape, dtype=np.int16)
        # (1 -> stock, dcount -> min)
        self.ucount = np.repeat(np.tile(self.dcount.astype(np.int16), self.ushape), self.dcount, axis=1) 
        self.useq = np.tile(np.repeat(np.arange(self.mod, dtype=np.int16), self.dcount.size)[self.mask], self.ushape)   # (1 -> stock, (day * mod) [mask] -> min)
        self.c_s = self.ucount - self.useq
        self.amount = np.zeros(self.shape, dtype=np.int16)

    @profile
    def map(self, config):
        shape = self.shape
        ushape = self.ushape
        stocks = self.shape[0]

        # Price Chain
        price = self.q[:,1,:]                # (stock, min)
        
        # Order Unit
        factor = self.q[:,0,:]               # (stock, min)
        np.true_divide(config['trade_unit'], factor, dtype=np.float32, out=self.unit)
        limit = False
        for i in range(shape[0]):            # loops : stocks 
            if (len(self._s[i]) > 0):
                limit = True
                self.unit[i, self._s[i]] = np.NaN

        # Amount Chain, simulate 2 segment linear trading amount split
        vidx = config['volume'].columns.values.astype('datetime64[s]').astype('datetime64[D]')
        _index = dict((d,i) for i,d in enumerate(vidx))                          # loops : days
        drop = np.ones(vidx.size, dtype=bool)
        if (vidx.shape != self.didx.shape):
            drop[[_index[i] for i in np.setdiff1d(vidx, self.didx)]] = False     # loops : 1 or less than days
        v = config['volume'].values[:,drop] * config['volume_ratio']                                                           # int   (stock, day)
        utotal = np.repeat(np.divide(v, self.unit[:,self.drange], dtype=np.float32).astype(np.int16), self.dcount, axis=1)     # int   (stock, dcount -> min) 
        np.add(np.divide(utotal, self.ucount, dtype=np.float32), 1, casting='unsafe', out=self.unum)                           # int   /((stock, min)) -> (stock, min)
        np.add(np.divide(utotal - self.useq * self.unum, self.c_s, dtype=np.float32), 1, casting='unsafe', out=self.utrade)    # int   L1((stock, min)) -> (stock, min)
        self.amount = np.where(self.utrade == self.unum, self.utrade, self.unum - 1)                                           # int   where((stock, min)) -> (stock, min)
        np.multiply(self.amount, self.unit, dtype=np.float32, out=self.m[3])                                                   # float L2((stock, min)) -> (stock, min)
        if (config['round_amount']):
            tu = config['trade_unit']
            self.m[3] = ((self.m[3] * factor) + 0.1) // tu * tu / factor
        np.multiply(price, self.m[3], dtype=np.float32, out=self.m[4])
        if limit:
            self.m[0][np.where(self.m[3] == np.NaN)] = 0

        self.price = price.sum(axis=0, dtype=np.float32)       # REALLY UGLY PA CALCULATION BUG
        self.dir = config['order_dir']
        return self

    @profile
    def reduce(self):
        m = self.m.sum(axis=1, dtype=np.float32)                      # aggregate by stock
        d = np.add.reduceat(m, self.drange, axis=1, dtype=np.float32) # aggregate one day

        base_price = np.add.reduceat(self.price / self.m.shape[0], self.drange, dtype=np.float32) / self.dcount
        d[1] = ((d[4] / d[3]) / base_price - 1) * self.dir
        d[2] = (d[1] > 0).astype(float) 
        d[[0,5]] /= self.dcount                     # avgerage on ffr, count

        return {
            "1day"    : pd.DataFrame(d.T, columns=self.indicators, index=pd.Index(self.didx)),
            "1minute" : pd.DataFrame(m.T, columns=self.indicators, index=pd.Index(self.midx)),
        }

    def get_all_stock(self):
        return self.keys

    def idx(self, i, b, e, f, numpy=False):
        j = np.datetime64(b).astype('datetime64[s]').astype('uint32')
        k = np.datetime64(e).astype('datetime64[s]').astype('uint32')
        s = self.i[self.n[i]][j:k]
        if (s.stop - s.start <= 0):
            return None
        if numpy:
            return self.d[i][self.c[f]][s]
        else:
            return self.p[i][self.c[f]].iloc[s]
            
    def end(self, i, e, f):
        k = np.datetime64(e).astype('datetime64[s]').astype('uint32')
        return self.d[i][self.c[f]][self.i[self.n[i]][:k]]
    
    def beg(self, i, b, f):
        j = np.datetime64(b).astype('datetime64[s]').astype('uint32')
        return self.d[i][self.c[f]][self.i[self.n[i]][j:]]

    def get_data(self, stock_id, start_time, end_time, fields=None, method=None):
        if (callable(method)):     # 49%, number
            return self.beg(stock_id, start_time, fields)
        elif (method == 'all'):    # 38%, bool
            if (fields == 'limit_sell'):
                 return self.s[stock_id].size != 0
            if (fields == 'limit_buy'):
                 return self.b[stock_id].size != 0
            return self.s[stock_id].size != 0 or self.b[stock_id].size != 0
        elif (fields is None and method is None):
            return 0
        elif (method is None):     # 13%, pd series
             # return n2d(self.idx(stock_id, start_time, end_time, fields))
             return self.idx(stock_id, start_time, end_time, fields)
        elif (method == 'last'):   # 0%, number
            return self.end(stock_id, end_time, fields)
        elif (method == 'sum'):    # 0%, number
            return self.idx(stock_id, start_time, end_time, fields, True).sum()
        elif (method == 'mean'):   # 0%, number
            return np.mean(self.idx(stock_id, start_time, end_time, fields, True))
        elif (method == 'any'):    # exception
            raise NotImplementedError("not implement")
        else:
            raise ValueError("method should be one of: last, sum, mean, all, None")
        return 

    def dlen(self, stock_id):
        return self.i[self.n[stock_id]].dlen

if True:                        # Calculation API test of MustelasQuote
    data_conf = "df"
    strategy = {
        'volume':pd.read_pickle(f"data/volume_{data_conf}.pkl"),
        'sample_ratio':1.0,     # shuffle/selection stock
        'volume_ratio':0.01,    # initial order amount
        'open_cost':0.0015,     # unused trade_cost not include in final report
        'close_cost':0.0025,    # unused ...
        'trade_unit':100,       # PRC market
        'limit_threshold':0.099,# unused
        'order_dir':-1,         # BUY, from RandomOrderStrategy
        'volume_threshold':None,# NOT _get_amount_by_volume
        'agg':'twap',           # base_price ignore volume (pa_config)
        'round_amount':False,   # round_amount_by_trade_unit,  What The Round ?
        'start_time':'',        # already in calendar
        'end_time':''           # ...
    }
    pr = cProfile.Profile()
    pr.enable()
    with threadpool_limits(limits=1, user_api='blas'):
         print(openblas_lib.openblas_get_num_threads())
         res = MustelasQuote(f"data/quote_{data_conf}.pkl").map(strategy).reduce()
    pr.disable()
    pr.dump_stats(f"data/{datetime.now().strftime('%y-%m-%d_%H:%M')}.prof")
    print(res);
else:                           # Query API test of MustelasQuote
    q = MustelasQuote(f"data/quote_df.pkl")
    print(f"\
     sum:{q.get_data('SH600004','2020-05-30','2020-06-12','$volume','sum')}\n\
     mean:{q.get_data('SH600004','2020-05-30','2020-06-12','$volume','mean')}\n\
     {q.get_data('SH600004','2020-06-11','2020-06-12','$volume')}\n\
     {q.get_data('SH600004','2020-06-11','2020-06-12','$close')}\n\
     {q.get_data('SH600000','2020-01-02 09:31:00', '2020-01-02 09:31:59', '$close')}\n\
     {q.dlen('SH600000')}\n\
     {q.days}\n\
     {q.drange}\n\
     ")

