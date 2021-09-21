# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import cProfile
from typing import List, Tuple, Union, Callable, Iterable, Dict
import pandas as pd
import numpy as np
import json,codecs
from pyxqlib import Tsidx
from pandas.testing import assert_index_equal
import ipdb
from datetime import datetime


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 16)

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
            j += 1

        # for indicators map.reduce (WRITE self.m)
        self.indicators = ["ffr", "pa", "pos", "deal_amount", "value", "count"]
        self.keys = self.n.keys()                                     # cache get_all_stock
        self.ii = self.i[0]                                           # shared time series index (CRC cache)
        self.days = self.ii.days                                      # valid trading days ts(day)
        self.drange = self.ii.drange                                  # valid trading days pos index
        self.dcount = self.ii.dcount                                  # valid trading days interval
        self.mod = np.max(np.unique(self.dcount))                     # trading minutes of day
        self.mask = np.ones(self.ii.dlen * self.mod, dtype=bool)
        for i in np.where(self.dcount != self.mod)[0]:                # loops : 1 or less than days
            b = self.drange[i] + self.dcount[i]
            e = self.drange[i] + self.mod
            self.mask[b:e] = False
        self.didx = self.days.astype('datetime64[s]').astype('datetime64[D]')
        pick = [self.c[f] for f in ['$factor', '$close']]             # picks for map.reduce, loops : 2
        self.q = np.stack([v.astype(float) for v in self.d.values()])[:, pick, :] # loops : columns

    def map(self, config):
        stocks = self.q.shape[0]

        # Price Chain
        price = self.q[:,1,:]                # (stock, min)
        shape = price.shape
        
        # Order Unit
        factor = self.q[:,0,:]               # (stock, min)
        unit = config['trade_unit'] / factor
        for i in range(self.q.shape[0]):     # loops : stocks 
            unit[i, self.s[self._n[i]]] = np.NaN

        # Amount Chain, simulate 2 segment linear trading amount split
        drop = drop_volume_data_absent_in_quote(config['volume'], self.days)
        v = config['volume'].values[:,drop] * config['volume_ratio']                              # (stock, day)
        ushape = tuple([shape[0], 1])                                                             # (stock, 1) 
        utotal = np.repeat(v // unit[:,self.drange], self.dcount, axis=1)                         # (stock, dcount -> min) 
        ucount = np.repeat(np.tile(self.dcount, ushape), self.dcount, axis=1)                     # (1 -> stock, dcount -> min)
        unum = np.ceil(utotal / ucount)                                                           # /((stock, min)) -> (stock, min)
        useq = np.tile(np.repeat(np.arange(self.mod), self.dcount.size)[self.mask], ushape)       # (1 -> stock, (day * mod) [mask] -> min)
        utrade = np.ceil((utotal - useq * unum) / (ucount - useq))                                # L1((stock, min)) -> (stock, min)
        deal_amount = np.where(utrade == unum, utrade, unum - 1) * unit                           # L2((stock, min)) -> (stock, min)
        if (config['round_amount']):
            tu = config['trade_unit']
            _deal_amount = ((deal_amount * factor) + 0.1) // tu * tu / factor
        value = price * deal_amount
        ffr = np.ones(shape, dtype=float) / stocks
        ffr[np.where(deal_amount == np.NaN)] = 0
        count = np.ones(shape) 

        # Price chain
        pa = np.zeros(shape, dtype=float)    # MAYBE A BUG (link "r2c.org" 2141) (link "r2c.org" 3265)
        pos = np.zeros(shape, dtype=float)

        # Result Tensor
        locals_ = locals()
        data = dict((k, locals_[k]) for k in self.indicators) # loops : indicators
        self.m = np.stack([data[k] for k in self.indicators]) # loops : indicators
        self.price = price.sum(axis=0)       # REALLY UGLY PA CALCULATION BUG
        self.dir = config['order_dir']
        return self

    def reduce(self):
        m = self.m.sum(axis=1)                      # aggregate by stock
        d = np.add.reduceat(m, self.drange, axis=1) # aggregate one day

        base_price = np.add.reduceat(self.price / self.m.shape[0], self.drange) / self.dcount
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

