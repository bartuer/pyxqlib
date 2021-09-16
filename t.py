# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from typing import List, Tuple, Union, Callable, Iterable, Dict
import pandas as pd
import numpy as np
import json,codecs
from pyxqlib import Tsidx
from pandas.testing import assert_index_equal
import ipdb

class BaseQuote:

    def __init__(self, quote_df: pd.DataFrame):
        pass

    def get_all_stock(self) -> Iterable:
        raise NotImplementedError(f"Please implement the `get_all_stock` method")

    def get_data(self, stock_id: str, start_time: Union[pd.Timestamp, str], end_time: Union[pd.Timestamp, str], field: str, method: str,) -> Union[None, float, pd.Series]:
        raise NotImplementedError(f"Please implement the `get_data` method")

def ag(volume, amount, ucount, seq, to, unit):
    while (np.all(seq < to)): # loop self.dcount
        yield amount
        i = int(np.unique(seq))
        ucount = volume // unit[:,:,i]
        amount = (ucount + to - seq - 1) // (to - seq) * unit[:,:,i]
        seq += 1
        volume -= amount

class MustelasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame):
        quote_df = pd.read_pickle(quote_df)
        super().__init__(quote_df=quote_df)

        self.i = Tsidx()        # index
        self.d = {}             # cache in numpy
        self.p = {}             # cache in pandas
        self.n = {}             # name
        self._n = {}            # id -> name
        self.b = {}             # buy limit
        self.s = {}             # sell limit
        j = 0

        # for quote query (READ ONLY)
        for s, v in quote_df.groupby(level="instrument"):
            self.n[s] = j       # stock name
            self._n[j] = s
            d = v.droplevel(level="instrument")
            if not hasattr(self, 'c'):
                self.c = dict((c,i) for i, c in enumerate(d.columns)) # column(feature) name
                self.midx = d.index.values.astype('datetime64[s]') # minutes time index
            self.i[j] = d.index.values.astype('datetime64[s]').astype('uint32') # time index
            self.d[s] = d.values.T # feature in numpy
            self.p[s] = [d[[f]] for f in self.c.keys()] # feature in pandas
            self.b[s] = np.asarray(np.where(self.d[s][self.c['limit_buy']]==True)[0], dtype=np.int32) # buy limitation
            self.s[s] = np.asarray(np.where(self.d[s][self.c['limit_sell']]==True)[0],dtype=np.int32) # sell limitation
            j += 1

        # for indicators caluculation
        self.ii = self.i[0]       # all stock in fact share time series index
        self.keys = self.n.keys() # stock names
        self.q = np.stack(list(self.d.values()))[:,[self.c['$factor'], self.c['$close']],:] # computational quote data
        self.indicators = ["ffr", "pa", "pos", "deal_amount", "value", "count"]
        # indicator matrix : (stock,indicator,time range)
        self.days = self.ii.days # valid trading days as Unix timestamp
        self.didx = self.days.astype('datetime64[s]').astype('datetime64[D]')
        self.drange = self.ii.drange # valid tradng days index
        self.dcount = np.diff(np.append(self.drange, self.midx.size))
        
    def map(self, config):
        markets = []
        factor = self.q[:,0,:].astype(float)
        unit = config['trade_unit'] / factor
        for i in range(self.q.shape[0]): # tradable
            unit[i, self.s[self._n[i]]] = np.NaN
        price = self.q[:,1,:].astype(float)
        base_price = np.add.reduceat(price, self.drange, axis=1) / self.dcount
        pa = price / np.repeat(base_price, self.dcount, axis=1) - 1
        pos = (pa > 0).astype(float)
        v = config['volume'].values.T * config['volume_ratio']
        sz = v.shape                     # (stock, day)
        # bz = int(np.unique(self.dcount)) # batch size, raise if diff dcount
        bz = 240
        s = np.zeros(sz, dtype=int)      # seq
        a = v / bz                       # amount
        c = np.zeros(sz, dtype=int)      # count
        t = np.zeros(sz, dtype=int) + bz # to
        u = unit.reshape(sz + (bz,))     # unit
        amount = np.stack([i for i in ag(v, a, c, s, t, u)])
        if np.all(amount[-1,:,:] < v):   # last one is min(a, v)
            raise ValueError('trade amount error')
        shape = (sz[0], sz[1] * bz)
        deal_amount = amount.T.reshape(shape)
        # _deal_amount = ((deal_amount * factor) + 0.1) // trade_unit * trade_unit / factor  # round doing?
        value = price * deal_amount
        ffr = np.ones(shape, dtype=float)
        ffr[np.where(deal_amount == np.NaN)] = 0
        count = np.ones(shape)
        indicators = {"ffr":ffr, "pa":pa, "pos":pos,
                      "deal_amount":deal_amount,
                      "value":value, "count":count}
        for s in range(shape[0]):
            markets.append(np.stack([indicators[k][s] for k in self.indicators]))
        self.m = np.stack(markets)
        return self

    def reduce(self):
        m = self.m.sum(axis=0)
        d = np.add.reduceat(m, self.drange, axis=1)
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
            raise ValueError("fields and method should be str, and method should be one of: last, sum, mean, all, None")
        return 

    def check_days(self, stock_id):
        d = pd.read_pickle('data/long_indicator2.pkl')
        pdi = pd.Index(d.index.values)
        res = self.i[self.n[stock_id]].days
        di = pd.Index(res.astype('datetime64[s]').astype('datetime64[D]'))
        pd.options.display.max_seq_items = 120
        # assert_index_equal(di, pdi);
        return res

    def dlen(self, stock_id):
        return self.i[self.n[stock_id]].dlen

data_conf = "df"
q = MustelasQuote(f"data/quote_{data_conf}.pkl")
print(q.map({
        'volume':pd.read_pickle(f"data/volume_{data_conf}.pkl"),
        'sample_ratio':1.0,     # shuffle stock
        'volume_ratio':0.01,    # initial order amount
        'open_cost':0.0015,     # unused
        'close_cost':0.0025,    # unused
        'trade_unit':100,       # / factor
        'limit_threshold':0.099,# unused
        'volume_threshold':None,# remove _get_amount_by_volume
        'start_time':'2020-01-01',
        'end_time':'2020-01-03 16:00'
    }).reduce())
# print(f"\
#  sum:{q.get_data('SH600004','2020-05-30','2020-06-12','$volume','sum')}\n\
#  mean:{q.get_data('SH600004','2020-05-30','2020-06-12','$volume','mean')}\n\
#  {q.get_data('SH600004','2020-06-11','2020-06-12','$volume')}\n\
#  {q.get_data('SH600004','2020-06-11','2020-06-12','$close')}\n\
#  {q.get_data('SH600000','2020-01-02 09:31:00', '2020-01-02 09:31:59', '$close')}\n\
#  {q.dlen('SH600000')}\n\
#  {q.days}\n\
#  {q.drange}\n\
#  ")

