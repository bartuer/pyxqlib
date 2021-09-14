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
        """return all stock codes

        Return
        ------
        Iterable
            all stock codes
        """

        raise NotImplementedError(f"Please implement the `get_all_stock` method")

    def get_data(
        self,
        stock_id: str,
        start_time: Union[pd.Timestamp, str],
        end_time: Union[pd.Timestamp, str],
        field: str,
        method: str,
    ) -> Union[None, float, pd.Series]:

        raise NotImplementedError(f"Please implement the `get_data` method")


class PandasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame):
        super().__init__(quote_df=quote_df)
        quote_dict = {}
        for stock_id, stock_val in quote_df.groupby(level="instrument"):
            quote_dict[stock_id] = stock_val.droplevel(level="instrument")
        self.data = quote_dict

    def get_all_stock(self):
        return self.data.keys()

    def get_data(self, stock_id, start_time, end_time, field, method):
        if field is None:
            return resam_ts_data(self.data[stock_id], start_time, end_time, method=method)
        elif isinstance(fields, (str, list)):
            return resam_ts_data(self.data[stock_id][fields], start_time, end_time, method=method)
        else:
            raise ValueError(f"fields must be None, str or list")

def t2i(ts):
    return np.asarray(ts, dtype='datetime64[s]').astype('uint32')

def d2n(df, f):
    return df[[f]].to_numpy(dtype=df[[f]].dtypes[0]).T[0]

def n2d(n):
    return pd.DataFrame.from_records(n.reshape(-1,1))

def d2j(a, name):
    json.dump(a.tolist(), codecs.open(name, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4) 
    
class MustelasQuote(BaseQuote):
    def __init__(self, quote_df: pd.DataFrame):
        quote_df = pd.read_pickle("data/quote_df.pkl")
        super().__init__(quote_df=quote_df)

        self.i = Tsidx()        # index
        self.d = {}             # cache in numpy
        self.p = {}             # cache in pandas
        self.n = {}             # name
        self.b = {}             # buy limit
        self.s = {}             # sell limit
        j = 0

        # for quote query (READ ONLY)
        for s, v in quote_df.groupby(level="instrument"):
            self.n[s] = j       # stock name
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
        self.q = np.stack(list(self.d.values()))[:,1:4,:] # computational quote data
        self.indicators = ["ffr", "pa", "pos", "deal_amount", "volume", "count"]
        # indicator matrix : (stock,indicator,time range)
        self.m = np.zeros((self.q.shape[0], len(self.indicators), self.q.shape[2]), dtype=float)
        self.days = self.ii.days # valid trading days as Unix timestamp
        self.didx = self.days.astype('datetime64[s]').astype('datetime64[D]')
        self.drange = self.ii.drange # valid trdding days index
        
    def map(self, config):
        ipdb.set_trace()

    def reduce(self, config):
        c = len(self.indicators)
        ms = self.midx.size
        ds = self.didx.size
        m = np.zeros((c, ms), dtype=float)
        self.m.sum(axis=0, out=m)
        d = np.zeros((c, ds), dtype=float)
        np.add.reduceat(m, self.drange, axis=1, out=d)
        return {
            "1day"    : pd.DataFrame(d, columns=self.indicators, index=pd.Index(self.didx)),
            "1minute" : pd.DataFrame(m, columns=self.indicators, index=pd.Index(self.midx)),
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
    
q = MustelasQuote("data/quote_df.pkl")
q.map({
        'volume':pd.read_pickle("data/volume_df.pkl"),
        'sample_ratio':1.0,
        'volume_ratio':0.01,
        'start_time':'2020-01-01',
        'end_time':'2020-01-03 16:00'
    })
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

