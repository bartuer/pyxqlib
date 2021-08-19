# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from typing import List, Tuple, Union, Callable, Iterable, Dict
import pandas as pd
import numpy as np
import json,codecs
from pyxqlib import Tsidx

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
    return pd.DafaFrame.from_records(n.reshape(-1,1))

def d2j(a, name):
    json.dump(a.tolist(), codecs.open(name, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4) 
    
class MustelasQuote(BaseQuote):
    def __init__(self, df_file: str):
        quote_df = pd.read_pickle(df_file)
        super().__init__(quote_df=quote_df)
        self.d = {}             # cache
        self.s = {}             # name
        self.i = Tsidx()        # index
        self.l = {}
        j = 0
        for s, v in quote_df.groupby(level="instrument"):
            self.s[s] = j
            d = v.droplevel(level="instrument")
            if not hasattr(self, 'c'):
                self.c = dict((c,i) for i, c in enumerate(d.columns))
            self.i[j] = t2i(d.index.view())
            self.d[s] = np.array([d2n(d,f) for f in self.c.keys()])
            self.l['b'] = np.asarray(np.where(self.d[s][self.c['limit_buy']]==True)[0], dtype=np.int32)
            self.l['s'] = np.asarray(np.where(self.d[s][self.c['limit_sell']]==True)[0],dtype=np.int32)
            j += 1
        
    def get_all_stock(self):
        return self.s.keys()

    def idx(self, i, b, e, f):
        j, k = t2i([b,e])
        return self.d[i][self.c[f]][self.i[self.s[i]][j:k]]
            
    def end(self, i, e, f):
        k = t2i([e])[0]
        return self.d[i][self.c[f]][self.i[self.s[i]][:k]]

    def get_data(self, stock_id, start_time, end_time, field, method):
        if (method == 'last'):  # number
            return self.end(stock_id, end_time, field)
        elif (method == 'sum'): # number
            return self.idx(stock_id, start_time, end_time, field).sum()
        elif (method == 'mean'): # number
            return np.mean(self.idx(stock_id, start_time, end_time, field))
        elif (method == 'all'): # bool
             if field == 'limit_sell':
                 return len(self.l['s']) == 0
             if field == 'limit_buy':
                 return len(self.l['b']) == 0
             return len(self.l['s']) == 0 or len(self.l['b']) == 0
        elif (method == "None"): # series
            return n2d(self.idx(stock_id, start_time, end_time, field))
        elif (method == 'any'): # 
            raise NotImplementedError("not implement")
        else:
            return n2d(self.idx(stock_id, start_time, end_time, field))
        return 


q = MustelasQuote("data/quote_df.pkl")
print(q.get_data('SH600004','2020-05-30','2020-06-12','$volume','all'))
print(q.get_data('SH600004','2020-05-30','2020-06-12','$volume','sum'))
print(q.get_data('SH600004','2020-05-30','2020-06-12','$volume','mean'))