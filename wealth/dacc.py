"""Data access"""

from py2store import FilesOfZip
from io import BytesIO

import pandas as pd
import numpy as np
from dol import StrTupleDict, wrap_kvs, cached_keys

from wealth.util import data_dir

DFLT_QUARTERLY_DATA_SRC = data_dir / 'csv_derived.zip'


def get_data(b: bytes) -> pd.DataFrame:
    return (
        pd.read_csv(BytesIO(b), index_col=0)
        .sort_index()
        .replace(np.nan, 0)  # TODO: Handle nans differently
        .drop(['period', 'Report Date'], axis=1)
    )


t = StrTupleDict('csv_derived/{year}-{quarter}.csv', process_info_dict={'year': int})


@cached_keys(keys_cache=sorted)
@wrap_kvs(
    key_of_id=t.str_to_tuple, id_of_key=t.tuple_to_str, obj_of_data=get_data,
)
class QuarterlyData(FilesOfZip):
    """Reads dataframes of quarterly data

    >>> from wealth.dacc import QuarterlyData
    >>> import pandas as pd
    >>> data = QuarterlyData()
    >>> len(data)
    44

    Keys are (year, quarter) pairs:

    >>> list(data)[:3]
    [(2010, 'Q1'), (2010, 'Q2'), (2010, 'Q3')]

    Values are pandas.DataFrames whose indices are tickers (or groups, or whatever,
    but rows are the items under study) and whose columns represent their features.

    >>> d = data[2010, 'Q1']
    >>> assert isinstance(d, pd.DataFrame)
    >>> list(d.index)[:4]
    ['AAP', 'AAPL', 'ABT', 'ACIW']
    >>> list(d.columns)[:4]
    ['EBITDA', 'Total Debt', 'Free Cash Flow', 'Gross Profit Margin']
    """

    def __init__(self, zip_file=DFLT_QUARTERLY_DATA_SRC, prefix='', open_kws=None):
        super().__init__(zip_file, prefix, open_kws)
