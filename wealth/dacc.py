"""Data access"""

from py2store import FilesOfZip
from io import BytesIO

import pandas as pd
import numpy as np
from dol import StrTupleDict, wrap_kvs, cached_keys

from wealth.util import data_dir

DFLT_QUARTERLY_DATA_SRC = data_dir / "csv_derived.zip"


def get_data(b: bytes) -> pd.DataFrame:
    return (
        pd.read_csv(BytesIO(b), index_col=0)
        .sort_index()
        .replace(np.nan, 0)  # TODO: Handle nans differently
        .drop(["period", "Report Date"], axis=1)
    )


t = StrTupleDict("csv_derived/{year}-{quarter}.csv", process_info_dict={"year": int})


@cached_keys(keys_cache=sorted)
@wrap_kvs(
    key_of_id=t.str_to_tuple,
    id_of_key=t.tuple_to_str,
    obj_of_data=get_data,
)
class QuarterlyData(FilesOfZip):
    """Reads dataframes of quarterly data"""

    def __init__(self, zip_file=DFLT_QUARTERLY_DATA_SRC, prefix="", open_kws=None):
        super().__init__(zip_file, prefix, open_kws)
