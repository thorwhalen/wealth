"""Aligned UMAP analysis"""

import os
from collections import Counter
from typing import Callable, Union, Optional, List

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# import matplotlib.pyplot as plt
#
# from celluloid import Camera
# from IPython.display import HTML

import umap

# import umap.utils as utils
import umap.aligned_umap

from wealth.dacc import QuarterlyData, DFLT_QUARTERLY_DATA_SRC

RawData = List[List[pd.DataFrame]]
RawDataGetter = Callable[[], RawData]
Data = List[pd.DataFrame]
Values = List[np.ndarray]

# DFLT_ROOTDIR = "./csv_derived"


# def get_raw_data(rootdir: str = DFLT_ROOTDIR) -> RawData:
#     return [
#         [
#             pd.read_csv(
#                 os.path.join(rootdir, f"{year}-Q{q}.csv"), index_col=0
#             ).sort_index()
#             for q in range(1, 5)
#         ]
#         for year in range(2010, 2021)
#     ]
#
#
# def raw_data_to_all_data(raw_data: Union[RawData, RawDataGetter] = get_raw_data) -> Data:
#     if isinstance(raw_data, Callable):
#         raw_data = raw_data()
#     dataFlat = [item for sublist in readData for item in sublist]
#     all_data = [d.copy() for d in dataFlat]
#     # for d in data: d = d.replace(np.nan, 0)
#     for idx, d in enumerate(all_data):
#         all_data[idx] = d.replace(np.nan, 0).drop(["period", "Report Date"], axis=1)
#     return all_data
#
#
def get_data(zip_file=DFLT_QUARTERLY_DATA_SRC) -> Data:
    return list(QuarterlyData(zip_file).values())


def relations_for_dfs(from_df, to_df):
    left = pd.DataFrame(data=np.arange(len(from_df)), index=from_df.index)
    right = pd.DataFrame(data=np.arange(len(to_df)), index=to_df.index)
    merge = pd.merge(left, right, left_index=True, right_index=True)
    return dict(merge.values)


def data_2_relations(data: Data):
    return [relations_for_dfs(x, y) for x, y in zip(data[:-1], data[1:])]


def assert_relations(data: Data, relations: dict):
    for data_idx, relation in enumerate(relations):
        for i, j in relation.items():
            assert data[data_idx].iloc[i].name, data[data_idx + 1].iloc[j].name


def assert_all_data_dfs_have_same_columns(data: Data):
    first_data_columns = data[0].columns
    for i, d in enumerate(data):
        assert all(d.columns == first_data_columns), (
            f"data[{i}] has different columns than " f"data[0]"
        )


def ticker_counter(data):
    c = Counter()
    for d in data:
        c.update(d.index.values)
    return pd.Series(c).sort_values(ascending=False)


def get_all_tickers(data):
    return sorted(ticker_counter(data))


def weird_tickers(tickers):
    def not_normal_ticker(t):
        return not isinstance(t, str) or len(t) > 4 or t != t.upper()

    return sorted(filter(not_normal_ticker, tickers))


def sample_tickers(
    data: Data,
    n_tickers: int = 50,
    ticker_choice_col: str = "EBITDA",
    sort_ascending: bool = False,
    only_tickers_present_everywhere: bool = True,
):
    c = ticker_counter(data)
    df = pd.concat(data).groupby("Ticker").mean()
    if only_tickers_present_everywhere:
        ticker_choices = c[c >= len(data)].index.values
        df = df.loc[ticker_choices]

    tickers = (
        df.sort_values(ticker_choice_col, ascending=sort_ascending)
        .iloc[:n_tickers]
        .index.values
    )
    return tickers


def _available_tickers(d, tickers):
    return [ticker for ticker in tickers if ticker in d.index]


def sample_data(
    data: Data,
    tickers: Union[int, List[str]] = 50,
    ticker_choice_col: str = "EBITDA",
    sort_ascending: bool = False,
    only_tickers_present_everywhere: bool = True,
):
    if isinstance(tickers, int):
        tickers = sample_tickers(
            data,
            tickers,
            ticker_choice_col,
            sort_ascending,
            only_tickers_present_everywhere,
        )
    return [d.loc[_available_tickers(d, tickers)] for d in data]


std_normalizer = StandardScaler()
pca_normalizer = Pipeline(
    steps=[("zscore", StandardScaler()), ("decomposer", PCA(n_components=17))]
)


def get_values(data: Data, normalizer=None):
    values = [d.values for d in data]
    if normalizer:
        normalizer.fit(np.vstack(values))
        values = list(map(normalizer.transform, values))
    return values


def get_values_and_relations(data: Data, normalizer=None):
    return get_values(data, normalizer), data_2_relations(data)


DFLT_ALIGNED_UMAP_KWARGS = dict(
    n_neighbors=5,  # default: 15
    n_components=2,  # 2
    metric="euclidean",  # 'euclidean'
    metric_kwds=None,  # None
    n_epochs=None,  # None
    learning_rate=1.0,  # 1.0
    init="spectral",  # 'spectral'
    alignment_regularisation=0.001,  # 0.01
    alignment_window_size=2,  # 3
    min_dist=0.1,  # 0.1
    spread=1.0,  # 1.0
    low_memory=False,  # False
    set_op_mix_ratio=1.0,  # 1.0
    local_connectivity=1.0,  # 1.0
    repulsion_strength=1.0,  # 1.0
    negative_sample_rate=5,  # 5
    transform_queue_size=4.0,  # 4.0
    a=None,  # None
    b=None,  # None
    random_state=None,  # None
    angular_rp_forest=False,  # False
    target_n_neighbors=-1,  # -1
    target_metric="categorical",  # 'categorical'
    target_metric_kwds=None,  # None
    target_weight=0.5,  # 0.5
    transform_seed=42,  # 42
    force_approximation_algorithm=False,  # False
    verbose=False,  # False
    unique=False,  # False
)


def get_embeddings(
    reducer,
    values,
    relations=None,
):
    if isinstance(reducer, dict):
        reducer_kwargs = reducer
        kwargs = dict(DFLT_ALIGNED_UMAP_KWARGS, **reducer_kwargs)
        reducer = umap.aligned_umap.AlignedUMAP(**kwargs)

    if relations is None:
        data = values
        values, relations = get_values_and_relations(data)
    fitted = reducer.fit(values, relations=relations)
    return fitted.embeddings_


from functools import partial
from sklearn.cluster import KMeans

from lined import Pipe

embedding_movement = Pipe(partial(np.diff, axis=0), np.abs, np.mean)
embedding_movement.__name__ = "embedding_movement"


def embedding_clusterness(embeddings, n_clusters=11):
    embeddings = np.array(embeddings)
    if embeddings.ndim == 3:
        return np.mean([embedding_clusterness(x, n_clusters) for x in embeddings])
    else:  # single embedding
        model = KMeans(n_clusters=11).fit(embeddings).inertia_
        return model


def embedding_stats(embeddings):
    funcs = [embedding_clusterness, embedding_movement]
    return {func.__name__: func(embeddings) for func in funcs}
