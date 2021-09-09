"""Aligned UMAP analysis"""

import os
from collections import Counter
from typing import Callable, Union, Iterable, List, Dict
import json
import itertools

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
from wealth.util import data_dir

DFLT_REDUCER_SPECS = str(data_dir / 'reducer_specs.json')

RawData = List[List[pd.DataFrame]]
RawDataGetter = Callable[[], RawData]
Data = List[pd.DataFrame]
Values = List[np.ndarray]
Relation = Dict[int, int]
Relations = List[Relation]

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
def data(zip_file=DFLT_QUARTERLY_DATA_SRC) -> Data:
    return list(QuarterlyData(zip_file).values())


_get_data = data  # alias to be used internally


def relations_for_dfs(from_df, to_df) -> Relation:
    left = pd.DataFrame(data=np.arange(len(from_df)), index=from_df.index)
    right = pd.DataFrame(data=np.arange(len(to_df)), index=to_df.index)
    merge = pd.merge(left, right, left_index=True, right_index=True)
    return dict(merge.values)


def relations(data: Data) -> Relations:
    return [relations_for_dfs(x, y) for x, y in zip(data[:-1], data[1:])]


def assert_relations(data: Data, relations: Relations):
    for data_idx, relation in enumerate(relations):
        for i, j in relation.items():
            assert data[data_idx].iloc[i].name, data[data_idx + 1].iloc[j].name


def assert_all_data_dfs_have_same_columns(data: Data):
    first_data_columns = data[0].columns
    for i, d in enumerate(data):
        assert all(d.columns == first_data_columns), (
            f'data[{i}] has different columns than ' f'data[0]'
        )


def ticker_counter(data: Data):
    c = Counter()
    for d in data:
        c.update(d.index.values)
    return pd.Series(c).sort_values(ascending=False)


def get_all_tickers(data: Data):
    return sorted(ticker_counter(data))


def weird_tickers(tickers):
    def not_normal_ticker(t):
        return not isinstance(t, str) or len(t) > 4 or t != t.upper()

    return sorted(filter(not_normal_ticker, tickers))


def tickers_present_everywhere(data):
    c = ticker_counter(data)
    return c[c >= len(data)].index.values


def sample_tickers(
    data: Data,
    n_tickers: int = 50,
    ticker_choice_col: str = 'EBITDA',
    sort_ascending: bool = False,
    only_tickers_present_everywhere: bool = True,
):
    df = pd.concat(data).groupby('Ticker').mean()
    if only_tickers_present_everywhere:
        ticker_choices = tickers_present_everywhere(data)
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
    ticker_choice_col: str = 'EBITDA',
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


# Pattern: meshed
def get_tickers(data: Data, spec='full_and_not_weird'):
    if isinstance(spec, str) and str.isnumeric(spec):
        spec = int(spec)
    if isinstance(spec, str):
        if spec == 'full_and_not_weird':
            tickers = tickers_present_everywhere(data)
            _weird_tickers = weird_tickers(tickers)
            return list(set(tickers) - set(_weird_tickers))
        elif os.path.isfile(spec):
            if spec.endswith('.json'):
                with open(spec) as fp:
                    spec = json.load(fp)
                return spec
        else:
            raise ValueError(f'Unknown spec: {spec}')
    elif isinstance(spec, int):
        return sample_tickers(data, n_tickers=spec)
    else:
        return spec


def slice_data_with_tickers(data: Data, tickers):
    return [d.loc[np.array(tickers)] for d in data]


std_normalizer = StandardScaler()
pca_normalizer = Pipeline(
    steps=[('zscore', StandardScaler()), ('decomposer', PCA(n_components=17))]
)


def values(data: Data, normalizer=None):
    _values = [d.values for d in data]
    if normalizer:
        normalizer.fit(np.vstack(_values))
        _values = list(map(normalizer.transform, _values))
    return _values


def get_values_and_relations(data: Data, normalizer=None):
    return values(data, normalizer), relations(data)


DFLT_ALIGNED_UMAP_KWARGS = dict(
    n_neighbors=5,  # default: 15
    n_components=2,  # 2
    metric='euclidean',  # 'euclidean'
    metric_kwds=None,  # None
    n_epochs=None,  # None
    learning_rate=1.0,  # 1.0
    init='spectral',  # 'spectral'
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
    target_metric='categorical',  # 'categorical'
    target_metric_kwds=None,  # None
    target_weight=0.5,  # 0.5
    transform_seed=42,  # 42
    force_approximation_algorithm=False,  # False
    verbose=False,  # False
    unique=False,  # False
)


def default_reducer():
    return umap.aligned_umap.AlignedUMAP(**DFLT_ALIGNED_UMAP_KWARGS)


# Pattern: meshed
def get_reducer(reducer=None):
    if reducer is None:
        return default_reducer()

    if isinstance(reducer, str) and os.path.isfile(reducer):
        if reducer.endswith('.json'):
            with open(reducer) as fp:
                reducer = json.load(fp)

    if isinstance(reducer, dict):
        kwargs = dict(DFLT_ALIGNED_UMAP_KWARGS, **reducer)
        return umap.aligned_umap.AlignedUMAP(**kwargs)
    else:
        return reducer


def get_dict_from_json(filepath=DFLT_REDUCER_SPECS):
    with open(filepath) as fp:
        reducers = json.load(fp)
    return reducers


def get_reducers(reducers=DFLT_REDUCER_SPECS):
    if isinstance(reducers, str):
        if os.path.isfile(reducers) and reducers.endswith('.json'):
            reducers = get_dict_from_json(reducers)
        elif str.isnumeric(reducers):
            reducers = int(reducers)
    if isinstance(reducers, int):
        reducers = get_dict_from_json()[:reducers]
    return map(get_reducer, reducers)


def embeddings(
    reducer,
    values,
    relations: Relations = None,
    # relations=None,
):
    reducer = get_reducer(reducer)
    if relations is None:
        data = values
        values, relations = get_values_and_relations(data)
    fitted = reducer.fit(values, relations=relations)
    return fitted.embeddings_


DFLT_TICKERS = tuple(get_tickers(data(), spec='full_and_not_weird'))

from wealth.util import print_progress


def default_reducer_to_save_name(reducer, ext='.json'):
    r = reducer
    return (
        'reducer_'
        f'{r.alignment_regularisation=},'
        f'{r.alignment_window_size=},'
        f'{r.n_neighbors=},'
        f'{r.n_components=}' + ext
    )


def reducer_specs_gen(**kwargs):
    fields = list(kwargs)
    assert set(fields).issubset(set(DFLT_ALIGNED_UMAP_KWARGS)), 'unknown reducer fields'

    def normalize_kwargs(kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, Iterable) or isinstance(v, str):
                v = [v]
            yield k, v

    kwargs = dict(normalize_kwargs(kwargs))

    for combos in itertools.product(*kwargs.values()):
        yield dict(zip(fields, combos))


from i2 import Sig

_reducer_fields = set(Sig(default_reducer().__init__))


def reducer_jdict(reducer):
    return {k: reducer.__dict__[k] for k in _reducer_fields & set(reducer.__dict__)}


def compute_and_save_embeddings_from_multiple_reducers(
    data: Data = None,
    tickers=DFLT_TICKERS,
    normalizer=None,
    reducers=None,
    reducer_to_save_name=default_reducer_to_save_name,
    save_dirpath='.',
):
    if reducers is None:
        reducers = [default_reducer()]
    if data is None:
        data = _get_data()
    tickers = get_tickers(data, tickers)

    reducers = list(get_reducers(reducers))[::-1]

    _data = slice_data_with_tickers(data, tickers)
    _values = values(_data, normalizer)
    _relations = relations(_data)

    # reducers = list(reducers)
    # print(f"{len(data)=}, {len(tickers)=}, {len(reducers)=}")

    from py2store import QuickJsonStore

    save_dirpath = os.path.expanduser(os.path.abspath(save_dirpath))
    store = QuickJsonStore(save_dirpath)

    for i, reducer in enumerate(reducers):
        print_progress(f'{i}: {reducer}')
        name = reducer_to_save_name(reducer)
        _embeddings = embeddings(reducer, _values, _relations)
        store[name] = {
            'tickers': np.array(list(tickers)).tolist(),
            'reducer': reducer_jdict(reducer),
            'embeddings': np.array(_embeddings).tolist(),
        }


from functools import partial
from sklearn.cluster import KMeans

from lined import Pipe

embedding_movement = Pipe(partial(np.diff, axis=0), np.abs, np.mean)
embedding_movement.__name__ = 'embedding_movement'


def embedding_clusterness(embeddings, n_clusters=11):
    embeddings = np.array(embeddings)
    if embeddings.ndim == 3:
        return np.mean([embedding_clusterness(x, n_clusters) for x in embeddings])
    else:  # single embedding
        model = KMeans(n_clusters=11).fit(embeddings).inertia_
        return model


from functools import partial
import json
from typing import Mapping
from py2store import FilesOfZip, wrap_kvs, filt_iter, QuickJsonStore
from graze import graze


StoreOrFuncToGetIt = Union[Mapping, Callable[[], Mapping]]
dflt_embeddings_store_location = (
    'https://www.dropbox.com/s/t1lbt21gezxg5ao/embedding_dump.zip?dl=0'
)


def _is_url(x):
    return x.startswith('http')


def get_embeddings_data_store(store: str = dflt_embeddings_store_location):
    if isinstance(store, str):
        store_location = store

        if _is_url(store_location):
            store = FilesOfZip(graze(store_location))
        elif os.path.isfile(store_location):
            store = FilesOfZip(store_location)
        elif os.path.isdir(store_location):
            store = QuickJsonStore(store_location)

        store_wrapper = Pipe(
            filt_iter(
                filt=lambda x: not x.startswith('__MACOSX')
                and not x.endswith('.DS_Store')
            ),
            wrap_kvs(obj_of_data=json.loads),
        )
        store = store_wrapper(store)

    assert isinstance(store, Mapping), (
        f'store should be a mapping at this point: ' f'{store}'
    )

    return store


def compute_embedding_stats(embedding, named_funcs):
    return {name: func(embedding) for name, func in named_funcs.items()}


def embedding_stats(*funcs, **named_funcs):
    named_funcs = dict({f.__name__: f for f in funcs}, **named_funcs)

    return partial(
        compute_embedding_stats,
        named_funcs=dict({f.__name__: f for f in funcs}, **named_funcs),
    )


dflt_embedding_stats = embedding_stats(embedding_movement, embedding_clusterness)


def reducer_and_embedding_stats(
    embeddings_store: StoreOrFuncToGetIt = dflt_embeddings_store_location,
    reducer_stats=lambda x: {k: v for k, v in x.items() if v is not None},
    embedding_stats=dflt_embedding_stats,
    verbose=False,
):
    s = get_embeddings_data_store(embeddings_store)
    for i, k in enumerate(s):
        if verbose:
            print_progress(f'reducer_and_embedding_stats {i}: {k}')
        try:
            v = s[k]
            yield dict(
                embedding_stats(v['embeddings']), key=k, **reducer_stats(v['reducer']),
            )
        except Exception as e:
            print(f'!!! Error with {k}')


DFLT_EMBEDDING_STATS_JSON_FILEPATH = str(data_dir / 'embeddings_stats.json')


def get_saved_embedding_stats(json_filepath=DFLT_EMBEDDING_STATS_JSON_FILEPATH):
    with open(json_filepath) as fp:
        d = json.load(fp)
    return d


if __name__ == '__main__':
    from argh import dispatch_command

    dispatch_command(compute_and_save_embeddings_from_multiple_reducers)
