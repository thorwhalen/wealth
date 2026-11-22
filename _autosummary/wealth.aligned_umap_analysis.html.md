# wealth.aligned_umap_analysis

Aligned UMAP analysis

Examples

```python
compute_and_save_embeddings_from_multiple_reducers(
    # specify tickers: 'all' or 'full_and_not_weird' (default),
    # or a number (of sampled tickers),
    # or explicit list or json filepath to the list
    tickers='all',
    # specify reducer specs
    reducers=[
        {
            "alignment_regularisation": 0.002,
            "alignment_window_size": 2,
            "n_neighbors": 5,
            "n_components": 2
        },
        {
            "alignment_regularisation": 0.002,
            "alignment_window_size": 2,
            "n_neighbors": 5,
            "n_components": 3
        },
    ],
)
```

From the command line:

```text
python aligned_umap_analysis --tickers all
```

### Functions

| `assert_all_data_dfs_have_same_columns`(data)                                                              |                                                                             |
|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `assert_relations`(data, relations)                                                                        |                                                                             |
| [`compute_and_save_embeddings_from_multiple_reducers`](#wealth.aligned_umap_analysis.compute_and_save_embeddings_from_multiple_reducers)([...]) | Compute one or several embeddings for the given data (and specific tickers) |
| `compute_embedding_stats`(embedding, named_funcs)                                                          |                                                                             |
| `data`([zip_file])                                                                                         |                                                                             |
| `default_reducer`()                                                                                        |                                                                             |
| `default_reducer_to_save_name`(reducer[, ext])                                                             |                                                                             |
| `embedding_clusterness`(embeddings[, n_clusters])                                                          |                                                                             |
| `embedding_stats`(\*funcs, \*\*named_funcs)                                                                |                                                                             |
| `embeddings`(reducer, values[, relations])                                                                 |                                                                             |
| `get_all_tickers`(data)                                                                                    |                                                                             |
| `get_dict_from_json`([filepath])                                                                           |                                                                             |
| `get_embeddings_data_store`([store])                                                                       |                                                                             |
| `get_reducer`([reducer])                                                                                   |                                                                             |
| `get_reducers`([reducers])                                                                                 |                                                                             |
| `get_saved_embedding_stats`([json_filepath])                                                               |                                                                             |
| `get_tickers`(data[, spec])                                                                                |                                                                             |
| `get_values_and_relations`(data[, normalizer])                                                             |                                                                             |
| `reducer_and_embedding_stats`([...])                                                                       |                                                                             |
| `reducer_jdict`(reducer)                                                                                   |                                                                             |
| `reducer_specs_gen`(\*\*kwargs)                                                                            |                                                                             |
| `relations`(data)                                                                                          |                                                                             |
| `relations_for_dfs`(from_df, to_df)                                                                        |                                                                             |
| `sample_data`(data[, tickers, ...])                                                                        |                                                                             |
| `sample_tickers`(data[, n_tickers, ...])                                                                   |                                                                             |
| `slice_data_with_tickers`(data, tickers)                                                                   |                                                                             |
| `ticker_counter`(data)                                                                                     |                                                                             |
| `tickers_present_everywhere`(data)                                                                         |                                                                             |
| `values`(data[, normalizer])                                                                               |                                                                             |
| `weird_tickers`(tickers)                                                                                   |                                                                             |

### wealth.aligned_umap_analysis.compute_and_save_embeddings_from_multiple_reducers(data=None, tickers=('AAP', 'AAPL', 'ABT', 'ACIW', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'ADUS', 'AEE', 'AEO', 'AHC', 'AIG', 'AIZ', 'ALL', 'AME', 'AMED', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANDE', 'ANTM', 'AON', 'APA', 'APD', 'APH', 'ARNC', 'AROC', 'ASGN', 'ATVI', 'AVB', 'AVP', 'AVT', 'AXP', 'AZO', 'BA', 'BAX', 'BBY', 'BC', 'BCOR', 'BDX', 'BG', 'BHI', 'BIIB', 'BMY', 'BRKA', 'BSX', 'BXC', 'BXP', 'C', 'CAG', 'CAH', 'CALX', 'CCI', 'CCL', 'CDZI', 'CE', 'CF', 'CFR', 'CFX', 'CHH', 'CHK', 'CHRW', 'CL', 'CLX', 'CMA', 'CME', 'CMI', 'CNP', 'COF', 'COG', 'CPB', 'CRI', 'CRM', 'CSCO', 'CSX', 'CTSH', 'CTXS', 'CVS', 'CVX', 'CW', 'D', 'DDS', 'DE', 'DEI', 'DFS', 'DHR', 'DISH', 'DIT', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DTE', 'DVA', 'DVN', 'EBAY', 'EFX', 'EIX', 'EMN', 'EMR', 'EOG', 'EPD', 'EQT', 'ESV', 'ETR', 'EVR', 'EXC', 'EXPD', 'EXPO', 'F', 'FAST', 'FCFS', 'FCN', 'FFIN', 'FIS', 'FISV', 'FITB', 'FL', 'FLR', 'FMC', 'FSLR', 'FTK', 'GD', 'GE', 'GILD', 'GLW', 'GME', 'GNRC', 'GNW', 'GOOG', 'GPC', 'GPS', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCP', 'HES', 'HIG', 'HOG', 'HOLX', 'HPQ', 'HROW', 'HST', 'HSY', 'HTLD', 'HUM', 'IBM', 'ICUI', 'IMO', 'INOD', 'INTC', 'INTU', 'IP', 'IR', 'IRDM', 'IRM', 'ISDR', 'ISRG', 'ITT', 'IVZ', 'JEC', 'JNJ', 'JNPR', 'JPM', 'JW-A', 'JWN', 'K', 'KBH', 'KBR', 'KDP', 'KEX', 'KEY', 'KMB', 'KO', 'KR', 'KSS', 'KSU', 'L', 'LAD', 'LB', 'LECO', 'LG', 'LH', 'LLY', 'LMNR', 'LMT', 'LNC', 'LOW', 'LUV', 'LVS', 'M', 'MA', 'MAR', 'MAS', 'MAT', 'MCD', 'MCO', 'MDCA', 'MDLZ', 'MDU', 'MET', 'MKC', 'MLM', 'MMC', 'MMM', 'MO', 'MRK', 'MRVL', 'MS', 'MSFT', 'MSI', 'MTB', 'MTG', 'MTN', 'MUR', 'NEE', 'NEU', 'NFLX', 'NKE', 'NOC', 'NOV', 'NRG', 'NS', 'NSP', 'NTRS', 'NUE', 'NVDA', 'NWPX', 'ODP', 'OI', 'OKE', 'OLN', 'OMC', 'OPRX', 'ORCL', 'ORLY', 'OTEL', 'OTEX', 'OXY', 'PAYX', 'PBCT', 'PCG', 'PCH', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PM', 'PNC', 'PNM', 'PNR', 'POOL', 'PPG', 'PPL', 'PRI', 'PRU', 'PSA', 'PSEG', 'PTC', 'PTEN', 'PVH', 'PWR', 'PZZA', 'QCOM', 'QDEL', 'RAVN', 'RCL', 'RF', 'RGLD', 'RIG', 'ROG', 'ROIA', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'SBUX', 'SCHL', 'SCHN', 'SCHW', 'SCI', 'SHEN', 'SHW', 'SLAB', 'SLB', 'SLGN', 'SLM', 'SMG', 'SPG', 'SPXC', 'SSD', 'STRA', 'SWN', 'SYK', 'T', 'TAP', 'TEL', 'TER', 'TGNA', 'TGT', 'TJX', 'TKR', 'TR', 'TROW', 'TRV', 'TSCO', 'TTC', 'TTEK', 'TXN', 'TXT', 'UA', 'ULTA', 'UMBF', 'UNH', 'UNM', 'UPS', 'UTL', 'V', 'VAR', 'VLO', 'VMC', 'VNO', 'VRSN', 'VTR', 'VZ', 'WCC', 'WDC', 'WEC', 'WFC', 'WM', 'WMB', 'WMG', 'WMT', 'WRB', 'WSM', 'WSO', 'WST', 'WU', 'WY', 'X', 'XEC', 'XEL', 'XOM', 'XRAY', 'XRX', 'YELL', 'ZBRA', 'ZMH'), normalizer=None, reducers=None, reducer_to_save_name=<function default_reducer_to_save_name>, save_dirpath='.')

Compute one or several embeddings for the given data (and specific tickers)

* **Parameters:**
  * **data** (`Data` ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`DataFrame`])) – list of dataframes with ticker indices.
    Note specifying it will result in getting this from data/csv_derived.zip
  * **tickers** (`TickerSpec` ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)])) – iterable of tickers,
    or string of json file to get these,
    or string specifying how to computer these from data
  * **normalizer** – sklearn transformer (has `.fit` & `.transform`) to preprocess
    values (example; std_normalizer (= StandardScaler()), or pca_normalizer
  * **reducers** – An AlignedUMAP instance or specification (ReducerSpec) to get one.
  * **reducer_to_save_name** – How to compute a save name/key from a reducer instance
  * **save_dirpath** – What directory to save the data into
* **Returns:**
