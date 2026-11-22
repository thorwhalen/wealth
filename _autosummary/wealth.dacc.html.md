# wealth.dacc

Data access

### Functions

| `get_data`(b)   |    |
|-----------------|----|

### Classes

| [`QuarterlyData`](#wealth.dacc.QuarterlyData)([zip_file, prefix, open_kws])   | Reads dataframes of quarterly data   |
|------------------------------------------------------------------------------------------------|--------------------------------------|

### *class* wealth.dacc.QuarterlyData(zip_file=PosixPath('/home/runner/work/wealth/wealth/wealth/data/csv_derived.zip'), prefix='', open_kws=None)

Bases: `Store`

Reads dataframes of quarterly data

```pycon
>>> from wealth.dacc import QuarterlyData
>>> import pandas as pd
>>> data = QuarterlyData()
>>> len(data)
44
```

Keys are (year, quarter) pairs:

```pycon
>>> list(data)[:3]
[(2010, 'Q1'), (2010, 'Q2'), (2010, 'Q3')]
```

Values are pandas.DataFrames whose indices are tickers (or groups, or whatever,
but rows are the items under study) and whose columns represent their features.

```pycon
>>> d = data[2010, 'Q1']
>>> assert isinstance(d, pd.DataFrame)
>>> list(d.index)[:4]
['AAP', 'AAPL', 'ABT', 'ACIW']
>>> list(d.columns)[:4]
['EBITDA', 'Total Debt', 'Free Cash Flow', 'Gross Profit Margin']
```

#### update(\*\*F) → None.  Update D from mapping/iterable E and F.

If E present and has a .keys() method, does:     for k in E.keys(): D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v

#### update_keys_cache(keys)

Updates the \_keys_cache by deleting the attribute
