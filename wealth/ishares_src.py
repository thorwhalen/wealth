"""Getting data from ishares.com"""

import pandas as pd
import matplotlib.pyplot as plt
from wealth.util import json_store

DFLT_HEADERS_STR = '''
-H ‘Accept: application/json, text/javascript, /; q=0.01’ -H ‘Cookie: AllowAnalytics=true; omni_newRepeat=1631073161998-Repeat; at_check=true; mbox=session#8889a7632acd48d4ab7c7afd31aab62e#1631074614; JSESSION_blk-one01=BEDD3BCE752184793FDA163E00C48453.04; ts-us-ishares-locale=en_US; us-ishares-recent-funds=239726’ -H ‘Accept-Language: en-us’ -H ‘Host: www.ishares.com’ -H ‘User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15’ -H ‘Referer: https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf’ -H ‘Accept-Encoding: gzip, deflate, br’ -H ‘Connection: keep-alive’ -H ‘X-Requested-With: XMLHttpRequest’
'''


def curl_headers_string_to_jdict(headers_string=DFLT_HEADERS_STR):
    import re

    headers = headers_string.replace('‘', '"').replace('’', '"').replace(': ', '": "')
    headers = re.sub(r'^\s*-H\s*', '{', headers)
    headers = re.sub(r'\s+$', '}', headers)
    headers = re.sub(r'\s*-H\s*', ', ', headers)
    return headers


def curl_headers_string_to_dict(headers_string=DFLT_HEADERS_STR):
    import json

    return json.loads(curl_headers_string_to_jdict(headers_string))


sp_500_composition_url_template = (
    'https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/'
    '1467271812596.ajax?tab=all&fileType=json&asOfDate={date}&_=1631072750760'
)


DFLT_HEADERS = {
    'Accept': 'application/json, text/javascript, /; q=0.01',
    'Cookie': 'AllowAnalytics=true; omni_newRepeat=1631073161998-Repeat; at_check=true; mbox=session#8889a7632acd48d4ab7c7afd31aab62e#1631074614; JSESSION_blk-one01=BEDD3BCE752184793FDA163E00C48453.04; ts-us-ishares-locale=en_US; us-ishares-recent-funds=239726',
    'Accept-Language': 'en-us',
    'Host': 'www.ishares.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Referer': 'https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'X-Requested-With': 'XMLHttpRequest',
}


DFLT_DATE = 2021_08_27

from typing import Union

Date = Union[int, str]


def response_for_date(date=DFLT_DATE, headers=None):
    headers = headers or DFLT_HEADERS
    import requests

    return requests.get(
        sp_500_composition_url_template.format(date=date), headers=headers
    )


def response_to_data(response):
    return response.json().get('aaData', [])


def is_valid_response(response):
    return response.status_code == 200 and len(response_to_data(response)) > 0


def day_of_int_integer(date=DFLT_DATE):
    return date % 100


def is_yyyymmdd_integer(date=DFLT_DATE):
    date = int(date)
    return date > 2000_00_00 and 1 <= day_of_int_integer(date) <= 31


def get_date(date: Date = DFLT_DATE):
    date = int(date)
    assert is_yyyymmdd_integer(date), f'date not valid: {date}'
    return date


def dflt_not_found_callback(x):
    print(f"Couldn't find anything with {x}")


def first_valid_from_date(
    start_date: Date = DFLT_DATE,
    include_response=True,
    not_found_callback=dflt_not_found_callback,
):
    start_date = get_date(start_date)
    max_day = max(day_of_int_integer(start_date), 28)
    for i in range(max_day + 1):
        date = start_date + i
        assert is_yyyymmdd_integer(date)
        r = response_for_date(date)
        if is_valid_response(r):
            if include_response:
                return date, r
            else:
                return date
    return not_found_callback(start_date)


def sorted_union_of_datas(*dicts):
    """Returns the union of the dicts, sorted (reversed) by keys"""
    from collections import ChainMap

    data_src = ChainMap(*dicts)
    return {k: data_src[k] for k in sorted(data_src)[::-1]}


# ---------------------------------------------------------------------------------------
# Applying these functions to actually get stuff

# Raw data ---------------------------------------------------
def _default_seed_start_dates():
    import numpy as np

    seed_start_dates = np.array([2020_10_01, 2020_07_01, 2020_04_01, 2020_01_01,])
    return list(
        map(
            int,
            np.concatenate(
                [
                    # seed_start_dates,
                    # seed_start_dates - 1_00_00,
                    # seed_start_dates - 2_00_00,
                    # seed_start_dates - 3_00_00,
                    # seed_start_dates - 4_00_00,
                    seed_start_dates + 1_00_00,
                    seed_start_dates - 5_00_00,
                    seed_start_dates - 6_00_00,
                    seed_start_dates - 7_00_00,
                    seed_start_dates - 8_00_00,
                    seed_start_dates - 9_00_00,
                ]
            ),
        )
    )


def _get_some_sp500_composition_raw_data():
    start_dates = _default_seed_start_dates()
    return map(first_valid_from_date, start_dates)


def prep_sp500_composition_raw_data(raw_data_iterator):
    return {date: response_to_data(r) for date, r in filter(None, raw_data_iterator)}


def get_some_sp500_composition_data():
    return prep_sp500_composition_raw_data(_get_some_sp500_composition_raw_data())


def data_getter(
    key='sp500_compositions.json',
    data_getter=get_some_sp500_composition_data,
    json_store=json_store,
):
    if key not in json_store:
        json_store[key] = data_getter()
    return json_store[key]


from functools import partial

get_raw_sp500_compositions = partial(
    data_getter,
    key='sp500_compositions.json',
    data_getter=get_some_sp500_composition_data,
)


# Data prep ---------------------------------------------------

import itertools
from operator import methodcaller
from lined import Pipe, map_star, iterize

item_2_kvs = Pipe(lambda item: ([item[0]], item[1]), map_star(itertools.product), tuple)
assert list(item_2_kvs([1, [2, 3, 4]])) == [(1, 2), (1, 3), (1, 4)]

data_2_kvs = Pipe(
    methodcaller('items'), iter, iterize(item_2_kvs), itertools.chain.from_iterable
)


def kv_to_data_dict(kv):
    date, vals = kv
    return dict(
        date=str(date),
        ticker=vals[0],  # AAPL
        name=vals[1],  # APPLE INC
        sector=vals[2],  # Information Technology
        asset_class=vals[3],  # Equity
        market_value=vals[4][
            'raw'
        ],  # {'display': '$16,943,729,311.11', 'raw': 16943729311.11}
        weight_perc=vals[5]['raw'],  # {'display': '5.88', 'raw': 5.88254}
        notational_value=vals[6][
            'raw'
        ],  # {'display': '16,943,729,311.11', 'raw': 16943729311.11}
        shares=vals[7]['raw'],  # {'display': '123,433,593.00', 'raw': 123433593}
        cusip=vals[8],  # 037833100
        isin=vals[9],  # US0378331005
        sedol=vals[10],  # 2046251
        some_num_1=vals[11]['raw'],  # {'display': '137.27', 'raw': 137.27}
        country=vals[12],  # United States
        market_group=vals[13],  # NASDAQ
        currency_1=vals[14],  # USD
        some_num_2=vals[15],  # 1.00
        currency_2=vals[16],  # USD
        no_clue=vals[17],  # -
    )


get_flattened_sp500_compositions = partial(
    data_getter,
    key='flattened_sp500_compositions.json',
    data_getter=lambda: list(
        map(kv_to_data_dict, data_2_kvs(get_raw_sp500_compositions()))
    ),
)


def get_df_for_sp500_compositions():
    df = pd.DataFrame(get_flattened_sp500_compositions())
    df['date'] = pd.to_datetime(df['date'].apply(str))
    return df.sort_values(['date', 'ticker']).reset_index(drop=True)


def double_plot(
    ticker,
    df=None,
    y1='market_value',
    y2='weight_perc',
    figsize=(16, 6),
    style='-o',
    **plot_kwargs,
):
    if df is None:
        df = get_df_for_sp500_compositions()
    d = df[df.ticker == ticker][['date', y1, y2]]
    ax = d.plot('date', 'market_value', figsize=figsize, style=style, **plot_kwargs)
    d.plot('date', 'weight_perc', secondary_y=True, ax=ax, style=style, **plot_kwargs)
    plt.title(ticker)
