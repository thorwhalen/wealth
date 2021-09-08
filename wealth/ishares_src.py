"""Getting data from ishares.com"""


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
