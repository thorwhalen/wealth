# wealth.ishares_src

Getting data from ishares.com

### Functions

| `curl_headers_string_to_dict`([headers_string])                                 |                                                           |
|---------------------------------------------------------------------------------|-----------------------------------------------------------|
| `curl_headers_string_to_jdict`([headers_string])                                |                                                           |
| `data_getter`([key, data_getter, json_store])                                   |                                                           |
| `day_of_int_integer`([date])                                                    |                                                           |
| `dflt_not_found_callback`(x)                                                    |                                                           |
| `double_plot`(ticker[, df, y1, y2, figsize, style])                             |                                                           |
| `first_valid_from_date`([start_date, ...])                                      |                                                           |
| `get_date`([date])                                                              |                                                           |
| `get_df_for_sp500_compositions`()                                               |                                                           |
| `get_some_sp500_composition_data`()                                             |                                                           |
| `is_valid_response`(response)                                                   |                                                           |
| `is_yyyymmdd_integer`([date])                                                   |                                                           |
| `kv_to_data_dict`(kv)                                                           |                                                           |
| `prep_sp500_composition_raw_data`(...)                                          |                                                           |
| `response_for_date`([date, headers])                                            |                                                           |
| `response_to_data`(response)                                                    |                                                           |
| [`sorted_union_of_datas`](#wealth.ishares_src.sorted_union_of_datas)(\*dicts) | Returns the union of the dicts, sorted (reversed) by keys |
| `stackedplot`(df[, value_col, date_col, ...])                                   |                                                           |

### wealth.ishares_src.sorted_union_of_datas(\*dicts)

Returns the union of the dicts, sorted (reversed) by keys
