#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

use pandas to process data.

@author: zxj

"""
import pandas as pd

# parse nan to '00000'
def nanToZero(value):
    if value == 'nan':
        return '00000'
    
    return value

# parse dataframe to map:
# use 'name' fro key, and 'value' for value
def dfToMap(df, name='name', value='value'):
    map = {}
    for index, series in df.iterrows():
        map[series[name]] = series[value]
    
    return map

# trim '-' & '+86'
def phoneNumberTrim(phone):
    return phone.replace('-', '').lstrip('+86')

# if phone in the map return map[phone]
# else return defaultValue
def phoneMap(map, phone, defautValue):
    if phone in map:
        return map[phone]
    
    return defautValue

# na values
na_values = ['nan', '0']

# source frame
# 4 - 行业  6 - 手机号  12 - 线索来源  16 - 交易所有者  14 - 阶段
all_data_frame = pd.DataFrame(pd.read_excel('/Users/adhoc-dev/Downloads/Books/adhoc/1.xls', usecols=[4, 6, 12, 16, 14], na_values=na_values, dtype={'行业': str, '手机': str, '线索来源': str, '交易所有者': str, '阶段': str}))
phone_data_frame = pd.DataFrame(pd.read_excel('./naive_bayes/datasets/source/phone.xls', usecols=[0, 1, 3], dtype={'手机号': str, '省份': str, '运营商': str}))
phone_data_frame = phone_data_frame.append(pd.DataFrame([['00000', '00000', '00000']], columns=['手机号', '省份', '运营商']), ignore_index=True)

# dict frame
dict_province = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_province.csv', dtype={'name': str, 'value': float}))
dict_operator = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_operator.csv', dtype={'name': str, 'value': float}))
dict_clue = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_clue.csv', dtype={'name': str, 'value': float}))
dict_owner = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_owner.csv', dtype={'name': str, 'value': float}))
dict_industry = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_industry.csv', dtype={'name': str, 'value': float}))

# dict to map
map_clue = dfToMap(dict_clue)
map_owner = dfToMap(dict_owner)
map_industry = dfToMap(dict_industry)
map_province = dfToMap(dict_province)
map_operator = dfToMap(dict_operator)
map_phone_province = dfToMap(phone_data_frame, '手机号', '省份')
map_phone_operator = dfToMap(phone_data_frame, '手机号', '运营商')

# process phone 
# phone_union = phone_data_frame.merge(dict_province, left_on=['省份'], right_on=['name'], how='left')
# phone_union = phone_union.merge(dict_operator, left_on=['运营商'], right_on=['name'], how='left')

# process all_data_frame
all_clue = all_data_frame['线索来源'].map(nanToZero).map(lambda x: map_clue[x])
all_owner = all_data_frame['交易所有者'].map(nanToZero).map(lambda x: map_owner[x])
all_industry = all_data_frame['行业'].map(nanToZero).map(lambda x: map_industry[x])

all_phone = all_data_frame['手机'].map(nanToZero).map(phoneNumberTrim)
all_province = all_phone.map(lambda x: phoneMap(map_phone_province, x, '00000')).map(nanToZero).map(lambda x: map_province[x])
all_operator = all_phone.map(lambda x: phoneMap(map_phone_operator, x, '00000')).map(nanToZero).map(lambda x: map_operator[x])
all_is_signed = all_data_frame['阶段'].str.contains('签约').astype(float)

# union
all_keys = ['PROVINCE', 'OPERATOR', 'CLUE', 'INDUSTRY', 'OWNER', 'CATEGORY']
all_union = pd.concat([all_province, all_operator, all_clue, all_industry, all_owner, all_is_signed], axis=1, keys=all_keys)

# rename
# all_union.rename(columns=({'手机': 'PROV', '手机': 'OPER', '线索来源': 'CLUE', '行业': 'INDU', '交易所有者': 'OWNE', '阶段': 'CATE'}), inplace=True)

# write to csv
all_union.to_csv('all_data.csv', float_format='%.3f', encoding='UTF-8', index=0)
print(all_union)