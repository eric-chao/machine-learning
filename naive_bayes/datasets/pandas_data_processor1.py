#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

use pandas to process data.

@author: zxj

"""
from pandas_func import nanToZero, dfToMap, phoneMap, phoneNumberTrim, PostNormal
from sklearn.utils import shuffle
import pandas as pd

# na values
na_values = ['nan', '0']

# source frame
# 0-INDEX, 1-ACCOUNTID, 2-PHONE, 3-PROVINCE, 4-OPERATOR, 5-POST, 6-TRADENAME, 7-CLUE, 8-INDUSTRY, 9-OWNER, 10-CATEGORY
source_file = './naive_bayes/datasets/source/all_rm_duplicated.xls'
col_dtype = {'ACCOUNTID': str, 'PHONE': str, 'PROVINCE': str, 'OPERATOR': str,'POST': str, 'TRADENAME':str, 'CLUE': str, 'INDUSTRY': str, 'OWNER': str, 'CATEGORY': str}
all_data_frame = pd.DataFrame(pd.read_excel(source_file, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], na_values=na_values, dtype=col_dtype))

# dict frame
dict_clue = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_clue.csv', dtype={'name': str, 'value': float}))
dict_post = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_post.csv', dtype={'name': str, 'value': float}))
dict_owner = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_owner.csv', dtype={'name': str, 'value': float}))
dict_industry = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_industry.csv', dtype={'name': str, 'value': float}))
dict_province = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_province.csv', dtype={'name': str, 'value': float}))
dict_operator = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_operator.csv', dtype={'name': str, 'value': float}))

# dict to map
map_clue = dfToMap(dict_clue)
map_post = dfToMap(dict_post)
map_owner = dfToMap(dict_owner)
map_industry = dfToMap(dict_industry)
map_province = dfToMap(dict_province)
map_operator = dfToMap(dict_operator)

# process all_data_frame
all_id = all_data_frame['ACCOUNTID']
all_clue = all_data_frame['CLUE'].map(nanToZero).map(lambda x: map_clue[x])
all_post = all_data_frame['POST'].map(PostNormal).map(lambda x: map_post[x])
all_owner = all_data_frame['OWNER'].map(nanToZero).map(lambda x: map_owner[x])
all_industry = all_data_frame['INDUSTRY'].map(nanToZero).map(lambda x: map_industry[x])
all_province = all_data_frame['PROVINCE'].map(nanToZero).map(lambda x: map_province[x])
all_operator = all_data_frame['OPERATOR'].map(nanToZero).map(lambda x: map_operator[x])
all_is_signed = all_data_frame['CATEGORY'].str.contains('签约').astype(int)

# union
all_keys = ['ACCOUNTID', 'PROVINCE', 'OPERATOR', 'CLUE', 'INDUSTRY', 'OWNER', 'POST', 'CATEGORY']
all_union = pd.concat([all_id, all_province, all_operator, all_clue, all_industry, all_owner, all_post, all_is_signed], axis=1, keys=all_keys)

# rename
# all_union.rename(columns=({'手机': 'PROV', '手机': 'OPER', '线索来源': 'CLUE', '行业': 'INDU', '交易所有者': 'OWNE', '阶段': 'CATE'}), inplace=True)

# write to csv
# all_union = shuffle(all_union)
all_union.to_csv('all_data.csv', float_format='%.3f', encoding='UTF-8', index=0)
print(all_union)