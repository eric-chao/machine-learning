#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

remove duplicated

@author: zxj

"""
from pandas_func import map_province, map_operator, map_phone_province, map_phone_operator
from pandas_func import nanToZero, dfToMap, phoneMap, phoneNumberTrim
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # parse data
    # 0 - ID  4 - 行业  6 - 手机号  8 - 职位  10 - 交易名称  12 - 线索来源  16 - 交易所有者  14 - 阶段
    xls_file = r"/Users/adhoc-dev/Downloads/Books/adhoc/1.xls"
    dest_file = r'./naive_bayes/datasets/source/all.0.xls'
    na_values = ['NO CLUE', 'N/A', '0']
    col_dtype = {'ACCOUNTID': str, '行业': str, '手机': str, '职位': str, '交易名称':str, '线索来源': str, '交易所有者': str, '阶段': str}
    all_data_frame = pd.DataFrame(pd.read_excel(xls_file, usecols=[0, 4, 6, 8, 10, 12, 16, 14], na_values=na_values, dtype=col_dtype))

    all_phone = all_data_frame['手机'].map(nanToZero).map(phoneNumberTrim)
    all_province = all_phone.map(lambda x: phoneMap(map_phone_province, x, '00000'))
    all_operator = all_phone.map(lambda x: phoneMap(map_phone_operator, x, '00000'))

    # union
    all_id = all_data_frame['ACCOUNTID']
    all_clue = all_data_frame['线索来源']
    all_industry = all_data_frame['行业']
    all_owner = all_data_frame['交易所有者']
    all_category = all_data_frame['阶段']
    all_post = all_data_frame['职位']
    all_tradename = all_data_frame['交易名称']

    all_keys = ['ACCOUNTID', 'PHONE', 'PROVINCE', 'OPERATOR', 'POST', 'TRADENAME', 'CLUE', 'INDUSTRY', 'OWNER', 'CATEGORY']
    all_data_union = pd.concat([all_id, all_phone, all_province, all_operator, all_post, all_tradename, all_clue, all_industry, all_owner, all_category], join='inner', axis=1, keys=all_keys)
    
    # filter all_data_union
    is_owned = all_data_union['OWNER'] != 'nan'
    all_data_union = all_data_union[~all_data_union['TRADENAME'].str.contains('续约')]
    all_data_union = all_data_union[is_owned]

    print(all_data_union.count())
    all_data_union.to_excel(dest_file)
