#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

self-defined functions

@author: zxj

"""
import pandas as pd

# parse nan to '00000'
def nanToZero(value):
    if value == 'nan':
        return '00000'
    
    return value


# parse dataframe to map:
# use 'name' for key, and 'value' for value
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


PostDict = {
    '总裁': ['vp', '总裁', '老板', '董事', '合伙人', '创始人', '总经理', '副总', 'ceo', 'cto', 'coo', 'cxo'],
    '总监': ['总监', '监事', '科学家'],
    '经理': ['经理', '科长', '部长', 'manager'],
    '组长': ['tl', 'leader', 'team leader', '组长', '主管', '负责人', 'pm'],
    '产品': ['产品'],
    '运营': ['运营', 'am', 'ab'],
    '商务': ['商务', '销售'],
    '市场': ['市场', '推广'],
    # '其他': ['***', 'nan', '未知', '待确认'],
    '技术': ['技术', 'it', '开发', '研发', '架构师', '工程师', '设计师', '分析师', '科员']}
def PostNormal(post):
    normal = post.lower()

    for key in PostDict.keys():
        for p in PostDict[key]:
            if p in normal:
                return key
                
    return '其他'


# parse data
phone_data_frame = pd.DataFrame(pd.read_excel('./naive_bayes/datasets/dict/phone.xls', usecols=[0, 1, 3], dtype={'手机号': str, '省份': str, '运营商': str}))
phone_data_frame = phone_data_frame.append(pd.DataFrame([['00000', '00000', '00000']], columns=['手机号', '省份', '运营商']), ignore_index=True)

# dict frame
dict_province = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_province.csv', dtype={'name': str, 'value': float}))
dict_operator = pd.DataFrame(pd.read_csv('./naive_bayes/datasets/dict/dict_operator.csv', dtype={'name': str, 'value': float}))

map_province = dfToMap(dict_province)
map_operator = dfToMap(dict_operator)
map_phone_province = dfToMap(phone_data_frame, '手机号', '省份')
map_phone_operator = dfToMap(phone_data_frame, '手机号', '运营商')
