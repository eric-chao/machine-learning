#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

use pandas to read/write excel files.

@author: zxj

"""
from pandas_func import PostNormal
import pandas as pd
import numpy as np


def getExcelDataFrame(xls_file, col_name):
    na_values = ['NO CLUE', 'N/A', '0']
    excel = pd.read_excel(xls_file, na_values=na_values, dtype={col_name: str})
    return pd.DataFrame(excel)


# col_index:
# 4 -- industry(行业)
# 8 -- position(职位)
# 12 -- clue 线索来源
# 16 -- owner 交易所有者
def dataToFloat(xls_file, col_name):
    df = getExcelDataFrame(xls_file, col_name)
    # column = []
    # for index, series in df.iterrows():
    #     column.append(series[col_index])
    # 获取整列的另一种方式
    # print(df['客户名称'])

    # column_set = set(column)
    column_set = df[col_name].unique()
    # column_set.sort()
    np.sort(column_set, axis=None, kind='quicksort', order=None)

    index = 6.001
    for data in column_set:
        # print(type(data))
        # print(str(data) + ',' + PostNormal(data) + ',' + str(round(index,3)))
        print(PostNormal(data) + ',' + str(data))
        index = index + 0.001

if __name__=='__main__':
    xls_file = r"./naive_bayes/datasets/dict/phone.xls"
    xls_file = r"./naive_bayes/datasets/source/all_rm_duplicated.xls"
    dataToFloat(xls_file, 'POST')