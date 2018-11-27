#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

features plot

pandas 按行取数据
1. 连续多行的选择用类似于python的列表切片
2. 按照指定的索引选择一行或多行，使用loc[]方法
3. 按照指定的位置选择一行多多行，使用iloc[]方法

pandas 按列取数据
1. 单列 all_data['POST']
2. 多列 all_data[['POST', 'CATEGORY']]

@author: zxj

"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

font = FontProperties(fname=r"./naive_bayes/datasets/font/PingFang.ttc", size=9)

# parse dataframe to map:
# use 'name' for key, and 'value' for value
def dfToMap(df, name='name', value='value'):
    map = {}
    for index, series in df.iterrows():
        map[series[name]] = series[value]
    
    return map


def plot(column, title, x_label_dict):
    # parse data
    # 'PROVINCE','OPERATOR','CLUE','INDUSTRY','OWNER','POST','CATEGORY'
    all_data = pd.read_csv('./naive_bayes/datasets/source/all_data.csv', header=0)

    # stat
    all_cate = all_data[[column, 'CATEGORY', 'ACCOUNTID']]
    all_rate = all_cate.groupby([column]).count().reset_index().set_index(column)
    all_converted = all_cate[all_cate['CATEGORY'] == 1].groupby([column, 'CATEGORY']).count().reset_index().set_index(column)

    # concat
    all_result = pd.concat([all_rate['CATEGORY'], all_converted['ACCOUNTID']], axis=1, sort=False)
    all_result = all_result.fillna(0).astype(float)
    all_result.rename(columns=({'CATEGORY': 'TOTAL', 'ACCOUNTID': 'SIGNED'}), inplace=True)
    # all_result_sum = all_result.sum()
    # all_result['TOTAL%'] = all_result['TOTAL'] / all_result_sum['TOTAL']
    all_result['SIGNED'] = all_result['SIGNED'] / all_result['TOTAL']

    # show data
    all_index = all_result.index.map(lambda x: '未知' if x_label_dict[x]=='00000' else x_label_dict[x])
    all_total = all_result[['TOTAL', 'SIGNED']]
    all_total = all_total.set_index(all_index)
    
    # print

    # plot
    # all_result.diff().hist()
    # xticks = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    width = len(all_total['TOTAL'])
    fig = plt.figure(figsize=[width, 6])
    ax1 = all_total.TOTAL.plot(kind='bar', color='xkcd:sky blue', legend=True)
    ax2 = all_total.SIGNED.plot(secondary_y=True, color='xkcd:mango', legend=True)

    for i, label in enumerate(list(all_total.index)):
        score = all_total.ix[label]['TOTAL']
        ax1.annotate(int(score), (i-0.1, score+0.6))

    for label in ax1.get_xticklabels(): 
        label.set_fontproperties(font)

    ax1.set_xlabel('')
    ax1.set_ylabel('样本总数', fontproperties=font)
    # ax2.set_xlabel('POST')
    ax2.set_ylabel('转化率', fontproperties=font)

    plt.grid(linestyle='--')
    plt.title(title, fontproperties=font, fontsize=12)
    plt.show()
    fig.savefig('./naive_bayes/stat.png')

if __name__ == "__main__":
    stat_key = 'INDUSTRY'
    dict_file = './naive_bayes/datasets/dict/dict_%(name)s.csv'%{'name':stat_key.lower()}
    x_label_df = pd.read_csv(dict_file, header=0)
    x_label_dict = dfToMap(x_label_df, name='value', value='name')
    plot(stat_key, title='统计图表', x_label_dict=x_label_dict)