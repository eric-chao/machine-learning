#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nb plot

@author: zxj

"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

font = FontProperties(fname=r"./naive_bayes/datasets/font/PingFang.ttc", size=9)

if __name__ == "__main__":
    # parse data
    all_data = pd.read_csv('./naive_bayes/datasets/source/high_auc_test_prob.csv', header=0, index_col=['NO'])
    all_data['QUOTIENT'] = round(all_data['PROB0'] / all_data['PROB1'])

    # stat
    all_result0 = all_data.groupby(['QUOTIENT']).count().reset_index()
    all_result1 = all_data.groupby(['QUOTIENT', 'TARGET']).count().reset_index()
    # all_result.rename(columns=({'PROB0': 'QUANTITY', 'PROB1': 'DENSITY'}), inplace=True)

    all_overall = all_result0[['QUOTIENT', 'PROB0']].set_index('QUOTIENT')
    all_signed = all_result1[all_result1['TARGET'] == 1][['QUOTIENT', 'PROB1']].set_index('QUOTIENT')

    # all_keys = ['QUOTIENT', 'PROB0', 'PROB1']
    all_result = pd.concat([all_overall, all_signed], axis=1)
    all_result = all_result.fillna(0).astype(float)
    all_result['RATIO'] = all_result['PROB1'] / all_result['PROB0']
    # print
    print(all_result)

    # plot
    all_index = all_result.index.map(lambda x: str(int(x)) + '倍')
    all_result = all_result.set_index(all_index)
    width = len(all_result.index)
    fig = plt.figure(figsize=[width * 0.8, 6])
    ax1 = all_result.PROB0.plot(kind='bar', color='xkcd:sky blue', legend=True)
    ax2 = all_result.RATIO.plot(secondary_y=True, color='xkcd:mango', legend=True)

    # 
    for label in ax1.get_xticklabels(): 
        label.set_fontproperties(font)

    # ax1.set_xlabel('')
    ax1.set_ylabel('QUANTITY')
    ax2.set_ylabel('RATIO')
    plt.grid(linestyle='--')
    plt.title('Overall', fontproperties=font, fontsize=12)
    plt.show()
