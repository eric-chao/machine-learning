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
    all_result = all_data.groupby(['QUOTIENT']).count().reset_index().set_index('QUOTIENT')
    all_result.rename(columns=({'PROB0': 'QUANTITY', 'PROB1': 'DENSITY'}), inplace=True)

    all_result
    # print
    print(all_result)

    # plot
    width = len(all_result.index)
    fig = plt.figure(figsize=[width * 0.8, 6])
    ax1 = all_result.QUANTITY.plot(kind='bar', color='xkcd:sky blue', legend=True)
    ax2 = all_result.QUANTITY.plot(kind='kde', secondary_y=True, color='xkcd:mango', legend=True)

    # ax1.set_xlabel('')
    ax1.set_ylabel('QUANTITY')
    ax2.set_ylabel('DENSITY')
    plt.grid(linestyle='--')
    plt.title('TEST', fontproperties=font, fontsize=12)
    plt.show()
