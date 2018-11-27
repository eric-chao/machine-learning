#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

use
    sklearn.metrics.precision_recall_curve
to get precision, recall and threshold

then, use the parameters to plot.

@author: zxj

"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import sklearn.metrics as mtx
import pandas as pd
import numpy as np

font = FontProperties(fname=r"./naive_bayes/datasets/font/PingFang.ttc", size=9)

if __name__ == "__main__":
    # parse data
    all_data = pd.read_csv('./naive_bayes/datasets/source/high_auc_test_prob.csv', header=0, index_col=['NO'])

    print(all_data.sort_values(by=['PROB0']))
    y_true, probas_pred = all_data['TARGET'], all_data['PROB1']
    precision, recall, threshold = mtx.precision_recall_curve(y_true, probas_pred)
    precision, recall, threshold = pd.Series(precision), pd.Series(recall), pd.Series(threshold)

    # print
    # print(precision)
    # print(recall)
    # print(threshold)

    # union
    width = len(threshold)
    # threshold = threshold.append(pd.Series(threshold[width - 1] + 0.1))
    all_keys = ['PRECISION', 'RECALL', 'THRESHOLD']
    all_union = pd.concat([precision, recall, threshold], axis=1, keys=all_keys)
    all_union = all_union.fillna(threshold[width-1] + 0.1).astype(float)
    all_union = all_union.set_index('THRESHOLD')
    print(all_union)

    # plot
    fig = plt.figure(figsize=[width, 6])
    # all_union.plot(use_index=True)
    ax1 = all_union.PRECISION.plot(color='xkcd:sky blue', legend=True)
    ax2 = all_union.RECALL.plot(secondary_y=True, color='xkcd:mango', legend=True)
    ax1.set_ylabel('PRECISION', fontproperties=font)
    ax2.set_ylabel('RECALL', fontproperties=font)
    plt.grid(linestyle='--')
    plt.show()
