#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

main 

@author: zxj

"""

from sklearn.model_selection import KFold
from collections import Counter
import sklearn.metrics as mtx
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # parse data
    all_data = pd.read_csv('./naive_bayes/datasets/source/all_data.1.csv', header=0)
    # drop duplicated records
    all_data = all_data.drop_duplicates()
    all_data.to_csv('all_data_rm_duplicated.csv', float_format='%.3f', encoding='UTF-8')
    all_vals = all_data[['PROVINCE','OPERATOR','CLUE','INDUSTRY','OWNER','CATEGORY']].values

    kf = KFold(n_splits=2, shuffle=True) 
    for train_indices, test_indices in kf.split(all_vals):
        x_train, x_test = all_vals[train_indices], all_vals[test_indices]
        x_train_target = x_train[:, 5]
        x_train_notsigned, x_train_signed = x_train[x_train_target==0][:, :5], x_train[x_train_target==1][:, :5]
        x_test_features, x_test_target = x_test[:, :5], x_test[:, 5]

        cond_prob = np.zeros_like(x_train)
        # 0 -- not signed, 1 -- signed
        x_train_total_0, x_train_total_1 = np.bincount(x_train_target.astype(int))

        print(Counter(x_train_notsigned[:, 0]))