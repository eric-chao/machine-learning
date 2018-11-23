#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

sklearn naive bayes

sklearn.metrics.accuracy_score
sklearn.metrics.recall_score
sklearn.metrics.roc_auc_score

@author: zxj

"""
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import KFold
import sklearn.metrics as mtx
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # parse data
    all_data = pd.read_csv('./naive_bayes/datasets/source/all_data.1.csv', header=0)
    # drop duplicated records
    cate_index = 6
    all_data = all_data.drop_duplicates()
    # all_data.to_csv('high_auc_all.csv', float_format='%.3f', encoding='UTF-8')
    all_vals = all_data[['PROVINCE','OPERATOR','CLUE','INDUSTRY','OWNER','POST','CATEGORY']].values

    kf = KFold(n_splits=10, shuffle=True) 
    for train_indices, test_indices in kf.split(all_vals):
        x_train, x_test = all_vals[train_indices], all_vals[test_indices]
        x_train_features, x_train_target = x_train[:, :cate_index], x_train[:, cate_index]
        x_test_features, x_test_target = x_test[:, :cate_index], x_test[:, cate_index]
        
        # fit
        # nb = MultinomialNB()
        # nb = BernoulliNB()
        nb = GaussianNB()
        nb.fit(x_train_features, x_train_target)

        # predict
        # predicted = nb.predict(x_test_features)
        # auc = mtx.roc_auc_score(x_test_target, predicted)
        # print('精度为：%f ' % np.mean(predicted == x_test_target))
        pred_prob = nb.predict_proba(x_test_features)
        auc = mtx.roc_auc_score(x_test_target, pred_prob[:, 1])

        # if auc > 0.85:
        #     pd.DataFrame(pred_prob).to_csv('high_auc_test_prob.csv', float_format='%.3f', encoding='UTF-8')
        #     pd.DataFrame(x_test).to_csv('high_auc_test.csv', float_format='%.3f', encoding='UTF-8')
        #     print(test_indices)

        print(auc)
