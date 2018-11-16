#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

main 

@author: zxj

"""
from sklearn.model_selection import KFold
import TFNaiveBayesClassifier as naive
import sklearn.metrics as mtx
import tensorflow as tf
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # parse data
    all_data = pd.read_csv('./naive_bayes/datasets/source/all_data.csv', header=0)
    # drop duplicated records
    all_data = all_data.drop_duplicates() 
    all_vals = all_data[['PROVINCE','OPERATOR','CLUE','INDUSTRY','OWNER','CATEGORY']].values

    # create session
    sess = tf.Session()

    kf = KFold(n_splits=10, shuffle=True) 
    for train_indices, test_indices in kf.split(all_vals):  
        x_train, x_test = all_vals[train_indices], all_vals[test_indices]
        x_train_target = x_train[:, 5]
        x_train_signed, x_train_notsigned = x_train[x_train_target==0][:, :5], x_train[x_train_target==1][:, :5]
        x_test_features, x_test_target = x_test[:, :5], x_test[:, 5]

        # get the univariate's mean and variance
        mean_signed, var_signed = tf.nn.moments(tf.constant(x_train_signed), axes=[0])
        mean_notsigned, var_notsigned = tf.nn.moments(tf.constant(x_train_notsigned), axes=[0])

        # use tf.concat(), numpy.append() or numpy.concatenate()
        # to merge (mean_signed, mean_notsigned) and (var_signed, var_notsigned)
        mean = tf.concat([[mean_signed], [mean_notsigned]], 0)
        var =  tf.concat([[var_signed], [var_notsigned]], 0)

        # fit
        # Create a [classes x features] univariate normal distribution 
        # with the known mean and variance
        tf_nb = naive.TFNaiveBayesClassifier()
        tf_nb.dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))

        # predict 
        # with priors probabilities
        # 0 -- not signed, 1 -- signed
        unique, counts = np.unique(x_train_target, return_counts=True)
        x_train_target_dict = dict(zip(unique, counts))
        x_train_total = x_train_target_dict[0] + x_train_target_dict[1]
        priors_prob = [x_train_target_dict[0]/x_train_total, x_train_target_dict[1]/x_train_total]

        # print(sess.run([mean, var]))
        # print(sess.run([tf_nb.dist.loc, tf_nb.dist.scale]))
        Z = sess.run(tf_nb.predict(x_test_features, priors_prob))
        # print(Z)

        # use auc to evaluate models
        # fpr, tpr, thresholds = mtx.roc_curve(x_test_target, Z[:, 1], pos_label=2)
        auc = mtx.roc_auc_score(x_test_target, Z[:, 1])
        print(auc)

    # close session
    sess.close()