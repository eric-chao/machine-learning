#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

main 

@author: zxj

"""
import TFNaiveBayesClassifier as naive
import sklearn.metrics as mtx
import tensorflow as tf
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # parse data
    all_data = pd.read_csv('./naive_bayes/datasets/source/all_data.1.csv', header=0)
    # drop duplicated records
    cate_index = 5
    all_data = all_data.drop_duplicates() 
    all_vals = all_data[['PROVINCE','OPERATOR','INDUSTRY','OWNER','POST','CATEGORY']].values

    # split data info train/test = 80%/20%
    train_indices = np.random.choice(len(all_vals), round(len(all_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(all_vals))) - set(train_indices)))

    x_train, x_test = all_vals[train_indices], all_vals[test_indices]
    x_train_target = x_train[:, cate_index]
    x_train_signed, x_train_notsigned = x_train[x_train_target==0][:, :cate_index], x_train[x_train_target==1][:, :cate_index]
    x_test_features, x_test_target = x_test[:, :cate_index], x_test[:, cate_index]

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

    # create session
    sess = tf.Session()

    # predict 
    # with priors probabilities
    # 0 -- not signed, 1 -- signed
    unique, counts = np.unique(x_train_target, return_counts=True)
    x_train_target_dict = dict(zip(unique, counts))
    x_train_total = x_train_target_dict[0] + x_train_target_dict[1]
    priors_prob = [x_train_target_dict[0]/x_train_total, x_train_target_dict[1]/x_train_total]

    t_unique, t_counts = np.unique(x_test_target, return_counts=True)
    x_test_target_dict = dict(zip(t_unique, t_counts))
    x_test_total = x_test_target_dict[0] + x_test_target_dict[1]
    test_prob = [x_test_target_dict[0]/x_test_total, x_test_target_dict[1]/x_test_total]

    # print(sess.run([mean, var]))
    # print(sess.run([tf_nb.dist.loc, tf_nb.dist.scale]))
    Z = sess.run(tf_nb.predict(x_test_features, priors_prob))
    print(Z)

    # auc
    # fpr, tpr, thresholds = mtx.roc_curve(x_test_target, Z[:, 1], pos_label=2)
    auc = mtx.roc_auc_score(x_test_target, Z[:, 1])
    print(auc)
    # evaluate models
    # threshold = Z[:, 0] > Z[:, 1]
    # threshold_0, threshold_1 = Z[:, 0] > 0.6, Z[:, 1] > 0.85
    # output = np.ones((Z.shape[0], 1))
    # output[threshold_0] = np.zeros((output[threshold_0].shape[0], 1))
    # result = np.abs(x_test_target - output.flatten())
    # accuracy = (result.shape[0] - result.sum()) / result.shape[0]

    # print(priors_prob, x_train_target.shape)
    # print(test_prob, x_test_target.shape)
    # print(accuracy)

    # close session
    sess.close()