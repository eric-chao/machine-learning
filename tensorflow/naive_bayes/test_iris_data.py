#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:26:39 2018

Function test.

@author: zxj

"""

from sklearn import datasets
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # open session
    sess = tf.Session()
    
    iris = datasets.load_iris()
    # Only take the first two features
    X = iris.data[:, :2]
    y = iris.target

    unique_y = np.unique(y)
    points_by_class = np.array([[x for x, t in zip(X, y) if t == c] for c in unique_y])
    # print(points_by_class)
    
    # Estimate mean and variance for each class / feature
    # shape: nb_classes * nb_features
    mean, var = tf.nn.moments(tf.constant(points_by_class), axes=[1])
    print(sess.run(mean))
    print(sess.run(var))
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    
    print(X[:, 0].min(), X[:, 0].max())
    print(X[:, 1].min(), X[:, 1].max())
    print(x_min, x_max)
    print(y_min, y_max)
    
    # linspace 线性等分向量
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
    print(len(xx), len(yy))
    # print(xx)
    # print(yy)
    
    # close session
    sess.close()

