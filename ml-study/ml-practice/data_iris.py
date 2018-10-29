#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:13:33 2018

@author: zxj

http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

"""
from sklearn import datasets

iris = datasets.load_iris()
# print(iris.data)
iris_data = iris.data
print(iris_data[0])

iris_target = iris.target
print(iris_target)
print(set(iris_target))

data_size = len(iris.data)
print(data_size)

target_size = len(iris.target)
print(target_size)

