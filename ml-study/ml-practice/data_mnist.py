#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:32:44 2018

Download the data from http://yann.lecun.com/exdb/mnist/,
And put them under the folder MNIST_data.

@author: zxj

http://yann.lecun.com/exdb/mnist/

"""
# import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

# print(tf.transpose(mnist.train.images)[0])

print(len(mnist.train.images))

print(len(mnist.test.images))

print(len(mnist.validation.images))

print(mnist.train.labels[0])
print(mnist.train.labels[1])

print(mnist.train.labels[1, :])