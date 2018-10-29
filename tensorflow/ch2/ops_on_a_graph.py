#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:38:09 2018

@author: zxj

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()


# create graph
sess = tf.Session()

# create tensors

# create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)

# multiplication
my_product = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))
    
# View the tensorboard graph by running the following code and then
#    going to the terminal and typing:
#    $ tensorboard --logdir=tensorboard_logs
merged = tf.summary.merge_all()
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')
    
my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)

# close seesion
sess.close()