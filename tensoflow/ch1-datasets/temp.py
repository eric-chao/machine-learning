# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

with tf.Session() as sess:
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    
    result = sess.run([product])
    print(result)
    print(result[0][0][0])
    

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result[0])
    print(result[1])
    

in1 = tf.placeholder(tf.float32)
in2 = tf.placeholder(tf.float32)
output = tf.multiply(in1, in2)
with tf.Session() as sess:
    result = sess.run([output], feed_dict={in1:[7.], in2:[2.]})
    print(result)
    print(result[0])
    

x = -1
with tf.control_dependencies([tf.assert_negative(x)]):
   output = tf.reduce_sum(x)
   print(output)