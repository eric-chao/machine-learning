#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:33:22 2018

basic test

@author: zxj

"""

import numpy as np

# test 1
data = [0, 1, 2, 3, 4, 5]
x_data = [ x for x in data if x >= 1]
print(x_data)

# test 2
str_data = ['a a', 'b b', 'c c', 'd d', 'e e']
str = [ s.split(' ') for s in str_data ]
print(str)

# test zip
cate_data = [[1, 0], [2, 0], [3, 0], [1, 1], [2, 1], [3, 1], [1, 2], [3, 2], [2, 2]]
catelog = [0, 0, 0, 1, 1, 1, 2, 2, 2]

for x, t in zip(cate_data, catelog):
    print(x, t)

print(np.unique(catelog))
result = np.array([[x for x, t in zip(cate_data, catelog) if t == c] for c in np.unique(catelog)])
print(result)