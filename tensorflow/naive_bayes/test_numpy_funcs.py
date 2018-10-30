#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:35:33 2018

numpy test

@author: zxj

"""

import numpy as np

nx, ny = (3, 2)

# linspace 线性等分向量
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
print(x, y)

xv, yv = np.meshgrid(x, y)
print(xv, yv)

print(np.c_[xv.ravel(), yv.ravel()])