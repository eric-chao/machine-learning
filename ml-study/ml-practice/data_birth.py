#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:22:36 2018

@author: adhoc-dev
"""

import requests as req

birth_data_url = 'http://www.umass.edu/statdata/data/lowbwt.dat'

birth_file = req.get(birth_data_url)
birth_data = birth_file.text.split('\r\n')
print(birth_data)