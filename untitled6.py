# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:48:53 2021

@author: Rylei
"""

import numpy as np
from time import time
rng = np.random.default_rng()

def squared(valToSquare):
    return valToSquare**2

size = 10000000
values = rng.random(size)

startTime = time()
result1 = np.empty(size)
for index, val in np.ndenumerate(values):
    result1[index] = squared(val)
print(f'time in a double loop is {time()-startTime} seconds')

startTime = time()
result2 = np.asarray([*map(squared, values)])
print(f'time using a map is {time()-startTime} seconds')
