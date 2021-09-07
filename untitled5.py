# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:28:19 2021

@author: Rylei
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:16:26 2021

@author: Rylei
"""

from time import time

import numpy as np
rng = np.random.default_rng()
import Functions as fun

mesh = np.empty((100,100))

startTime = time()

for index, value in np.ndenumerate(mesh):
    newValue = rng.random()
    mesh[index] = newValue

print(f'writing one at a time takes {time()-startTime} seconds')

##################################

startTime = time()

valuesToAdd = []
for index, value in np.ndenumerate(mesh):
    newValue = rng.random()
    valuesToAdd.append(newValue)
valuesToAddReshaped = np.reshape(valuesToAdd, mesh.shape)
mesh[:] = valuesToAddReshaped

print(f'writing all at once takes {time()-startTime} seconds')