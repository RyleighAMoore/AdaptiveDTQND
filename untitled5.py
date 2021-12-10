# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 22:53:19 2021

@author: Rylei
"""

counter = {}
for n in range(1,51):
    counter[n] = 0

for n in range(1,51):
    x = 1
    while (x*n) <= 50:
        counter[x*n] += 1
        x += 1

[print(num) for num in counter if counter[num]%2 == 1]