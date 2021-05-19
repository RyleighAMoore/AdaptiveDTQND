# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:40:20 2021

@author: Rylei
"""
mesh = Meshes[-1]
for i in range(len(Meshes[-1])):
    for j in range(len(Meshes[-1])):
        if (mesh[i,0] == mesh[j,0]) and (mesh[i,1] == mesh[j,1]) and (i != j):
            print(i)
            print(j)
        