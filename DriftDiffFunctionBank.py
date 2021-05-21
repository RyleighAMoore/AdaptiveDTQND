# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:21:51 2021

@author: Rylei
"""
import numpy as np
from scipy.special import erf

def SpiralDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    y = mesh[:,1]
    r = np.sqrt(x ** 2 + y ** 2)
    return np.asarray([3*(10*erf(10*x)+5*y)/(r+10), 6*(-2*x+y)/(r+10)]).T

def MovingHillDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.asarray([np.ones((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T

def VolcanoDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    y = mesh[:,1]
    r = np.sqrt(x ** 2 + y ** 2)
    return np.asarray([5*x*(3- r ** 2), 5*y*(3- r ** 2)]).T

def FourHillDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    y = mesh[:,1]
    return np.asarray([3*erf(10*x), 3*erf(10*y)]).T

def TwoHillDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    return np.asarray([2*erf(10*x), np.zeros((np.size(mesh,0)))]).T

def DiagDiffptThree(mesh):
    return np.diag([0.3,0.3])

def DiagDiffptSix(mesh):
    return np.diag([0.6,0.6])

def DiagDiffptSevenFive(mesh):
    return np.diag([0.75,0.75])

def DiagDiffOne(mesh):
    return np.diag([1,1])

def ComplexDiff(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.diag([0.01*mesh[:,0][0]**2+0.5,0.01*mesh[:,1][0]**2+0.5]) + np.ones((2,2))*0.2


# def drift(mesh):
#     if mesh.ndim ==1:
#         mesh = np.expand_dims(mesh, axis=0)
#     x = mesh[:,0]
#     y = mesh[:,1]
#     r = np.sqrt(x ** 2 + y ** 2)
#     # return np.asarray([x**2/2-y*x, x*y+y**2/2]).T
#     # return np.asarray([x-y,x+y]).T
#     # return np.asarray([np.ones((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
#     # return np.asarray([5*x*(3- r ** 2), 5*y*(3- r ** 2)]).T
#     # return np.asarray([3*erf(10*x), 3*erf(10*y)]).T
#     # return np.asarray([np.ones((np.size(mesh,0))), 5*np.ones((np.size(mesh,0)))]).T
#     # return np.asarray([-2*np.ones((np.size(mesh,0))), 2*np.ones((np.size(mesh,0)))]).T
#     # return np.asarray([2*erf(10*x), np.zeros((np.size(mesh,0)))]).T
#     return np.asarray([3*(10*erf(10*x)+5*y)/(r+10), 6*(-2*x+y)/(r+10)]).T



# def diff(mesh):
#     if mesh.ndim == 1:
#         mesh = np.expand_dims(mesh, axis=0)
#     # return np.diag([1,1])
#     # return np.diag([0.01*mesh[:,0][0]**2+0.5,0.01*mesh[:,1][0]**2+0.5]) + np.ones((2,2))*0.2
#     # return np.diag([mesh[:,0][0],mesh[:,1][0]])
#     return np.diag([.6,.6])

#     # return np.diag([0.75,0.75])
#     # return [[mesh[:,0][0]**2,1],[1, mesh[:,1][0]**2]]