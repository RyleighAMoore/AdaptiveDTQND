# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:14:41 2020

@author: Rylei
"""
from pyopoly1 import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
from pyopoly1 import opolynd
from mpl_toolkits.mplot3d import Axes3D
from pyopoly1.families import HermitePolynomials
import pyopoly1.indexing
from pyopoly1 import QuadratureRules as QR
from Functions import *
import UnorderedMesh as UM


from pyopoly1.Class_Gaussian import GaussScale

def productGaussians2D(scaling,scaling2):
    
    sigmaNew = np.linalg.inv(np.linalg.inv(scaling.cov)+ np.linalg.inv(scaling2.cov))     
    muNew = np.matmul(np.linalg.inv(np.linalg.inv(scaling.cov) + np.linalg.inv(scaling2.cov)), np.matmul(np.linalg.inv(scaling.cov),scaling.mu) + np.matmul(np.linalg.inv(scaling2.cov),scaling2.mu))
    
    c = 1/(np.sqrt(np.linalg.det(2*np.pi*(scaling.cov+scaling2.cov))))
    cc = np.matmul(np.matmul(-(1/2)*(scaling.mu-scaling2.mu).T, np.linalg.inv(scaling.cov+scaling2.cov)),(scaling.mu-scaling2.mu))
    cfinal = c*np.exp(cc)
    
    scale = GaussScale(len(muNew))
    scale.setMu(muNew)
    scale.setCov(sigmaNew)
    
    return scale, cfinal[0][0]


def PlotG(Px, Py, h):
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, 0.05, includeOrigin=True)
    G = GVals(Px,Py,mesh, h)
    # N = Gaussian(Px, Py, np.sqrt(h)*g1(), np.sqrt(h)*g2(), mesh)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], G,  c='k', marker='o')
    # ax.scatter(mesh[:,0], mesh[:,1], N,  c='r', marker='.')
    plt.show()
    
# PlotG(0,0,0.01)

def PlotH(Px, Py, h):
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, 0.05, includeOrigin=True)
    H = HVals(Px,Py,mesh, h)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], H,  c='k', marker='o')
    plt.show()
    
# PlotH(0,0,0.01)

def PlotGH(Px, Py, h):
    mesh = UM.generateOrderedGridCenteredAtZero(-0.5, 0.5, -0.5, 0.5, 0.05, includeOrigin=True)
    H = HVals(Px,Py,mesh, h)
    G = GVals(Px,Py,mesh, h)
    Normal = Gaussian(Px, Py, np.sqrt(h)*g1(), np.sqrt(h)*g2(), mesh)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mesh[:,0], mesh[:,1], H*Normal,  c='k', marker='o')
    ax.scatter(mesh[:,0], mesh[:,1], G,  c='r', marker='.')
    plt.show()
