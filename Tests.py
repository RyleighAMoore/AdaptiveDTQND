# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:46:57 2021

@author: Rylei
"""
from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt


dimension = 1
beta = 3
radius = 3
kstepMin= 0.06
kstepMax = 0.07
h = 0.01
endTime = 1

def driftFunction(mesh):
      if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return -1*mesh
      return np.zeros(np.shape(mesh))

def diffusionFunction(mesh):
    return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "EM")

from QuadraticFit import LaplaceApproximation
def testQuadraticFit():
    la = LaplaceApproximation(sde)
    QuadMesh = simulation.pdf.meshCoordinates
    laplaceFitPdf = simulation.pdf.meshCoordinates

    la.copmuteleastSquares(QuadMesh, laplaceFitPdf, simulation.pdf, sde, parameters)
    vals = la.ComputeDividedOut(simulation.pdf, sde)
    plt.scatter(simulation.pdf.meshCoordinates, vals)


testQuadraticFit()




