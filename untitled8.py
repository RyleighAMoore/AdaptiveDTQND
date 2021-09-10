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
h = 0.1

def driftFunction(mesh):
      if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return -1*mesh
      return np.ones(np.shape(mesh))

def diffusionFunction(mesh):
    return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "EM")
pdf = PDF(sde, parameters)
simulation = Simulation(sde, parameters, pdf)
pdf.plot()
plt.scatter(simulation.meshTrajectory[-1],simulation.pdfTrajectory[-1])





