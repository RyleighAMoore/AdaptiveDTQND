from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt


dimension = 1
beta = 3
radius = 3.75
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
simulation = Simulation(sde, parameters, endTime)
plt.scatter(simulation.pdf.meshCoordinates,simulation.pdf.pdfVals)
plt.scatter(simulation.meshTrajectory[-1],simulation.pdfTrajectory[-1])


import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_graph(num):
    graph.set_data(simulation.meshTrajectory[num], simulation.pdfTrajectory[num])
    return title, graph

fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title('2D Test')

graph, = ax.plot(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], linestyle="", marker=".")
ax.set_xlim(-20, 20)
ax.set_ylim(0, np.max(simulation.pdfTrajectory[0]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(simulation.pdfTrajectory), interval=1000, blit=False)
plt.show()

from exactSolutions import OneDdiffusionEquation

trueSoln = []
for i in range(len(simulation.meshTrajectory)):
    truepdf = OneDdiffusionEquation(simulation.meshTrajectory[i], sde.diffusionFunction(simulation.meshTrajectory[i]), (i+1)*h, sde.driftFunction(simulation.meshTrajectory[i]))
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))

from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)


