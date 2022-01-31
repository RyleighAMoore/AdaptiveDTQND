from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time
from PlottingResults import plotRowSixPlots
import matplotlib.pyplot as plt
import matplotlib.animation as animation

problem = "erf" # "spiral" "complex"

dimension =2
beta = 3
radius = 2
kstepMin = 0.25
kstepMax = 0.3
h = 0.05

if problem == "erf":
    driftFunction = functionBank.erfDrift
    diffusionFunction = functionBank.pt75Diffusion
    spatialDiff = False
    endTime = 4

if problem == "spiral":
    driftFunction = functionBank.spiral
    diffusionFunction = functionBank.pt6Diffusion
    spatialDiff = False
    endTime = 5


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "EM", integratorType = "LQ")
simulation = Simulation(sde, parameters, endTime)

start = time.time()
simulation.setUpTransitionMatrix(sde, parameters)
TMTime = time.time()-start

start = time.time()
simulation.computeAllTimes(sde, parameters)
end = time.time()
print("\n")
print("Transition Matrix timing:", TMTime)
print("\n")
print("Stepping timing",end-start, '*****************************************')


animate = True
if animate ==True:
    Meshes = simulation.meshTrajectory
    PdfTraj = simulation.pdfTrajectory
    def update_graph(num):
        graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
        graph.set_3d_properties(PdfTraj[num])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
    ax.set_zlim(0,np.max(simulation.pdfTrajectory[10]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()

from PlottingResults import plotRowSixPlots

if problem == "erf":
    plottingMax = 0.1
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [15, (len(simulation.meshTrajectory)-1)//2,len(simulation.meshTrajectory)-1])


if problem == "spiral":
    plottingMax = 0.1
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [15, (len(simulation.meshTrajectory)-1)//2,len(simulation.meshTrajectory)-1])

