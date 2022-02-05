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
timeDiscretizationType = "EM"
integratorType = "LQ"

if problem == "hill":
    driftFunction = functionBank.oneDrift
    diffusionFunction = functionBank.oneDiffusion
    spatialDiff = False
    kstepMin = 0.25
    kstepMax = 0.3
    endTime = 0.5
    h=0.1

if problem == "erf":
    driftFunction = functionBank.erfDrift
    diffusionFunction = functionBank.pt75Diffusion
    spatialDiff = False
    kstepMin = 0.25
    kstepMax = 0.3
    endTime = 1#4
    h=0.05

if problem == "spiral":
    driftFunction = functionBank.spiralDrift_2D
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.15
    kstepMax = 0.15
    endTime = 3
    h=0.05

if problem == "complex":
    driftFunction = functionBank.complextDrift_2D
    diffusionFunction = functionBank.complexDiff
    spatialDiff = True
    kstepMin = 0.1
    kstepMax = 0.1
    endTime = 1.5
    h=0.01


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = timeDiscretizationType, integratorType = integratorType)
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
    ax.set_zlim(0,np.max(simulation.pdfTrajectory[2]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()

from PlottingResults import plotRowSixPlots

if problem == "erf":
    plottingMax = 0.3
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [5, 15,len(simulation.meshTrajectory)-1], [-10,10,-10,10])


if problem == "spiral":
    plottingMax = 0.3
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [15, 35 ,len(simulation.meshTrajectory)-1], [-12,12,-12,12])


if problem == "complex":
    plottingMax = 0.3
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [59, 99 ,len(simulation.meshTrajectory)-1], [-6,6,-6,6])


'''Compute Leja reuse and Alt method use'''
lengths = []
for mesh in simulation.meshTrajectory[1:]:
    lengths.append(len(mesh))

percentLejaReuse = np.asarray(simulation.LPReuseCount)/np.asarray(lengths)*100

print("Average LEJA REUSE Percent: ", np.mean(percentLejaReuse))

percentAltMethodUse = np.asarray(simulation.AltMethodUseCount)/np.asarray(lengths)*100
print("Average ALT METHOD USE Percent: ", np.mean(percentAltMethodUse))

from Errors import ErrorValsOneTime
if problem == "hill":
    meshTrueSoln = simulation.meshTrajectory[-1]
    pdfTrueSoln = sde.exactSolution(simulation.meshTrajectory[-1], endTime)
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSoln, pdfTrueSoln, interpolate=False)
    print(L2wErrors)




