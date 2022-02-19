import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time
from PlottingResults import plotRowSixPlots
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Class_Parameters import Parameters
from Class_SDE import SDE
from Class_Simulation import Simulation


problem = "erf" # "spiral" "complex" "hill"

dimension = 2
timeDiscretizationType = "EM"
integratorType = "LQ"


if problem == "hill":
    driftFunction = functionBank.oneDrift
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.2
    kstepMax = 0.2
    endTime =2
    radius = 2
    beta = 4
    h=0.05

if problem == "erf":
    driftFunction = functionBank.erfDrift
    diffusionFunction = functionBank.pt75Diffusion
    spatialDiff = False
    kstepMin = 0.25
    kstepMax = 0.25
    endTime = 4
    radius = 3
    beta = 3
    h=0.04

if problem == "spiral":
    driftFunction = functionBank.spiralDrift_2D
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.2
    kstepMax = 0.2
    endTime = 3
    radius = 2
    beta = 3
    h=0.02

if problem == "complex":
    driftFunction = functionBank.complextDrift_2D
    diffusionFunction = functionBank.complexDiff
    spatialDiff = True
    kstepMin = 0.1
    kstepMax = 0.1
    endTime = 1.5
    radius = 2
    beta = 3
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
    ax.set_zlim(0,np.max(simulation.pdfTrajectory[-20]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()

plottingMax = 1
from PlottingResults import plotRowSixPlots
if problem == "hill":
    # plottingMax = 1
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [5, 15,len(simulation.meshTrajectory)-1], [-12,12,-12,12])

if problem == "erf":
    # plottingMax = 1
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [9, 15,len(simulation.meshTrajectory)-1], [-12,12,-12,12])

if problem == "spiral":
    # plottingMax = 1
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [19, 59 , 120],[-10,10,-10,10])
    # plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [50, 85 ,len(simulation.meshTrajectory)-1],[-10,10,-10,10])

if problem == "complex":
    # plottingMax =1
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [29, 49 ,len(simulation.meshTrajectory)-1], [-6,6,-6,6])


print("Number of starting points: " + str(len(simulation.pdfTrajectory[0])))
print("Number of ending points: " + str(len(simulation.pdfTrajectory[-1])))
print("Starting range: [" + str(min(simulation.meshTrajectory[0][:,0])) + ", " + str(max(simulation.meshTrajectory[0][:,1])) + "]")
print("Ending range: [" + str(min(simulation.meshTrajectory[-1][:,0])) + ", " + str(max(simulation.meshTrajectory[-1][:,1])) + "]")



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



if kstepMax == kstepMin:
    volumes = []
    count = 0
    for pdf in simulation.pdfTrajectory:
        volumes.append(kstepMin**2*np.sum(simulation.pdfTrajectory[count]))
        count +=1


index = 58
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
graph, = ax.plot(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], linestyle="", marker="o")
# graph, = ax.plot(Meshes[index+1][:,0], Meshes[index+1][:,1], PdfTraj[index+1], linestyle="", marker=".")

ax.set_zlim(0,np.max(simulation.pdfTrajectory[2]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()

index = 58
fig = plt.figure()
plt.plot(Meshes[index][:,0], Meshes[index][:,1], linestyle="", marker="o")
# graph, = ax.plot(Meshes[index+1][:,0], Meshes[index+1][:,1], PdfTraj[index+1], linestyle="", marker=".")

ax.set_zlim(0,np.max(simulation.pdfTrajectory[2]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()