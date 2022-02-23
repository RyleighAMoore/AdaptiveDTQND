import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time
from PlottingResults import plotRowSixPlots, plotRowNinePlots
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Class_Parameters import Parameters
from Class_SDE import SDE
from Class_Simulation import Simulation
from Functions import get2DTrapezoidalMeshBasedOnLejaQuadratureSolutionMovingHill, get2DTrapezoidalMeshBasedOnDefinedRange
from Errors import ErrorValsOneTime


problem = "spiral" # "spiral" "complex" "hill"
approxError = False
dimension = 2
timeDiscretizationType = "EM"
integratorType = "LQ"

if problem == "hill":
    driftFunction = functionBank.oneDrift
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.2
    kstepMax = 0.2
    endTime =1
    radius = 2
    beta = 4
    h=0.05

if problem == "erf":
    driftFunction = functionBank.erfDrift
    diffusionFunction = functionBank.pt75Diffusion
    spatialDiff = False
    kstepMin = 0.25
    kstepMax = 0.3
    endTime = 4
    radius = 3
    beta = 4
    h=0.04

if problem == "spiral":
    driftFunction = functionBank.spiralDrift_2D
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.2
    kstepMax = 0.2
    endTime = 2.4
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


# '''Approximate Errror'''
# if approxError:
#     spacingTR = 0.05
#     h= 0.01
#     buffer=0.3
#     meshTR = get2DTrapezoidalMeshBasedOnLejaQuadratureSolutionMovingHill(simulation.meshTrajectory, spacingTR, bufferVal=buffer)
#     parametersTR = Parameters(sde, beta, radius, spacingTR, spacingTR, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = meshTR, saveHistory=False)

#     simulationTR = Simulation(sde, parametersTR, endTime)
#     startTimeTR = time.time()
#     simulationTR.setUpTransitionMatrix(sde, parametersTR)

#     stepByStepTimingTR = simulationTR.computeAllTimes(sde, parametersTR)
#     totalTimeTR = time.time() - startTimeTR

#     meshTrueSolnTR = simulationTR.meshTrajectory[-1]
#     pdfTrueSolnTR = simulationTR.pdfTrajectory[-1]
#     LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSolnTR, pdfTrueSolnTR, interpolate=True)

#     print(L2wErrors)

# Plot = True
# if Plot:
#     animate = True
#     if animate ==True:
#         Meshes = simulation.meshTrajectory
#         PdfTraj = simulation.pdfTrajectory
#         def update_graph(num):
#             graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
#             graph.set_3d_properties(PdfTraj[num])
#             title.set_text('3D Test, time={}'.format(num))
#             return title, graph

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         title = ax.set_title('3D Test')

#         graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
#         ax.set_zlim(0,np.max(simulation.pdfTrajectory[-20]))
#         ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
#         plt.show()

    # plottingMax = 5
#     from PlottingResults import plotRowSixPlots
#     from PlottingResults import plotRowSixPlots

#     if problem == "hill":
#         # plottingMax = 1
#         plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [5, 15,-1], [-12,12,-12,12], simulation.times)

#     if problem == "erf":
#         # plottingMax = 1
#         plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [3, 15,-1], [-14,14,-14,14], simulation.times)

#     if problem == "spiral":
#         # plottingMax = 1
#         plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [19, 59 ,-1],[-10,10,-10,10], simulation.times)
#         # plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [50, 85 ,len(simulation.meshTrajectory)-1],[-10,10,-10,10])

#     if problem == "complex":
#         # plottingMax =1
#         plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [29, 49 ,-1], [-6,6,-6,6], simulation.times)


    # print("Number of starting points: " + str(len(simulation.pdfTrajectory[0])))
    # print("Number of ending points: " + str(len(simulation.pdfTrajectory[-1])))
    # print("Starting range: [" + str(min(simulation.meshTrajectory[0][:,0])) + ", " + str(max(simulation.meshTrajectory[0][:,1])) + "]")
    # print("Ending range: [" + str(min(simulation.meshTrajectory[-1][:,0])) + ", " + str(max(simulation.meshTrajectory[-1][:,1])) + "]")



'''Compute Leja reuse and Alt method use'''
simulation.computeLejaAndAlternativeUse()
simulation.computeTotalPointsUsed()

if problem == "hill":
    meshTrueSoln = simulation.meshTrajectory[-1]
    pdfTrueSoln = sde.exactSolution(simulation.meshTrajectory[-1],  simulation.times[-1])
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSoln, pdfTrueSoln, interpolate=False)
    print(L2wErrors)



if kstepMax == kstepMin:
    volumes = []
    count = 0
    for pdf in simulation.pdfTrajectory:
        volumes.append(kstepMin**2*np.sum(simulation.pdfTrajectory[count]))
        count +=1



spacingTR = 0.1
endTime = simulation.times[-1]
range = [-10,10,-10,10]
meshTR = get2DTrapezoidalMeshBasedOnDefinedRange(-8,8,-8,8, spacingTR, 0)
parametersTR = Parameters(sde, beta, radius, spacingTR, spacingTR, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = meshTR, saveHistory=True)

simulationTR = Simulation(sde, parametersTR, endTime)
simulationTR.setUpTransitionMatrix(sde, parametersTR)

stepByStepTimingTR = simulationTR.computeAllTimes(sde, parametersTR)
simulationTR.meshUpdater.removePointsFromMeshProcedure(simulationTR.pdf, simulationTR, parameters, sde)
simulationTR.meshUpdater.removeOutlierPoints(simulationTR.pdf, simulationTR, parameters, sde)
simulationTR.pdfTrajectory[-1] =simulationTR.pdf.pdfVals
simulationTR.meshTrajectory[-1] = simulationTR.pdf.meshCoordinates

# meshTrueSolnTR = simulationTR.meshTrajectory[-1]
# pdfTrueSolnTR = simulationTR.pdfTrajectory[-1]
# LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSolnTR, pdfTrueSolnTR, interpolate=True)
# print(L2wErrors)
assert simulationTR.times[-1] == simulation.times[-1]

# plotRowNinePlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory,simulationTR.meshTrajectory, simulationTR.pdfTrajectory, h, [5, 15,-1], [-12,12,-12,12], simulation.times)

'''Spiral
Number of starting points: 325
Number of ending points: 2356
Starting range: [-2.0, 2.0]
Ending range: [-5.0000000000000036, 7.0000000000000036]
Average LEJA REUSE Percent:  83.03200089720583
Average ALT METHOD USE Percent:  0.11721058387151188
Number of Points Used:  134184
Average of Points Used Per Time Step:  1118.2
'''

Meshes = simulationTR.meshTrajectory
PdfTraj = simulationTR.pdfTrajectory
def update_graph(num):
    graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
    graph.set_3d_properties(PdfTraj[num])
    title.set_text('3D Test, time={}'.format(num))
    return title, graph


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
ax.set_zlim(0,np.max(simulation.pdfTrajectory[20]))
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()


# timestr = time.strftime("%Y%m%d-%H%M%S")
# ListToSave = [simulation.meshTrajectory, simulation.pdfTrajectory,simulationTR.meshTrajectory, simulationTR.pdfTrajectory]
# import pickle

# # define dictionary
# # create a binary pickle file
# f = open('Output/TwoDSimulation_'+str(timestr)+ "_" + str(problem)+ '.pkl',"wb")
# pickle.dump(ListToSave,f)
# f.close()

# readIn = True

# if readIn:
#     objects = []
#     with (open("Output//TwoDSimulation_20220223-102054_spiral.pkl", "rb")) as openfile:
#         while True:
#             try:
#                 objects.append(pickle.load(openfile))
#             except EOFError:
#                 break

#     meshTrajectoryLQ =  objects[0][0]
#     pdfLQ =  objects[0][1]
#     mesTrajectoryTR =  objects[0][2]
#     pdfTR =  objects[0][3]

# else:
meshTrajectoryLQ =  simulation.meshTrajectory
pdfLQ =  simulation.pdfTrajectory
meshTrajectoryTR =  simulationTR.meshTrajectory
pdfTR =  simulation.pdfTrajectory

plottingMax = 5
if problem == "hill":
        # plottingMax = 1
       plotRowNinePlots(plottingMax, meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [5, 15,-1], [-12,12,-12,12], simulation.times)

if problem == "erf":
    # plottingMax = 1
    plotRowNinePlots(plottingMax, meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [3, 15,-1], [-14,14,-14,14], simulation.times)

if problem == "spiral":
    # plottingMax = 1
    plotRowNinePlots(plottingMax,meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [19, 49 ,-1],[-10,10,-10,10], simulation.times)
    # plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [50, 85 ,len(simulation.meshTrajectory)-1],[-10,10,-10,10])

if problem == "complex":
    # plottingMax =1
    plotRowNinePlots(plottingMax,meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [29, 49 ,-1], [-6,6,-6,6], simulation.times)
