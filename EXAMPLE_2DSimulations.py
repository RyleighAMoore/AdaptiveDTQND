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


problem = "erf" # "spiral" "complex" "hill"
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
    h=0.02

if problem == "erf":
    driftFunction = functionBank.erfDrift
    diffusionFunction = functionBank.pt75Diffusion
    spatialDiff = False
    kstepMin = 0.2
    kstepMax = 0.22
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
    beta = 4
    h=0.02

if problem == "complex":
    driftFunction = functionBank.complextDrift_2D
    diffusionFunction = functionBank.complexDiff
    spatialDiff = True
    kstepMin = 0.2
    kstepMax = 0.2
    endTime = 1.5
    radius = 2
    beta = 4
    h=0.02


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


'''Compute Leja reuse and Alt method use'''
simulation.computeLejaAndAlternativeUse()
simulation.computeTotalPointsUsed()


endTime = simulation.times[-1]

hnew=0.01
hfactor = int(h/hnew)
assert hnew*hfactor == h
h = hnew
spacingTR = 0.05
if problem =="hill":
    meshTR = get2DTrapezoidalMeshBasedOnDefinedRange(-3, 3,-3, 3, spacingTR, 0)# '''erf'''

if problem =="erf":
    meshTR = get2DTrapezoidalMeshBasedOnDefinedRange(-14,14,-14, 14, spacingTR, 0)# '''erf'''

if problem =="spiral":
    meshTR = get2DTrapezoidalMeshBasedOnDefinedRange(-8,8,-8, 8, spacingTR, 0)# '''erf'''

if problem =="complex":
    meshTR = get2DTrapezoidalMeshBasedOnDefinedRange(-8,8,-8, 8, spacingTR, 0)# '''erf'''


parametersTR = Parameters(sde, beta, radius, spacingTR, spacingTR, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = meshTR, saveHistory=True)

simulationTR = Simulation(sde, parametersTR, endTime)
simulationTR.setUpTransitionMatrix(sde, parametersTR)

stepByStepTimingTR = simulationTR.computeAllTimes(sde, parametersTR)


meshTrajectoryLQ =  simulation.meshTrajectory
pdfLQ =  simulation.pdfTrajectory
meshTrajectoryTR = []
pdfTR = []
timesTR = []
for i in range(len(simulationTR.pdfTrajectory)):
    if (i+1)% hfactor ==0:
        meshTrajectoryTR.append(simulationTR.meshTrajectory[i])
        pdfTR.append(np.copy(simulationTR.pdfTrajectory[i]))
        timesTR.append(np.copy(simulationTR.times[i]))


plottingMax = 5
if problem == "hill":
    plotRowNinePlots(plottingMax, meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [5, 15,-1], [-12,12,-12,12], simulation.times)

if problem == "erf":
    plotRowNinePlots(plottingMax, meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [3, 29,-1], [-12,12,-12,12], simulation.times)

if problem == "spiral":
    plotRowNinePlots(plottingMax,meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [19, 69 ,-1],[-10,10,-10,10], simulation.times)

if problem == "complex":
    plotRowNinePlots(plottingMax,meshTrajectoryLQ, pdfLQ, meshTrajectoryTR, pdfTR, h, [14, 44 ,-1], [-8,8,-8,8], simulation.times)


assert timesTR == simulation.times
print(max(simulation.pdfTrajectory[19]))
print(max(pdfTR[19]))
