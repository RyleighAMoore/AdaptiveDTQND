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
from Errors import ErrorValsOneTime


problem = "hill" # "spiral" "complex"

driftFunction = functionBank.oneDrift
diffusionFunction = functionBank.ptSixDiffusion
spatialDiff = False

for dimension in [5,4,3]:
    if dimension == 1:
        beta = 4
        radius = 3
        kstepMin= 0.2
        kstepMax = kstepMin
        h = 0.05
        endTime =10
        useAdaptiveMesh =True

    if dimension == 2:
        kstepMin = 0.25
        kstepMax = 0.25
        endTime =5
        radius = 2
        h=0.05

    if dimension == 3:
        kstepMin = 0.22
        kstepMax = 0.22
        endTime = 1
        radius = 1
        h=0.02
        beta = 3


    if dimension == 4:
        kstepMin = 0.18
        kstepMax = 0.18
        endTime = 0.5
        radius = 0.8
        h=0.02
        beta = 3

    if dimension == 5:
        kstepMin = 0.1
        kstepMax = 0.1
        endTime = 0.04
        radius = 0.5
        h=0.01
        beta = 3



    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "EM", integratorType = "LQ", saveHistory=False)
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

    meshTrueSolnLQ = simulation.meshTrajectory[-1]
    pdfTrueSolnLQ = sde.exactSolution(simulation.meshTrajectory[-1],  simulation.times[-1])

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSolnLQ, pdfTrueSolnLQ, interpolate=False)

    print("dimension", dimension)
    print("L2w Error:", L2wErrors)
    print("Starting mesh size:", len(simulation.pdfTrajectory[0]))
    print("Ending mesh size:", len(simulation.pdfTrajectory[-1]))



