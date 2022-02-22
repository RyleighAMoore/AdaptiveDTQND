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
from Functions import get2DTrapezoidalMeshBasedOnLejaQuadratureSolutionMovingHill
from Errors import ErrorValsOneTime

mainFig = plt.figure()
problems = ["hill", "erf", "spiral", "complex"]
# problems = ["spiral"]
AllErrors = []
for problem in problems:
    approxError = False
    dimension = 2
    timeDiscretizationType = "EM"
    integratorType = "TR"
    radius = 4.5
    endTime = 1.2
    beta = 3
    h=0.05

    if problem == "hill":
        driftFunction = functionBank.oneDrift
        diffusionFunction = functionBank.ptSixDiffusion
        spatialDiff = False


    if problem == "erf":
        driftFunction = functionBank.erfDrift
        diffusionFunction = functionBank.pt75Diffusion
        spatialDiff = False


    if problem == "spiral":
        driftFunction = functionBank.spiralDrift_2D
        diffusionFunction = functionBank.ptSixDiffusion
        spatialDiff = False


    if problem == "complex":
        driftFunction = functionBank.complextDrift_2D
        diffusionFunction = functionBank.complexDiff
        spatialDiff = True


    times = [0.05, 0.1, 0.15, 0.2, 0.4, 0.6]
    times = [0.03,0.04, 0.05, 0.1, 0.2, 0.4, 0.6]
    Errors = []
    for h in times:
        kstepMin = 0.07
        kstepMax = kstepMin
        sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
        parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =False, timeDiscretizationType = timeDiscretizationType, integratorType = integratorType, saveHistory=False)
        simulation = Simulation(sde, parameters, endTime)
        simulation.setUpTransitionMatrix(sde, parameters)

        simulation.computeAllTimes(sde, parameters)

        if h ==times[0]:
            simulationSolution = simulation
        else:
            meshTrueSolnTR = simulationSolution.meshTrajectory[-1]
            pdfTrueSolnTR = simulationSolution.pdfTrajectory[-1]
            LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSolnTR, pdfTrueSolnTR, interpolate=True)
            Errors.append(L1Errors)
            print(L2wErrors)


        # Meshes = simulation.meshTrajectory
        # PdfTraj = simulation.pdfTrajectory
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # title = ax.set_title('3D Test')

        # graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
        # plt.show()




    AllErrors.append(Errors)
    plt.loglog(times[1:], Errors, label = problem)


plt.legend()
plt.xlabel("Temporal Step Size")
plt.ylabel("Error")
plt.show()


