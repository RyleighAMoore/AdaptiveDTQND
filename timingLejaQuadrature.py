from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension = 2
if dimension ==1:
    beta = 5
    radius = 3
    # kstepMin= 0.08
    # kstepMax = 0.09
    kstepMin= 0.15
    kstepMax = 0.2
    h = 0.01

if dimension ==2:
    beta = 3
    radius =2
    # radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.09
    kstepMin= 0.13
    kstepMax = 0.15
    h = 0.05

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift

spatialDiff = False


diffusionFunction = functionBank.ptSixDiffusion


adaptive = True

ApproxSolution =False

sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)

ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
betaVals = [2, 3]
# betaVals = [2.5]
# radiusVals = [1, 4]
# spacingVals = [0.08, 0.05]
spacingVals = [0.2, 0.1, 0.05]

# times = [4,8,10]
# times = [12]
times= [12]
endTime = 0
numPoints = []

for beta in betaVals:
    for spacing in spacingVals:
        parametersEM = Parameters(sde, beta, radius, spacing, spacing+0.2, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ")
        startEM = time.time()
        simulationEM = Simulation(sde, parametersEM, endTime)
        simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
        endEM = time.time()
        timesEM.append(np.copy(endEM-startEM))

        meshApprox = simulationEM.meshTrajectory[-1]
        pdfApprox = sde.exactSolution(simulationEM.meshTrajectory[-1], endTime)

        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
        ErrorsEM.append(np.copy(L2wErrors))
        numPoints.append(np.copy(simulationEM.pdf.meshLength))



plt.figure()
simulation = simulationEM
Meshes = simulation.meshTrajectory
plt.loglog(np.asarray(numPoints),np.asarray(timesEM), 'or')
plt.ylabel("time seconds")
plt.xlabel("Number of Points")






