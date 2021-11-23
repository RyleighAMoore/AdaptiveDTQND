from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension =1
if dimension ==1:
    beta = 3
    radius =67/2
    kstepMin= 0.06
    kstepMax = 0.065
    h = 0.01
    endTime =4

if dimension ==2:
    beta = 3
    radius =1.5
    kstepMin= 0.08
    kstepMax = 0.09
    h = 0.01
    endTime = 0.5

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    # h = 0.01
    endTime = 0.1

# driftFunction = functionBank.zeroDrift
driftFunction = functionBank.erfDrift
# driftFunction = functionBank.oneDrift

spatialDiff = False

diffusionFunction = functionBank.oneDiffusion

adaptive = True
integrationType = "LQ"

ApproxSolution =False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)


parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType)
simulationEM = Simulation(sde, parametersEM, endTime)

parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType)
simulationAM = Simulation(sde, parametersAM, endTime)

iters = 1

radiusVals = [1, 10, 100]

timingEM = []
timingAM = []
numPointsArr = []
for radius in radiusVals:
    '''Time Transition Matrix'''
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType)
    simulationEM = Simulation(sde, parametersEM, endTime)

    startEM = time.time()
    for i in range(iters):
        simulationEM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersEM)
    endEM = time.time()
    endTime = endEM-startEM
    timingEM.append(endTime)


    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType)
    simulationAM = Simulation(sde, parametersAM, endTime)
    startAM = time.time()
    for i in range(iters):
        simulationAM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersAM)
    endAM = time.time()
    endTime = endAM-startAM
    timingAM.append(endTime)
    assert simulationAM.pdf.meshLength == simulationEM.pdf.meshLength
    numPointsArr.append(simulationAM.pdf.meshLength)



plt.figure()
plt.loglog(np.asarray(numPointsArr), np.asarray(timingEM), 'o', label="EM")
plt.loglog(np.asarray(numPointsArr), np.asarray(timingAM),'o', label = "AM")
plt.ylabel("Time")
plt.title("Timing of Transition Matrix Formation 1D")
plt.xlabel("Number of points")
plt.legend()












