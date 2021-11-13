from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension =2
if dimension ==1:
    beta = 3
    radius =20
    kstepMin= 0.06
    kstepMax = 0.065
    h = 0.01
    endTime =4

if dimension == 2:
    beta = 5
    radius =1.5
    kstepMin= 0.08
    kstepMax = 0.09
    h = 0.05
    endTime = 0.3

if dimension == 3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    # h = 0.01
    endTime = 0.1

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift

spatialDiff = False

diffusionFunction = functionBank.oneDiffusion

adaptive = True
integrationType = "LQ"

ApproxSolution = False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)


'''Time Adding New points'''
spacingVals =  [0.1]
spacingVals = [0.01, 0.05, 0.1]


timingEM = []
timingAM = []
lengths = []
Errors = []

from tqdm import trange
from Errors import ErrorValsExact
from exactSolutions import Solution

h=0.01
for AMMeshSpacing in spacingVals:
    parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType,  AMSpacing = AMMeshSpacing)
    start = time.time()
    simulation = Simulation(sde, parameters, endTime)
    simulation.computeAllTimes(sde, simulation.pdf, parameters)
    end = time.time()
    timingAM.append(end-start)

    meshApprox = simulation.pdfTrajectory[-1]
    pdfApprox = sde.exactSolution(simulation.meshTrajectory[-1], endTime)
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
    Errors.append(np.copy(L2wErrors))




plt.figure()
plt.loglog(np.asarray(spacingVals), np.asarray(Errors), 'o')
plt.ylabel("Error")
plt.title("Errors vs. h=0.01")
plt.xlabel("AM spacing")


plt.figure()
plt.loglog(np.asarray(spacingVals), np.asarray(timingAM), 'o')
plt.ylabel("time")
plt.title("Time vs. h=0.01")
plt.xlabel("AM spacing")




