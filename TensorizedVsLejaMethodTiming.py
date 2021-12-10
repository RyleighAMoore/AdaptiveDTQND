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
    beta = 4
    radius = 4
    kstepMin= 0.1
    kstepMax = 0.12
    h = 0.1
    endTime =0

if dimension ==2:
    radius =1.5
    kstepMin= 0.08
    kstepMax = 0.09
    kstepMin= 0.1
    kstepMax = 0.12
    h = 0.01
    endTime = 1

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01
    endTime = 0.1


# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift

spatialDiff = True

diffusionFunction = functionBank.oneDiffusion

adaptive = True
integrationType = "LQ"

ApproxSolution =False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)




iters = 3



ErrorsEM = []
ErrorsTR = []
TimesEM = []
TimesTR = []

for beta in [3, 5, 8]:
    start = time.time()
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType)
    simulationEM = Simulation(sde, parametersEM, endTime)
    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
    end = time.time()
    print(end-start)
    TimesEM.append(end-start)

    from exactSolutions import Solution

    trueSoln = []
    for i in range(len(simulationEM.meshTrajectory)): #diff, drift, mesh, t, dim
        truepdf = Solution(1, 1, simulationEM.meshTrajectory[i], (i+1)*h, dimension)
        trueSoln.append(np.squeeze(np.copy(truepdf)))

    from Errors import ErrorValsExact
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulationEM.meshTrajectory, simulationEM.pdfTrajectory, trueSoln, h, plot=False)
    ErrorsEM.append(L2wErrors[-1])


for kstep in [0.4, 0.3,0.2]:#,0.12, 0.1]:
    start = time.time()
    radius = np.max(simulationEM.meshTrajectory[-1])
    parametersEMT = Parameters(sde, beta, radius+1, kstep, kstep, h, useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR")
    simulationEMT = Simulation(sde, parametersEMT, endTime)
    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    simulationEMT.computeAllTimes(sde, simulationEMT.pdf, parametersEMT)
    end = time.time()
    print(end-start)
    TimesTR.append(end-start)

    trueSoln = []
    for i in range(len(simulationEMT.meshTrajectory)): #diff, drift, mesh, t, dim
        truepdf = Solution(1, 1, simulationEMT.meshTrajectory[i], (i+1)*h, dimension)
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulationEMT.meshTrajectory, simulationEMT.pdfTrajectory, trueSoln, h, plot=False)
    ErrorsTR.append(L2wErrors[-1])


plt.figure()
plt.semilogx(np.asarray(ErrorsTR), np.asarray(TimesTR), label="TR")
plt.semilogx(np.asarray(ErrorsEM), np.asarray(TimesEM), label="EM")
plt.legend()
plt.show()
# plt.save("timing.png")






