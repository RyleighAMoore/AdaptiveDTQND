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

if dimension ==2:
    beta = 3
    radius =2
    kstepMin= 0.15
    kstepMax = 0.17
    h = 0.01
    endTime =3.5

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    # h = 0.01
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

for beta in [1.55, 3, 5]:
    start = time.time()
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType)
    simulationEM = Simulation(sde, parametersEM, endTime)
    startEM = time.time()
    spatialDiff = False
    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    simulation = Simulation(sde, parametersEM, endTime)
    simulation.computeAllTimes(sde, simulation.pdf, parametersEM)
    end = time.time()
    print(end-start)
    TimesEM.append(end-start)

    from exactSolutions import Solution

    trueSoln = []
    for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
        truepdf = Solution(1, 1, simulation.meshTrajectory[i], (i+1)*h, dimension)
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))

    from Errors import ErrorValsExact
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)
    ErrorsEM.append(L2wErrors[-1])


for kstep in [0.2, 0.15, 0.1]:#,0.12, 0.1]:
    start = time.time()
    radius = np.max(simulation.meshTrajectory[-1])
    parametersEMT = Parameters(sde, beta, radius+1, kstep, kstep, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="TR")
    simulationEMT = Simulation(sde, parametersEMT, endTime)

    startEM = time.time()
    spatialDiff = False
    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    simulation = Simulation(sde, parametersEMT, endTime)
    simulation.computeAllTimes(sde, simulation.pdf, parametersEMT)
    end = time.time()
    print(end-start)
    TimesTR.append(end-start)


    from exactSolutions import Solution

    trueSoln = []
    for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
        truepdf = Solution(1, 1, simulation.meshTrajectory[i], (i+1)*h, dimension)
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))

    from Errors import ErrorValsExact
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)
    ErrorsTR.append(L2wErrors[-1])


plt.figure()
plt.semilogx(np.asarray(ErrorsTR), np.asarray(TimesTR), label="TR")
plt.semilogx(np.asarray(ErrorsEM), np.asarray(TimesEM), label="EM")
plt.legend()
plt.show()
plt.save("timing.png")






