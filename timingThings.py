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
    radius =1.5
    kstepMin= 0.08
    kstepMax = 0.09
    h = 0.05
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

parametersEMT = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="TR")
simulationEMT = Simulation(sde, parametersEMT, endTime)

# parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType)
# simulationAM = Simulation(sde, parametersAM, endTime)

iters = 3

'''Time Transition Matrix'''
startEM = time.time()
for i in range(iters):
    simulationEM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersEM)
endEM = time.time()
endTime = endEM-startEM
print(endTime/iters)

startEM = time.time()
for i in range(iters):
    simulationEMT.timeDiscretizationMethod.computeTransitionMatrix(simulationEMT.pdf, sde, parametersEM)
endEM = time.time()
endTime = endEM-startEM
print(endTime/iters)


# startAM = time.time()
# for i in range(iters):
#     simulationAM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, h)
# endAM = time.time()
# endTime = endAM-startAM
# print(endTime/iters)


'''Time Adding New point'''
# startEM = time.time()
# for i in range(iters):
#     simulationEM.timeDiscretizationMethod.AddPointToG(simulationEM.pdf.meshCoordinates, 0, parametersEM, sde, simulationEM.pdf, simulationEM.integrator)
# endEM = time.time()
# endTime = endEM-startEM
# print(endTime/iters)

# startAM = time.time()
# for i in range(iters):
#     simulationAM.timeDiscretizationMethod.AddPointToG(simulationAM.pdf, [0], parametersAM, simulationAM.integrator, sde)
# endAM = time.time()
# endTime = endAM-startAM
# print(endTime/iters)



'''Time Adding New points'''
numPointsArr = [2]
timingEM = []
timingAM = []


# for numPoints in numPointsArr:
    # startEM = time.time()
    # for i in range(iters):
    #     for i in range(numPoints):
    #         simulationEM.timeDiscretizationMethod.AddPointToG(simulationEM.pdf.meshCoordinates, i, parametersEM, sde, simulationEM.pdf, simulationEM.integrator)
    # endEM = time.time()
    # endTime = endEM-startEM
    # print(endTime/iters)
    # timingEM.append(endTime/iters)

    # l = [i for i in range(numPoints)]
    # startAM = time.time()
    # for i in range(iters):
    #     simulationAM.timeDiscretizationMethod.AddPointToG(simulationAM.pdf, l, parametersAM, simulationAM.integrator, sde)
    # endAM = time.time()
    # endTime = endAM-startAM
    # print(endTime/iters)
    # timingAM.append(endTime/iters)

# plt.figure()
# plt.plot(np.asarray(numPointsArr), np.asarray(timingEM))
# plt.plot(np.asarray(numPointsArr), np.asarray(timingAM))
# plt.ylabel("Time")
# plt.xlabel("Number of added points")

# plt.figure()
# plt.semilogy(np.asarray(numPointsArr), np.asarray(timingAM)/np.asarray(timingEM))
# plt.ylabel("Time multiplier")
# plt.xlabel("timingAM/timingEM")











