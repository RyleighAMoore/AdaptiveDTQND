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
    radius =10
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


iters = 10

# '''Time Transition Matrix'''
# startEM = time.time()
# for i in range(iters):
#     simulationEM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersEM)
# endEM = time.time()
# endTime = endEM-startEM
# print(endTime/iters)

# startAM = time.time()
# for i in range(iters):
#     simulationAM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersAM)
# endAM = time.time()
# endTime = endAM-startAM
# print(endTime/iters)


# '''Time Adding New point'''
# # startEM = time.time()
# # for i in range(iters):
# #     simulationEM.timeDiscretizationMethod.AddPointToG(simulationEM.pdf.meshCoordinates, 0, parametersEM, sde, simulationEM.pdf, simulationEM.integrator)
# # endEM = time.time()
# # endTime = endEM-startEM
# # print(endTime/iters)

# # startAM = time.time()
# # for i in range(iters):
# #     simulationAM.timeDiscretizationMethod.AddPointToG(simulationAM.pdf, [0], parametersAM, simulationAM.integrator, sde)
# # endAM = time.time()
# # endTime = endAM-startAM
# # print(endTime/iters)



'''Time Adding New points'''
rvals = [80, 40, 20, 10]

rvals = [1.5, 1, 0.5, 0.2]

timingEM = []
timingAM = []
lengths = []

from tqdm import trange
for radius in rvals:
    numPoints  =1
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType)
    simulationEM = Simulation(sde, parametersEM, endTime)

    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType)
    simulationAM = Simulation(sde, parametersAM, endTime)
    startEM = time.time()
    for i in trange(iters):
        for i in range(numPoints):
            simulationEM.timeDiscretizationMethod.AddPointToG(simulationEM.pdf.meshCoordinates, i, parametersEM, sde, simulationEM.pdf, simulationEM.integrator)
    endEM = time.time()
    endTime = endEM-startEM
    print(endTime/iters)
    timingEM.append(endTime/iters)

    l = [i for i in range(numPoints)]
    startAM = time.time()
    for i in trange(iters):
        simulationAM.timeDiscretizationMethod.AddPointToG(simulationAM.pdf, l, parametersAM, simulationAM.integrator, sde)
    endAM = time.time()
    endTime = endAM-startAM
    print(endTime/iters)
    timingAM.append(endTime/iters)
    lengths.append(np.copy(simulationAM.pdf.meshLength))



# # if timingEM[0] == 0:
# #     timingEM[0] = 10**(-128)
plt.figure()
plt.loglog(np.asarray(lengths), np.asarray(timingEM), 'o', label="EM")
plt.loglog(np.asarray(lengths), np.asarray(timingAM),'o', label = "AM")
plt.ylabel("Time")
plt.title("Timing to add one point 2D")
plt.xlabel("Number of points in mesh")
plt.legend()

# plt.figure()
# plt.semilogy(np.asarray(numPointsArr), np.asarray(timingAM)/np.asarray(timingEM))
# plt.ylabel("Time multiplier")
# plt.xlabel("timingAM/timingEM")











