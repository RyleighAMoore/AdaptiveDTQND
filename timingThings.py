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

# parametersEMT = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="TR")
# simulationEMT = Simulation(sde, parametersEMT, endTime)

parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType)
simulationAM = Simulation(sde, parametersAM, endTime)

# AM = simulationAM.integrator.TransitionMatrix
# EM = simulationEM.integrator.TransitionMatrix

# print(np.nanmax(abs(AM-EM)))

iters = 1

'''Time Transition Matrix'''
startEM = time.time()
for i in range(iters):
    simulationEM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersEM)
endEM = time.time()
endTime = endEM-startEM
print(endTime/iters)

startAM = time.time()
for i in range(iters):
    simulationAM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersAM)
endAM = time.time()
endTime = endAM-startAM
print(endTime/iters)


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



# '''Time Adding New points'''
# numPointsArr = [10000, 6000, 2000, 1000,10,1]
# numPointsArr = [900, 500, 300,100,10,1]

# timingEM = []
# timingAM = []

# from tqdm import trange
# for numPoints in numPointsArr:
#     startEM = time.time()
#     for i in trange(iters):
#         for i in range(numPoints):
#             simulationEM.timeDiscretizationMethod.AddPointToG(simulationEM.pdf.meshCoordinates, i, parametersEM, sde, simulationEM.pdf, simulationEM.integrator)
#     endEM = time.time()
#     endTime = endEM-startEM
#     print(endTime/iters)
#     timingEM.append(endTime/iters)

#     l = [i for i in range(numPoints)]
#     startAM = time.time()
#     for i in trange(iters):
#         simulationAM.timeDiscretizationMethod.AddPointToG(simulationAM.pdf, l, parametersAM, simulationAM.integrator, sde)
#     endAM = time.time()
#     endTime = endAM-startAM
#     print(endTime/iters)
#     timingAM.append(endTime/iters)

# # if timingEM[0] == 0:
# #     timingEM[0] = 10**(-128)
plt.figure()
plt.loglog(np.asarray(numPointsArr), np.asarray(timingEM), 'o', label="EM")
plt.loglog(np.asarray(numPointsArr), np.asarray(timingAM),'o', label = "AM")
plt.ylabel("Time")
plt.title("Timing to add points 1D")
plt.xlabel("Number of added points")
plt.legend()

# plt.figure()
# plt.semilogy(np.asarray(numPointsArr), np.asarray(timingAM)/np.asarray(timingEM))
# plt.ylabel("Time multiplier")
# plt.xlabel("timingAM/timingEM")











