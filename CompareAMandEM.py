from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension = 1
if dimension ==1:
    beta = 4
    radius = 5
    kstepMin= 0.06
    kstepMax = 0.07
    h = 0.01
    endTime =10

if dimension ==2:
    beta = 3
    radius =1
    kstepMin= 0.08
    kstepMax = 0.09
    h = 0.05
    endTime = 0.5


if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01
    endTime = 0.1

# driftFunction = functionBank.zeroDrift
driftFunction = functionBank.erfDrift
spatialDiff = False


diffusionFunction = functionBank.oneDiffusion

ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
timesNoStartupEM = []
timesNoStartupAM = []


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
meshApprox, pdfApprox = sde.ApproxExactSoln(endTime,20, 0.05)
hvals = [0.01, 0.05, 0.1]
# hvals =[0.05]
for h in hvals:

    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =True, timeDiscretizationType = "EM")
    startEM = time.time()
    simulationEM = Simulation(sde, parametersEM, endTime)

    startEMNoStartup = time.time()
    simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
    endEM = time.time()
    timesEM.append(np.copy(endEM-startEM))
    timesNoStartupEM.append(np.copy(endEM-startEMNoStartup))

    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =True, timeDiscretizationType = "AM")
    startAM = time.time()
    simulationAM = Simulation(sde, parametersAM, endTime)
    startAMNoStartup = time.time()
    simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
    endAM =time.time()
    timesAM.append(np.copy(endAM-startAM))
    timesNoStartupAM.append(np.copy(endEM-startAMNoStartup))


    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, h)
    ErrorsEM.append(np.copy(L2wErrors))
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1], meshApprox, pdfApprox, h)
    ErrorsAM.append(np.copy(L2wErrors))

from mpl_toolkits.mplot3d import Axes3D

# fig =plt.figure()
# ax = Axes3D(fig)
# ax.scatter(meshApprox[:,0],meshApprox[:,1], pdfApprox)
# plt.scatter(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1])

# plt.scatter(simulationEM.meshTrajectory[-1],simulationEM.pdfTrajectory[-1])

plt.figure()
plt.semilogy(np.asarray(timesEM), np.asarray(ErrorsEM),label= "EM")
plt.semilogy(np.asarray(timesAM), np.asarray(ErrorsAM), label="AM")
plt.semilogy(np.asarray(timesNoStartupEM), np.asarray(ErrorsEM), label= "EM: No Startup")
plt.semilogy(np.asarray(timesNoStartupAM), np.asarray(ErrorsAM), label="AM: No Startup")
plt.legend()
plt.show()
plt.savefig('result.png')


# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# simulation = simulationAM
# if dimension ==1:
#     def update_graph(num):
#         graph.set_data(simulation.meshTrajectory[num], simulation.pdfTrajectory[num])
#         return title, graph

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     title = ax.set_title('2D Test')

#     graph, = ax.plot(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], linestyle="", marker=".")
#     ax.set_xlim(-20, 20)
#     ax.set_ylim(0, np.max(simulation.pdfTrajectory[0]))
#     ani = animation.FuncAnimation(fig, update_graph, frames=len(simulation.pdfTrajectory), interval=50, blit=False)
#     plt.show()




