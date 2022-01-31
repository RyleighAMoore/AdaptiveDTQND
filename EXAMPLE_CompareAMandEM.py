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
    beta = 5
    radius = 5
    kstepMin= 0.01
    kstepMax = 0.01
    # kstepMin= 0.15
    # kstepMax = 0.2
    # h = 0.01
    endTime =3

if dimension ==2:
    beta = 3
    radius =2
    kstepMin= 0.3
    kstepMax = 0.3
    h = 0.05
    endTime =1

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


diffusionFunction = functionBank.ptSixDiffusion


adaptive = False
integrationType = "TR"

ApproxSolution =True


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
if ApproxSolution:
    meshApprox, pdfApprox = sde.ApproxExactSoln(endTime,radius, 0.005, sde)

# with open('time4Erf.npy', 'wb') as f:
#     np.save(f, meshApprox)
#     np.save(f, pdfApprox)

# with open('time10Erf.npy', 'rb') as f:
#     meshApprox = np.load(f)
#     pdfApprox = np.load(f)



ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
timesNoStartupEM = []
timesNoStartupAM = []
timesStartupEM = []
timesStartupAM = []
hvals = [0.25, 0.35, 0.45]
hvalsAM =hvals# [0.4, 0.3, 0.2, 0.15, 0.1]
# hvals = [0.05]
# hvalsAM = [0.05]

initialCentering = [0]

for h in hvals:
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType, initialMeshCentering=initialCentering)
    startEM = time.time()
    simulationEM = Simulation(sde, parametersEM, endTime)
    simulationEM.setUpTransitionMatrix(sde, parametersEM)
    timeStartupEM = time.time() - startEM
    startEMNoStartup = time.time()
    simulationEM.computeAllTimes(sde, parametersEM)
    endEM = time.time()
    timesEM.append(np.copy(endEM-startEM))
    timesNoStartupEM.append(np.copy(endEM-startEMNoStartup))
    timesStartupEM.append(np.copy(timeStartupEM))


    if not ApproxSolution:
        meshApprox = simulationEM.pdfTrajectory[-1]
        pdfApprox = sde.exactSolution(simulationEM.meshTrajectory[-1], endTime)

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
    ErrorsEM.append(np.copy(L2wErrors))

for h in hvalsAM:
    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType, initialMeshCentering=initialCentering)
    startAM = time.time()
    simulationAM = Simulation(sde, parametersAM, endTime)
    simulationAM.setUpTransitionMatrix(sde, parametersAM)

    timeStartupAM = time.time() - startAM
    startAMNoStartup = time.time()
    simulationAM.computeAllTimes(sde, parametersAM)
    endAM =time.time()
    timesAM.append(np.copy(endAM-startAM))
    timesNoStartupAM.append(np.copy(endAM-startAMNoStartup))
    timesStartupAM.append(np.copy(timeStartupAM))


    if not ApproxSolution:
        meshApprox = simulationAM.pdfTrajectory[-1]
        pdfApprox = sde.exactSolution(simulationAM.meshTrajectory[-1], endTime)
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
    ErrorsAM.append(np.copy(L2wErrors))
    # del simulationAM


from mpl_toolkits.mplot3d import Axes3D

# fig =plt.figure()
# ax = Axes3D(fig)
# plt.scatter(meshApprox, pdfApprox)
# plt.scatter(simulationEM.meshTrajectory[0], simulationEM.pdfTrajectory[0])
# plt.scatter(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1])


# plt.scatter(simulationEM.meshTrajectory[-1],simulationEM.pdfTrajectory[-1])

# plt.figure()
# plt.semilogy(np.asarray(timesEM), np.asarray(ErrorsEM),'o-',label= "EM")
# plt.semilogy(np.asarray(timesAM), np.asarray(ErrorsAM), 'o-', label="AM")
# plt.semilogy(np.asarray(timesNoStartupEM), np.asarray(ErrorsEM),'.-', label= "EM: No Startup")
# plt.semilogy(np.asarray(timesNoStartupAM), np.asarray(ErrorsAM),'.-', label="AM: No Startup")
# plt.semilogy(np.asarray(timesStartupAM), np.asarray(ErrorsAM),'.-', label="AM: Startup")
# plt.semilogy(np.asarray(timesStartupEM), np.asarray(ErrorsEM),'.-', label="EM: Startup")

# plt.legend()
# plt.xlabel("Time (Seconds)")
# plt.ylabel("Error")

# plt.show()
# plt.savefig('result.png')

# plt.figure()
# plt.loglog(np.asarray(timesEM), np.asarray(ErrorsEM),'o-',label= "EM: Total Time")
# plt.loglog(np.asarray(timesAM), np.asarray(ErrorsAM), 'o-', label="AM: Total Time")
# # plt.loglog(np.asarray(timesNoStartupEM), np.asarray(ErrorsEM),'.-', label= "EM: No Startup")
# # plt.loglog(np.asarray(timesNoStartupAM), np.asarray(ErrorsAM),'.-', label="AM: No Startup")
# # plt.loglog(np.asarray(timesStartupAM), np.asarray(ErrorsAM),'.-', label="AM: Startup")
# # plt.loglog(np.asarray(timesStartupEM), np.asarray(ErrorsEM),'.-', label="EM: Startup")

# plt.legend()
# plt.xlabel("Time (Seconds)")
# plt.ylabel("Error")

# plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

simulation = simulationEM
if dimension ==1:
    def update_graph(num):
        graph.set_data(simulation.meshTrajectory[num], simulation.pdfTrajectory[num])
        return title, graph

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = ax.set_title('2D Test')

    graph, = ax.plot(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], linestyle="", marker=".")
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, np.max(simulation.pdfTrajectory[0]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(simulation.pdfTrajectory), interval=50, blit=False)
    plt.show()

plt.figure()

lengths = []
for  i in range(len(simulation.pdfTrajectory)):
    l = len(np.asarray((simulation.pdfTrajectory[i])))
    lengths.append(l)


plt.figure()
plt.semilogy(np.asarray(hvals), np.asarray(ErrorsEM),'o', label= "EM")
plt.semilogy(np.asarray(hvalsAM), np.asarray(ErrorsAM),'o', label="AM")
plt.ylabel("Errors")
plt.xlabel("Temporal Step Size")
plt.legend()

# plt.figure()
# plt.plot(np.asarray(hvals), np.asarray(timesEM),label= "EM")
# plt.plot(np.asarray(hvalsAM), np.asarray(timesAM), label="AM")
# plt.ylabel("Time")
# plt.xlabel("Temporal Step Size")
# plt.legend()




