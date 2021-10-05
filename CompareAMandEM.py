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
    beta = 3
    radius =15
    kstepMin= 0.06
    kstepMax = 0.065
    # h = 0.01
    endTime =4

if dimension ==2:
    beta = 3
    radius =3
    kstepMin= 0.08
    kstepMax = 0.09
    h = 0.05
    endTime = 0.4

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    # h = 0.01
    endTime = 0.1

driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
# driftFunction = functionBank.oneDrift

spatialDiff = False


diffusionFunction = functionBank.oneDiffusion

ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
timesNoStartupEM = []
timesNoStartupAM = []

adaptive = False
integrationType = "TR"

ApproxSolution =False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
if ApproxSolution:
    meshApprox, pdfApprox = sde.ApproxExactSoln(endTime,4, 0.005)


hvals = [0.01, 0.1, 0.2, 0.3]
for h in hvals:
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType=integrationType)
    startEM = time.time()
    simulationEM = Simulation(sde, parametersEM, endTime)

    startEMNoStartup = time.time()
    simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
    endEM = time.time()
    timesEM.append(np.copy(endEM-startEM))
    timesNoStartupEM.append(np.copy(endEM-startEMNoStartup))

    if not ApproxSolution:
        meshApprox = simulationEM.pdfTrajectory[-1]
        pdfApprox = sde.exactSolution(simulationEM.meshTrajectory[-1], endTime)

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
    ErrorsEM.append(np.copy(L2wErrors))
    # del simulationEM

    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType)
    startAM = time.time()
    simulationAM = Simulation(sde, parametersAM, endTime)
    startAMNoStartup = time.time()
    simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
    endAM =time.time()
    timesAM.append(np.copy(endAM-startAM))
    timesNoStartupAM.append(np.copy(endAM-startAMNoStartup))
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

plt.figure()
plt.semilogy(np.asarray(timesEM), np.asarray(ErrorsEM),'o-',label= "EM")
plt.semilogy(np.asarray(timesAM), np.asarray(ErrorsAM), 'o-', label="AM")
plt.semilogy(np.asarray(timesNoStartupEM), np.asarray(ErrorsEM),'.-', label= "EM: No Startup")
plt.semilogy(np.asarray(timesNoStartupAM), np.asarray(ErrorsAM),'.-', label="AM: No Startup")
plt.legend()
plt.xlabel("Time (Seconds)")
plt.ylabel("Error")

plt.show()
plt.savefig('result.png')


import matplotlib.pyplot as plt
import matplotlib.animation as animation
simulation = simulationAM
if dimension ==1:
    def update_graph(num):
        graph.set_data(simulation.meshTrajectory[num], simulation.pdfTrajectory[num])
        return title, graph

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = ax.set_title('2D Test')

    graph, = ax.plot(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], linestyle="", marker=".")
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, np.max(simulation.pdfTrajectory[0]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(simulation.pdfTrajectory), interval=50, blit=False)
    plt.show()

# plt.figure()

# lengths = []
# for  i in range(len(simulation.pdfTrajectory)):
#     l = len(np.asarray((simulation.pdfTrajectory[i])))
#     lengths.append(l)


plt.figure()
plt.semilogy(np.asarray(hvals), np.asarray(ErrorsEM),label= "EM")
plt.semilogy(np.asarray(hvals), np.asarray(ErrorsAM), label="AM")
plt.ylabel("Errors")
plt.xlabel("Temporal Step Size")
plt.legend()

plt.figure()
plt.plot(np.asarray(hvals), np.asarray(timesEM),label= "EM")
plt.plot(np.asarray(hvals), np.asarray(timesAM), label="AM")
plt.ylabel("Time")
plt.xlabel("Temporal Step Size")
plt.legend()




