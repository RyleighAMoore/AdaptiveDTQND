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
    endTime = 1

if dimension == 3:
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

adaptive = True
integrationType = "LQ"

ApproxSolution = False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)


'''Time Adding New points'''
spacingVals =  [0.1]
spacingVals = [0.05, 0.1, 0.2]
hvals = [0.1, 0.2]


from tqdm import trange
from Errors import ErrorValsExact
from exactSolutions import Solution

plt.figure()

for h in hvals:
    timingEM = []
    timingAM = []
    lengths = []
    Errors = []
    for AMMeshSpacing in spacingVals:
        parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM", integratorType=integrationType,  AMSpacing = AMMeshSpacing)
        simulation = Simulation(sde, parameters, endTime)
        simulation.computeAllTimes(sde, simulation.pdf, parameters)

        trueSoln = []
        for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
            truepdf = Solution(1, 0, simulation.meshTrajectory[i], (i+1)*h, dimension)
            trueSoln.append(np.squeeze(np.copy(truepdf)))

        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)
        Errors.append(L2wErrors[-1])


    plt.loglog(np.asarray(spacingVals), np.asarray(Errors), 'o', label = h)
plt.ylabel("Errors")
plt.title("")
plt.xlabel("SpacingVals")
plt.legend()



import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

if dimension ==2:
    Meshes = simulation.meshTrajectory
    PdfTraj = simulation.pdfTrajectory
    def update_graph(num):
        graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
        graph.set_3d_properties(PdfTraj[num])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
    ax.set_zlim(0, 1.5)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()






