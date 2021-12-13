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
    beta = 5
    radius = 3
    # kstepMin= 0.08
    # kstepMax = 0.09
    kstepMin= 0.15
    kstepMax = 0.2
    h = 0.01

if dimension ==2:
    beta = 3
    radius =2
    # radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.09
    kstepMin= 0.13
    kstepMax = 0.15
    h = 0.05

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift

spatialDiff = False


diffusionFunction = functionBank.ptSixDiffusion


adaptive = True

ApproxSolution =False

sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)

ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
betaVals = [3]
# betaVals = [2.5]
# radiusVals = [1, 4]
# spacingVals = [0.08, 0.05]
spacingVals = [0.12]

# times = [4,8,10]
# times = [12]
times= [1,5,10]
numPointsAM = []
numPointsEM = []

for endTime in times:
    for beta in betaVals:
        for spacing in spacingVals:
            parametersEM = Parameters(sde, beta, radius, spacing, spacing+0.2, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ")
            startEM = time.time()
            simulationEM = Simulation(sde, parametersEM, endTime)
            simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
            endEM = time.time()
            timesEM.append(np.copy(endEM-startEM))

            meshApprox = simulationEM.meshTrajectory[-1]
            pdfApprox = sde.exactSolution(simulationEM.meshTrajectory[-1], endTime)

            LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
            ErrorsEM.append(np.copy(L2wErrors))
            numPointsEM.append(np.copy(simulationEM.pdf.meshLength))


    xmin = min(np.min(simulationEM.meshTrajectory[-1][:,0]),np.min(simulationEM.meshTrajectory[0][:,0]))
    xmax = max(np.max(simulationEM.meshTrajectory[-1][:,0]),np.max(simulationEM.meshTrajectory[0][:,0]))
    ymin = min(np.min(simulationEM.meshTrajectory[-1][:,1]),np.min(simulationEM.meshTrajectory[0][:,1]))
    ymax = max(np.max(simulationEM.meshTrajectory[-1][:,1]),np.max(simulationEM.meshTrajectory[0][:,1]))


    for spacing in spacingVals:
        buffer = 1
        xstart = np.floor(xmin) - buffer
        xs = []
        xs.append(xstart)
        while xstart< xmax + buffer:
            xs.append(xstart+spacing)
            xstart += spacing

        ystart = np.floor(ymin) - buffer
        ys = []
        ys.append(ystart)

        while ystart< ymax+ buffer:
            ys.append(ystart+spacing)
            ystart += spacing

        mesh = []
        for i in xs:
            for j in ys:
                mesh.append([i,j])
        mesh = np.asarray(mesh)
        parametersAM = Parameters(sde, beta, radius, spacing, spacing, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = mesh)
        startAM = time.time()
        simulationAM = Simulation(sde, parametersAM, endTime)
        # plt.figure()
        # plt.scatter(simulationAM.pdf.meshCoordinates[:,0], simulationAM.pdf.meshCoordinates[:,1])
        timeStartupAM = time.time() - startAM
        startAMNoStartup = time.time()
        simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
        endAM =time.time()
        timesAM.append(np.copy(endAM-startAM))


        if not ApproxSolution:
            meshApprox = simulationAM.meshTrajectory[-1]
            pdfApprox = sde.exactSolution(simulationAM.meshTrajectory[-1], endTime)
        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
        ErrorsAM.append(np.copy(L2wErrors))
        numPointsAM.append(np.copy(simulationAM.pdf.meshLength))



plt.figure()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
Meshes = simulationAM.meshTrajectory
PdfTraj = simulationAM.pdfTrajectory
graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
ax.plot(meshApprox[:,0], meshApprox[:,1], pdfApprox, color= "red",linestyle="", marker=".")


plt.figure()
simulation = simulationAM
Meshes = simulation.meshTrajectory
plt.plot(Meshes[-1][:,0], Meshes[-1][:,1],'or')

simulation = simulationEM
Meshes = simulation.meshTrajectory
plt.plot(Meshes[-1][:,0], Meshes[-1][:,1], '.b')
plt.plot(Meshes[0][:,0], Meshes[0][:,1], '.b')



from mpl_toolkits.mplot3d import Axes3D
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
    ax.set_xlim(-40, 40)
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
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=10, blit=False)
    plt.show()




plt.figure()
plt.semilogx(np.asarray(timesEM), np.asarray(ErrorsEM),'o', label= "EM LQ")
plt.semilogy(np.asarray(timesAM), np.asarray(ErrorsAM),'o', label="EM TR")
plt.semilogx(np.asarray(timesEM), np.asarray(numPointsEM),'.', label= "LQ Points")
plt.semilogy(np.asarray(timesAM), np.asarray(numPointsAM),'.', label="TR Points")
plt.xlabel("Time")
plt.ylabel("Errors")
plt.legend()
plt.savefig('timingPlot.png')


plt.figure()
plt.semilogx(np.asarray(times), np.asarray(timesEM),'o', label= "EM LQ")
plt.semilogy(np.asarray(times), np.asarray(timesAM),'o', label="EM TR")
plt.semilogx(np.asarray(times), np.asarray(numPointsEM),'.', label= "LQ Points")
plt.semilogy(np.asarray(times), np.asarray(numPointsAM),'.', label="TR Points")
plt.xlabel("End Time")
plt.ylabel("timing")
plt.legend()






