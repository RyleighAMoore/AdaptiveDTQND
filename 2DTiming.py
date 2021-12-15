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
if dimension ==2:
    radius = 2
    h = 0.05

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
beta = 3

numPointsAM = []
numPointsEM = []

endTimes = [10]
bufferVals = [0,0.5]
timingVals = []

plt.figure()
for endTime in endTimes:

    spacing = 0.38
    parametersEM = Parameters(sde, beta, radius, spacing, spacing+0.1, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ")

    startEM = time.time()
    simulationEM = Simulation(sde, parametersEM, endTime)
    startupEM = time.time() - startEM

    timingEM = simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
    for i in range(len(timingEM)):
        timingEM[i] += startupEM

    timingEM.insert(0, startupEM)

    meshApprox = simulationEM.meshTrajectory[-1]
    pdfApprox = sde.exactSolution(simulationEM.meshTrajectory[-1], endTime)

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
    ErrorsEM.append(np.copy(L2wErrors))
    numPointsEM.append(np.copy(simulationEM.pdf.meshLength))
    times = np.asarray(list(range(0,len(timingEM))))*h
    plt.loglog(times, np.asarray(timingEM),'o', label= "LQ")



    xmin = min(np.min(simulationEM.meshTrajectory[-1][:,0]),np.min(simulationEM.meshTrajectory[0][:,0]))
    xmax = max(np.max(simulationEM.meshTrajectory[-1][:,0]),np.max(simulationEM.meshTrajectory[0][:,0]))
    ymin = min(np.min(simulationEM.meshTrajectory[-1][:,1]),np.min(simulationEM.meshTrajectory[0][:,1]))
    ymax = max(np.max(simulationEM.meshTrajectory[-1][:,1]),np.max(simulationEM.meshTrajectory[0][:,1]))

    for bufferVal in bufferVals:
        spacing = 0.2
        bufferX =bufferVal*(xmax-xmin)/2
        bufferY = bufferVal*(ymax-ymin)/2
        xstart = np.floor(xmin) - bufferX
        xs = []
        xs.append(xstart)
        while xstart< xmax + bufferX:
            xs.append(xstart+spacing)
            xstart += spacing

        ystart = np.floor(ymin) - bufferY
        ys = []
        ys.append(ystart)

        while ystart< ymax+ bufferY:
            ys.append(ystart+spacing)
            ystart += spacing

        mesh = []
        for i in xs:
            for j in ys:
                mesh.append([i,j])
        mesh = np.asarray(mesh)
        parametersAM = Parameters(sde, beta, radius, spacing, spacing, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = mesh)

        startAM = time.time()
        print(time.time())
        simulationAM = Simulation(sde, parametersAM, endTime)
        print(time.time())

        startupAM = time.time()-startAM
        timingAM = simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
        for i in range(len(timingAM)):
            timingAM[i] += startupAM
        timingAM.insert(0, startupAM)
        plt.loglog(times, np.asarray(timingAM),'o', label= bufferVal)



        if not ApproxSolution:
            meshApprox = simulationAM.meshTrajectory[-1]
            pdfApprox = sde.exactSolution(simulationAM.meshTrajectory[-1], endTime)
        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
        ErrorsAM.append(np.copy(L2wErrors))
        numPointsAM.append(np.copy(simulationAM.pdf.meshLength))
plt.legend()

# plt.figure()
# plt.loglog(np.asarray(times), np.asarray(timingEM),'o', label= "LQ")
# plt.loglog(np.asarray(times), np.asarray(timingAM),'.', label="TR")
# plt.xlabel("Time")
# plt.ylabel("Total Time")

# plt.figure()
# plt.plot(np.asarray(times), np.asarray(timingEM),'o', label= "LQ")
# plt.plot(np.asarray(times), np.asarray(timingAM),'.', label="TR")
# plt.xlabel("Time")
# plt.ylabel("Total Time")


plt.legend()
plt.savefig('timingPlot.png')


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
    # ax.set_zlim(0, 1.5)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=10, blit=False)
    plt.show()






