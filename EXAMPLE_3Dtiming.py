from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension = 3
radius = 0.5
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
beta = 3

numPointsAM = []
numPointsEM = []

endTimes = [0.12]
bufferVals = [0]
timingVals = []

plt.figure()
for endTime in endTimes:

    spacing = 0.1
    parametersEM = Parameters(sde, beta, radius, spacing, spacing+0.05, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ")

    startEM = time.time()
    simulationEM = Simulation(sde, parametersEM, endTime)
    simulationEM.setUpTransitionMatrix(sde, parametersEM)
    startupEM = time.time() - startEM

    timingEM = simulationEM.computeAllTimes(sde, parametersEM)
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
    zmin = min(np.min(simulationEM.meshTrajectory[-1][:,2]),np.min(simulationEM.meshTrajectory[0][:,2]))
    zmax = max(np.max(simulationEM.meshTrajectory[-1][:,2]),np.max(simulationEM.meshTrajectory[0][:,2]))

    for bufferVal in bufferVals:
        spacing = 0.07
        bufferX = bufferVal*(xmax-xmin)/2
        bufferY = bufferVal*(ymax-ymin)/2
        bufferZ = bufferVal*(zmax-zmin)/2

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


        zstart = np.floor(xmin) - bufferX
        zs = []
        zs.append(zstart)
        while zstart< zmax + bufferZ:
            zs.append(zstart+spacing)
            zstart += spacing

        mesh = []
        for i in xs:
            for j in ys:
                for k in zs:
                    mesh.append([i,j, k])
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

if dimension ==3:
    Meshes = simulation.meshTrajectory
    PdfTraj = simulation.pdfTrajectory
    from mpl_toolkits.mplot3d.art3d import juggle_axes
    def update_graph(num):
        # print(num)
        # graph._offsets3d=(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2])
        # graph.set_array(PdfTraj[num])
        indx = 0
        indy = 1
        indz = 2
        ax.clear()
        ax.set_zlim(np.min(Meshes[-1][:,indz]),np.max(Meshes[-1][:,indz]))
        ax.set_xlim(np.min(Meshes[-1][:,indx]),np.max(Meshes[-1][:,indx]))
        ax.set_ylim(np.min(Meshes[-1][:,indy]),np.max(Meshes[-1][:,indy]))
        graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='bone_r', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")

        # graph.set_data(Meshes[num][:,0], Meshes[num][:,1])
        # graph.set_3d_properties(Meshes[num][:,2], color=PdfTraj[num], cmap='binary')
        # title.set_text('3D Test, time={}'.format(num))
        return graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
    ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
    ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))


    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=1000, blit=False)
    plt.show()





