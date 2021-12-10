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
    endTime = 3

if dimension ==2:
    beta = 3
    radius =2
    # radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.09
    kstepMin= 0.1
    kstepMax = 0.12
    h = 0.05
    endTime = 1

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01
    endTime = 0.4

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift

spatialDiff = False


diffusionFunction = functionBank.ptSixDiffusion


adaptive = True

ApproxSolution =False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
if ApproxSolution:
    meshApprox, pdfApprox = sde.ApproxExactSoln(endTime,1, 0.02)

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
betaVals = [2,3]
# radiusVals = [1, 4]
spacingVals = [0.08, 0.05]
# times = [4,8,10]
# times = [12]
times= [15]


for EndTime in times:
    for beta in betaVals:
        parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ")
        startEM = time.time()
        simulationEM = Simulation(sde, parametersEM, EndTime)
        timeStartupEM = time.time() - startEM
        startEMNoStartup = time.time()
        simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
        endEM = time.time()
        timesEM.append(np.copy(endEM-startEM))
        timesNoStartupEM.append(np.copy(endEM-startEMNoStartup))
        timesStartupEM.append(np.copy(timeStartupEM))


        if not ApproxSolution:
            meshApprox = simulationEM.pdfTrajectory[-1]
            pdfApprox = sde.exactSolution(simulationEM.meshTrajectory[-1], EndTime)

        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationEM.meshTrajectory[-1], simulationEM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
        ErrorsEM.append(np.copy(L2wErrors))

    spans = []
    centering = []
    minMin = 1000
    maxMax = -1000
    for i in range(sde.dimension):
        vals = simulationEM.meshTrajectory[-1][:,i]
        vals0 = simulationEM.meshTrajectory[0][:,i]
        minVal = min(np.min(vals), np.min(vals0))
        maxVal = max(np.max(vals), np.max(vals0))
        minMin = min(minVal, minMin)
        maxMax = max(maxVal, maxMax)
        centering.append(maxVal/2+minVal/2)
        spans.append(maxVal - minVal)


    radius = max(spans)/4
    initialCentering = np.asarray(centering)

    xmin = min(np.min(simulationEM.meshTrajectory[-1][:,0]),np.min(simulationEM.meshTrajectory[0][:,0]))
    xmax = max(np.max(simulationEM.meshTrajectory[-1][:,0]),np.max(simulationEM.meshTrajectory[0][:,0]))
    ymin = min(np.min(simulationEM.meshTrajectory[-1][:,1]),np.min(simulationEM.meshTrajectory[0][:,1]))
    ymax = max(np.max(simulationEM.meshTrajectory[-1][:,1]),np.max(simulationEM.meshTrajectory[0][:,1]))


    h= 0.01
    nx, ny = (int(np.ceil((xmax-xmin))/h), int(np.ceil((ymax-ymin))/h))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    # xstart = np.floor(xmin)
    # xs = []
    # xs.append(xstart)
    # while xstart< xmax:
    #     xs.append(xstart+spacing)
    #     xstart += spacing

    # ystart = np.floor(ymin)
    # ys = []
    # ys.append(ystart)

    # while ystart< ymax:
    #     ys.append(ystart+spacing)
    #     ystart += spacing


    # mesh = []
    # for i in xs:
    #     for j in ys:
    #         mesh.append([i,j])
    # mesh = np.asarray(mesh)

    # plt.figure()
    # plt.scatter(simulationEM.meshTrajectory[-1][:,0], simulationEM.meshTrajectory[-1][:,1])
    for spacing in spacingVals:
        xstart = np.floor(xmin)
        xs = []
        xs.append(xstart)
        while xstart< xmax:
            xs.append(xstart+spacing)
            xstart += spacing

        ystart = np.floor(ymin)
        ys = []
        ys.append(ystart)

        while ystart< ymax:
            ys.append(ystart+spacing)
            ystart += spacing


        mesh = []
        for i in xs:
            for j in ys:
                mesh.append([i,j])
        mesh = np.asarray(mesh)
        parametersAM = Parameters(sde, beta, radius, spacing, spacing, 0.01,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", initialMeshCentering=initialCentering, OverideMesh = mesh)
        startAM = time.time()
        simulationAM = Simulation(sde, parametersAM, EndTime)
        # plt.figure()
        # plt.scatter(simulationAM.pdf.meshCoordinates[:,0], simulationAM.pdf.meshCoordinates[:,1])
        timeStartupAM = time.time() - startAM
        startAMNoStartup = time.time()
        simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
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

# simulation = simulationEM
# if dimension ==1:
#     def update_graph(num):
#         graph.set_data(simulation.meshTrajectory[num], simulation.pdfTrajectory[num])
#         return title, graph

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     title = ax.set_title('2D Test')

#     graph, = ax.plot(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], linestyle="", marker=".")
#     ax.set_xlim(-40, 40)
#     ax.set_ylim(0, np.max(simulation.pdfTrajectory[0]))
#     ani = animation.FuncAnimation(fig, update_graph, frames=len(simulation.pdfTrajectory), interval=50, blit=False)
#     plt.show()

# if dimension ==2:
#     Meshes = simulation.meshTrajectory
#     PdfTraj = simulation.pdfTrajectory
#     def update_graph(num):
#         graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
#         graph.set_3d_properties(PdfTraj[num])
#         title.set_text('3D Test, time={}'.format(num))
#         return title, graph

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     title = ax.set_title('3D Test')

#     graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
#     ax.set_zlim(0, 1.5)
#     ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=10, blit=False)
#     plt.show()




plt.figure()
plt.semilogx(np.asarray(timesEM), np.asarray(ErrorsEM),'o', label= "EM LQ")
plt.semilogy(np.asarray(timesAM), np.asarray(ErrorsAM),'o', label="EM TR")
plt.xlabel("Time")
plt.ylabel("Errors")
plt.legend()
plt.savefig('timingPlot.png')


# plt.figure()
# plt.semilogx(np.asarray(times), np.asarray(timesEM),'o', label= "EM LQ")
# plt.semilogy(np.asarray(times), np.asarray(timesAM),'o', label="EM TR")
# # plt.semilogx(np.asarray(times), np.asarray(timesNoStartupEM),'.', label= "EM LQ")
# # plt.semilogy(np.asarray(times), np.asarray(timesNoStartupAM),'.', label="EM TR")
# plt.xlabel("End Time")
# plt.ylabel("timing")
# plt.legend()





