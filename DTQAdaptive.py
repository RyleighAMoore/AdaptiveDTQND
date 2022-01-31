from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time
from PlottingResults import plotRowSixPlots
#TODO: Update this and clean up
dimension =1
if dimension ==1:
    beta = 4
    radius = 6
    kstepMin= 0.06
    kstepMax = 0.07
    kstepMin= 0.25
    kstepMax = 0.25
    h = 0.05
    endTime =5

if dimension ==2:
    beta = 3
    radius =2
    # radius = 0.5
    kstepMin= 0.2
    kstepMax = 0.3
    h = 0.05
    endTime = 3

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01
    endTime = 0.1

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift


diffusionFunction = functionBank.oneDiffusion


spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "EM", integratorType = "LQ")
simulation = Simulation(sde, parameters, endTime)

start = time.time()
simulation.setUpTransitionMatrix(sde, parameters)
TMTime = time.time()-start

start = time.time()
simulation.computeAllTimes(sde, parameters)
end = time.time()
print("\n")
print("Transition Matrix timing:", TMTime)
print("\n")
print("Stepping timing",end-start, '*****************************************')



# from exactSolutions import Solution

# trueSoln = []
# for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
#     truepdf = Solution(1, 0, simulation.meshTrajectory[i], (i+1)*h, dimension)
#     # truepdf = solution(xvec,-1,T)
#     trueSoln.append(np.squeeze(np.copy(truepdf)))

# from Errors import ErrorValsExact
# LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)



# endTime = 2

# spatialDiff = False
# sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
# parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "EM")
# simulation = Simulation(sde, parameters, endTime)
from exactSolutions import Solution

trueSoln = []
for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
    truepdf = Solution(1, 1, simulation.meshTrajectory[i], (i+1)*h, dimension)
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))

from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)



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
    ax.set_zlim(0,np.max(simulation.pdfTrajectory[10]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# index = 20
# ax.plot(Meshes[index][:,0], Meshes[index][:,1], PdfTraj[index], linestyle="", marker=".")

# if dimension ==3:
#     Meshes = simulation.meshTrajectory
#     PdfTraj = simulation.pdfTrajectory
#     from mpl_toolkits.mplot3d.art3d import juggle_axes
#     def update_graph(num):
#         # print(num)
#         # graph._offsets3d=(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2])
#         # graph.set_array(PdfTraj[num])
#         indx = 0
#         indy = 1
#         indz = 2
#         ax.clear()
#         ax.set_zlim(np.min(Meshes[-1][:,indz]),np.max(Meshes[-1][:,indz]))
#         ax.set_xlim(np.min(Meshes[-1][:,indx]),np.max(Meshes[-1][:,indx]))
#         ax.set_ylim(np.min(Meshes[-1][:,indy]),np.max(Meshes[-1][:,indy]))
#         graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='bone_r', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")

#         # graph.set_data(Meshes[num][:,0], Meshes[num][:,1])
#         # graph.set_3d_properties(Meshes[num][:,2], color=PdfTraj[num], cmap='binary')
#         # title.set_text('3D Test, time={}'.format(num))
#         return graph

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     title = ax.set_title('3D Test')
#     ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
#     ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
#     ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))


#     ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=1000, blit=False)
#     plt.show()


# from exactSolutions import Solution

# trueSoln = []
# for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
#     truepdf = Solution(1, 0, simulation.meshTrajectory[i], (i+1)*h, dimension)
#     # truepdf = solution(xvec,-1,T)
#     trueSoln.append(np.squeeze(np.copy(truepdf)))

# from Errors import ErrorValsExact
# LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)


# plotRowSixPlots(simulation.meshTrajectory, simulation.pdfTrajectory, h, [5, 10,20])

