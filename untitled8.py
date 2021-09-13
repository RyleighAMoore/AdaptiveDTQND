from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt


dimension = 1
beta = 2
radius = 2
kstepMin= 0.06
kstepMax = 0.07
h = 0.01
endTime =1

dimension =2
beta = 3
radius =0.8
kstepMin= 0.06
kstepMax = 0.06
h = 0.01
endTime = 0.6

# def driftFunction(mesh):
#       if mesh.ndim ==1:
#         mesh = np.expand_dims(mesh, axis=0)
#     # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return -1*mesh
#       return np.zeros(np.shape(mesh))

# def diffusionFunction(mesh):
#     return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

def driftFunction(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 2
    return dr

def diffusionFunction(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    if dimension ==1:
        return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    else:
        return 0.5*np.diag(np.ones(dimension))


spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "EM")
simulation = Simulation(sde, parameters, endTime)
# plt.scatter(simulation.pdf.meshCoordinates,simulation.pdf.pdfVals)
# plt.scatter(simulation.meshTrajectory[-1],simulation.pdfTrajectory[-1])


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
    ax.set_zlim(0, 4.5)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()


from exactSolutions import Solution

trueSoln = []
for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
    truepdf = Solution(0.5, 2, simulation.meshTrajectory[i], (i+1)*h, dimension)
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))

from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)


