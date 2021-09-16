from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt


dimension = 1
beta = 4
radius = 2
kstepMin= 0.06
kstepMax = 0.07
h = 0.1
endTime = 2


def driftFunction(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 0
    return dr

def diffusionFunction(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    if dimension ==1:
        return 1*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    else:
        return 1*np.diag(np.ones(dimension))


spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "AM")
simulation = Simulation(sde, parameters, endTime)


from exactSolutions import Solution

trueSoln = []
for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
    truepdf = Solution(1, 0, simulation.meshTrajectory[i], (i+1)*h, dimension)
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))

from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(simulation.meshTrajectory, simulation.pdfTrajectory, trueSoln, h, plot=False)



endTime = 2

spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "EM")
simulation = Simulation(sde, parameters, endTime)
from exactSolutions import Solution

trueSoln = []
for i in range(len(simulation.meshTrajectory)): #diff, drift, mesh, t, dim
    truepdf = Solution(1, 0, simulation.meshTrajectory[i], (i+1)*h, dimension)
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
#     ax.set_zlim(0, 4.5)
#     ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
#     plt.show()

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


