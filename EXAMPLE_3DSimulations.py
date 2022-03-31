from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time
from PlottingResults import plotRowSixPlots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Errors import ErrorValsOneTime


problem = "erf" # "spiral" "complex"

dimension =3
beta = 3
radius = 1


if problem == "hill":
    driftFunction = functionBank.oneDrift
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.22
    kstepMax = 0.22
    endTime = 1
    h=0.02
    vminVal = -5
    vmaxVal = 1


if problem == "erf":
    driftFunction = functionBank.erfDrift
    diffusionFunction = functionBank.pt75Diffusion
    spatialDiff = False
    kstepMin = 0.25
    kstepMax = 0.25
    endTime =1.22
    h=0.02
    vminVal = -5
    vmaxVal = 1

if problem == "spiral":
    driftFunction = functionBank.spiralDrift_2D
    diffusionFunction = functionBank.ptSixDiffusion
    spatialDiff = False
    kstepMin = 0.15
    kstepMax = 0.15
    endTime = 3
    h=0.05

if problem == "complex":
    driftFunction = functionBank.complextDrift_2D
    diffusionFunction = functionBank.complexDiff
    spatialDiff = True
    kstepMin = 0.1
    kstepMax = 0.1
    endTime = 1.5
    h=0.01


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "EM", integratorType = "LQ", saveHistory=False)
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

meshTrueSolnLQ = simulation.meshTrajectory[-1]
pdfTrueSolnLQ = sde.exactSolution(simulation.meshTrajectory[-1],  simulation.times[-1])

LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSolnLQ, pdfTrueSolnLQ, interpolate=False)

print(L2wErrors)

animate = True
if animate ==True:
    Meshes = simulation.meshTrajectory
    PdfTraj = simulation.pdfTrajectory

    from mpl_toolkits.mplot3d.art3d import juggle_axes
    def update_graph(num):
        indx = 0
        indy = 1
        indz = 2
        ax.clear()
        ax.set_zlim(np.min(Meshes[-1][:,indz]),np.max(Meshes[-1][:,indz]))
        ax.set_xlim(np.min(Meshes[-1][:,indx]),np.max(Meshes[-1][:,indx]))
        ax.set_ylim(np.min(Meshes[-1][:,indy]),np.max(Meshes[-1][:,indy]))
        # graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='bone_r', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")
        graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='binary', vmax=0.1, vmin=-7, marker=".")
        ax.set_title('t= %.2f' %simulation.times[num])
        return graph

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('xkcd:salmon')
    title = ax.set_title('3D Test')
    ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
    ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
    ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    # fig.colorbar(ani)

    plt.show()



from PlottingResults import plotRowSixPlots

if problem == "erf":
    plottingMax = 0.3
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [5, 15,len(simulation.meshTrajectory)-1], [-10,10,-10,10])


if problem == "spiral":
    plottingMax = 0.3
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [15, 35 ,len(simulation.meshTrajectory)-1], [-12,12,-12,12])


if problem == "complex":
    plottingMax = 0.3
    plotRowSixPlots(plottingMax, simulation.meshTrajectory, simulation.pdfTrajectory, h, [59, 99 ,len(simulation.meshTrajectory)-1], [-6,6,-6,6])


'''Compute Leja reuse and Alt method use'''
lengths = []
for mesh in simulation.meshTrajectory[1:]:
    lengths.append(len(mesh))

percentLejaReuse = np.asarray(simulation.LPReuseCount)/np.asarray(lengths)*100

print("Average LEJA REUSE Percent: ", np.mean(percentLejaReuse))

percentAltMethodUse = np.asarray(simulation.AltMethodUseCount)/np.asarray(lengths)*100
print("Average ALT METHOD USE Percent: ", np.mean(percentAltMethodUse))

"""
Created on Fri Nov 20 18:06:14 2020

@author: Rylei
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from scipy.spatial import Delaunay
import pickle
from Errors import ErrorVals
from datetime import datetime
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import ticker, cm
import time


# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '12'
fontprops = {'fontweight': 'bold'}


Meshes = simulation.meshTrajectory
PdfTraj = simulation.pdfTrajectory
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
num=-15
graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='bone_r', vmax=vmaxVal, vmin=vminVal, marker=".")

ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))
ax.set_xlabel(r'$x^{(1)}$')
ax.yaxis._axinfo['label']['space_factor'] = 22.0

ax.set_ylabel(r'$x^{(2)}$')
ax.set_zlabel(r'$x^{(3)}$')




Meshes = simulation.meshTrajectory
PdfTraj = simulation.pdfTrajectory
from mpl_toolkits.mplot3d.art3d import juggle_axes

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_facecolor('xkcd:salmon')
color = (0.5, 0.75,0.5,0.5)
ax.w_xaxis.set_pane_color(color)
ax.w_yaxis.set_pane_color(color)
ax.w_zaxis.set_pane_color(color)
index = -1
ax.set_zlim(np.min(Meshes[index][:,2]),np.max(Meshes[index][:,2]))
ax.set_xlim(np.min(Meshes[index][:,0]),np.max(Meshes[index][:,0]))
ax.set_ylim(np.min(Meshes[index][:,1]),np.max(Meshes[index][:,1]))
graph = ax.scatter3D(Meshes[index][:,0], Meshes[index][:,1],  Meshes[index][:,2], c=np.log(PdfTraj[index]), cmap='binary', vmax=0.1, vmin=-7, marker=".")

# fig.colorbar(ani)
ax.tick_params(pad=3)

plt.xlabel(r'$x^{(1)}$')
plt.ylabel(r'$x^{(2)}$')
ax.set_zlabel(r'$x^{(3)}$', rotation = 180)
plt.show()



'''Compute Leja reuse and Alt method use'''
simulation.computeLejaAndAlternativeUse()
simulation.computeTotalPointsUsed()






