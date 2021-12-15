from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

# startup parameters
dimension = 2
radius = 2
h = 0.05
beta = 3
bufferVals = [0,0.5]
endTime = 20
spacingLQ = 0.38
spacingTR = 0.2


# SDE creation
driftFunction = functionBank.oneDrift
diffusionFunction = functionBank.ptSixDiffusion
spatialDiff = False
adaptive = True
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)

# SDE parameter creation
parametersLQ = Parameters(sde, beta, radius, spacingLQ, spacingLQ+0.1, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ")


# Data Storage
ErrorsLQ = []
ErrorsTR = []
numPointsLQ = []
numPointsTR = []
stepByStepTimingArrayStorageLQ = []
stepByStepTimingArrayStorageTR = []

def get2DTrapezoidalMeshBasedOnLejaQuadratureSolution(simulationLQ, bufferVal = 0):
    xmin = min(np.min(simulationLQ.meshTrajectory[-1][:,0]),np.min(simulationLQ.meshTrajectory[0][:,0]))
    xmax = max(np.max(simulationLQ.meshTrajectory[-1][:,0]),np.max(simulationLQ.meshTrajectory[0][:,0]))
    ymin = min(np.min(simulationLQ.meshTrajectory[-1][:,1]),np.min(simulationLQ.meshTrajectory[0][:,1]))
    ymax = max(np.max(simulationLQ.meshTrajectory[-1][:,1]),np.max(simulationLQ.meshTrajectory[0][:,1]))

    bufferX =bufferVal*(xmax-xmin)/2
    bufferY = bufferVal*(ymax-ymin)/2
    xstart = np.floor(xmin) - bufferX
    xs = []
    xs.append(xstart)
    while xstart< xmax + bufferX:
        xs.append(xstart+spacingTR)
        xstart += spacingTR

    ystart = np.floor(ymin) - bufferY
    ys = []
    ys.append(ystart)

    while ystart< ymax+ bufferY:
        ys.append(ystart+spacingTR)
        ystart += spacingTR

    mesh = []
    for i in xs:
        for j in ys:
            mesh.append([i,j])
    mesh = np.asarray(mesh)

    return mesh


startTimeLQ = time.time()
simulationLQ = Simulation(sde, parametersLQ, endTime)
TransitionMatrixCreationTimeLQ = time.time() - startTimeLQ

stepByStepTimingLQ = simulationLQ.computeAllTimes(sde, simulationLQ.pdf, parametersLQ)
for i in range(len(stepByStepTimingLQ)):
    stepByStepTimingLQ[i] += TransitionMatrixCreationTimeLQ

stepByStepTimingLQ.insert(0, TransitionMatrixCreationTimeLQ)

meshTrueSolnLQ = simulationLQ.meshTrajectory[-1]
pdfTrueSolnLQ = sde.exactSolution(simulationLQ.meshTrajectory[-1], endTime)

LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationLQ.meshTrajectory[-1], simulationLQ.pdfTrajectory[-1], meshTrueSolnLQ, pdfTrueSolnLQ, interpolate=False)
ErrorsLQ.append(np.copy(L2wErrors))
stepByStepTimingArrayStorageLQ.append(np.copy(np.asarray(stepByStepTimingLQ)))
numPointsLQ.append(np.copy(simulationLQ.pdf.meshLength))
times = np.asarray(list(range(1,len(stepByStepTimingLQ)+1)))*h

for bufferVal in bufferVals:
    meshTR = get2DTrapezoidalMeshBasedOnLejaQuadratureSolution(simulationLQ, bufferVal)
    parametersTR = Parameters(sde, beta, radius, spacingTR, spacingTR, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = meshTR)

    startTimeTR = time.time()
    simulationTR = Simulation(sde, parametersTR, endTime)
    TransitionMatrixCreationTimeTR = time.time()-startTimeTR

    stepByStepTimingTR = simulationTR.computeAllTimes(sde, simulationTR.pdf, parametersTR)
    for i in range(len(stepByStepTimingTR)):
        stepByStepTimingTR[i] += TransitionMatrixCreationTimeTR
    stepByStepTimingTR.insert(0, TransitionMatrixCreationTimeTR)

    meshTrueSolnTR = simulationTR.meshTrajectory[-1]
    pdfTrueSolnTR = sde.exactSolution(simulationTR.meshTrajectory[-1], endTime)
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationTR.meshTrajectory[-1], simulationTR.pdfTrajectory[-1], meshTrueSolnTR, pdfTrueSolnTR, interpolate=False)
    ErrorsTR.append(np.copy(L2wErrors))
    numPointsTR.append(np.copy(simulationTR.pdf.meshLength))
    stepByStepTimingArrayStorageTR.append(np.copy(np.asarray(stepByStepTimingTR)))


plt.plot()
plt.loglog(times, stepByStepTimingArrayStorageLQ[0],'o', label= "LQ")
plt.loglog(times, stepByStepTimingArrayStorageTR[1],'o', label= "TR, 0.5% padding")
plt.loglog(times, stepByStepTimingArrayStorageTR[0],'o', label= "TR, 0% padding")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Total Time")
plt.savefig('timingFigure.png')

import sys
original_stdout = sys.stdout # Save a reference to the original standard output
with open('outputInformation.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("Erorrs LQ", ErrorsLQ)
    print("Errors TR", ErrorsTR)
    print("# points LQ", numPointsLQ)
    print("# points TR", numPointsTR)
    print("Step By Step times LQ", stepByStepTimingArrayStorageLQ)
    print("Step By Step times TR", stepByStepTimingArrayStorageTR)
    print("times", times)
    sys.stdout = original_stdout # Reset the standard output to its original value

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# simulation = simulationTR
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
#     # ax.set_zlim(0, 1.5)
#     ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=10, blit=False)
#     plt.show()



