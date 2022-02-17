from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time
from PlottingResults import plotRowSixPlots
from exactSolutions import Solution
from Errors import ErrorValsExact
from Errors import ErrorValsOneTime

from matplotlib import rcParams

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '18'
fontprops = {'fontweight': 'bold'}



dimension =1
beta = 4
radius = 2
kstepMin= 0.4
kstepMax = kstepMin
h = 0.05
endTime =10

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.twoDrift
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

meshTrueSoln = simulation.meshTrajectory[-1]
pdfTrueSoln = sde.exactSolution(simulation.meshTrajectory[-1], endTime)
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], meshTrueSoln, pdfTrueSoln, interpolate=False)
print(L2wErrors)

plt.figure()
indices = [0,49,99,149,199]
# indices = [199,149,99,49,0]



for ind in indices:
    time = (ind+1)*h
    labelString = 't = %.2f' % time
    plt.plot(simulation.meshTrajectory[ind], simulation.pdfTrajectory[ind], '.', label=labelString)

plt.xlim([-5, 35])
plt.xlabel(r'$\mathbf{x}$')
plt.ylabel(r'$\hat{p}(\mathbf{x}, t_n)$')

plt.legend(markerscale=2)



Animate = False
if Animate:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
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

simulation.computeLejaAndAlternativeUse()
