from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
import time

dimension =2

if dimension ==1:
    beta = 4
    radius = 4
    kstepMin= 0.06
    kstepMax = 0.07
    h = 0.1
    endTime =4

if dimension ==2:
    beta = 3
    radius =0.5
    kstepMin= 0.08
    kstepMax = 0.09
    h = 0.05
    endTime = 0


if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01
    endTime = 0.1

# driftFunction = functionBank.zeroDrift
driftFunction = functionBank.erfDrift
# driftFunction = functionBank.oneDrift


diffusionFunction = functionBank.oneDiffusion

spatialDiff = False
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
numTimes = 2
rvals = [1,10,20,50,80,100, 150]
times = []
meshSize = []
for radius in rvals:
    parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "AM", integratorType = "LQ")
    simulation = Simulation(sde, parameters, endTime)
    start = time.time()
    for i in range(numTimes):
        simulation.timeDiscretizationMethod.computeTransitionMatrix(simulation.pdf, sde, h)
    end = time.time()
    print((end-start)/numTimes)
    times.append((end-start)/numTimes)
    meshSize.append(simulation.pdf.meshLength)

plt.figure()
plt.loglog(np.asarray(meshSize), np.asarray(times))
plt.show()

