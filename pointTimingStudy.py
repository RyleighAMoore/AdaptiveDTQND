from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension =2
if dimension ==1:
    beta = 3
    radius = 10
    kstepMin= 0.06
    kstepMax = 0.065
    # h = 0.01
    endTime =0

if dimension ==2:
    beta = 4
    radius =1.5
    kstepMin= 0.08
    kstepMax = 0.09
    # h = 0.05
    endTime = 0

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    # h = 0.01
    endTime = 0.1

# driftFunction = functionBank.zeroDrift
driftFunction = functionBank.erfDrift
# driftFunction = functionBank.oneDrift

spatialDiff = False


diffusionFunction = functionBank.oneDiffusion

ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
timesNoStartupEM = []
timesNoStartupAM = []
adaptive = False


sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)

rvals = [0.1, 1, 5, 10,20,50]
rvals = [0.2, 0.5, 1, 1.2]

meshLengths = []
for radius in rvals:
    h=0.1
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM")
    startEM = time.time()
    simulationEM = Simulation(sde, parametersEM, endTime)

    startEMNoStartup = time.time()
    simulationEM.computeAllTimes(sde, simulationEM.pdf, parametersEM)
    endEM = time.time()
    timesEM.append(np.copy(endEM-startEM))
    timesNoStartupEM.append(np.copy(endEM-startEMNoStartup))

    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM")
    startAM = time.time()
    simulationAM = Simulation(sde, parametersAM, endTime)
    startAMNoStartup = time.time()
    simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
    endAM =time.time()
    timesAM.append(np.copy(endAM-startAM))
    timesNoStartupAM.append(np.copy(endAM-startAMNoStartup))

    assert simulationAM.pdf.meshLength == simulationEM.pdf.meshLength
    meshLengths.append(simulationAM.pdf.meshLength)

    # del simulationAM



plt.figure()
plt.loglog(np.asarray(meshLengths), np.asarray(timesEM),label= "EM")
plt.loglog(np.asarray(meshLengths), np.asarray(timesAM), label="AM")
plt.xlabel("# points")
plt.ylabel("Time (seconds)")
plt.title("1D Transition Matrix Formation Cost")

plt.legend()




