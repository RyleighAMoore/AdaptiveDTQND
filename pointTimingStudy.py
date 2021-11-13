from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension =1

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

rvals = [80, 40, 20, 10, 5]
# rvals = [2,1.5, 1, 0.5, 0.2]

meshLengths = []
for radius in rvals:
    h=0.1
    parametersEM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM")
    simulationEM = Simulation(sde, parametersEM, endTime)

    startEMNoStartup = time.time()
    simulationEM.timeDiscretizationMethod.computeTransitionMatrix(simulationEM.pdf, sde, parametersEM)
    endEM = time.time()
    timesNoStartupEM.append(np.copy(endEM-startEMNoStartup))

    parametersAM = Parameters(sde, beta, radius, kstepMin, kstepMax, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "AM")
    simulationAM = Simulation(sde, parametersAM, endTime)
    startAMNoStartup = time.time()
    simulationAM.timeDiscretizationMethod.computeTransitionMatrix(simulationAM.pdf, sde, parametersAM)
    endAM = time.time()
    timesNoStartupAM.append(np.copy(endAM-startAMNoStartup))

    assert simulationAM.pdf.meshLength == simulationEM.pdf.meshLength
    meshLengths.append(simulationAM.pdf.meshLength)

    # del simulationAM


plt.figure()
plt.loglog(np.asarray(meshLengths), np.asarray(timesNoStartupEM), 'o', label= "EM")
plt.loglog(np.asarray(meshLengths), np.asarray(timesNoStartupAM), 'o', label="AM")
plt.xlabel("# points in mesh")
plt.ylabel("Time (seconds)")
plt.title("1D Transition Matrix Formation Cost")

plt.legend()
plt.show()
# plt.savefig('result.png')




