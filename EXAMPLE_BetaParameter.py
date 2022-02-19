from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime

dimension =2
driftFunction = functionBank.oneDrift
diffusionFunction = functionBank.oneDiffusion
spatialDiff = False


'''Initialization Parameters'''


betaVals = [1,2,3,4,5,6,7,8,9,10]

endTime = 1.15
h=0.01
# NumSteps = int(endTime/h)

# times = np.asarray(np.arange(h,(NumSteps+0.5)*h,h))

timesArray = []
stepArray = []
table = ""

a = 1
kstepMin = 0.2 # lambda
kstepMax = 0.2 # Lambda
radius = 2 # R
SpatialDiff = False

for beta in betaVals:
    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "EM", integratorType = "LQ")
    simulation = Simulation(sde, parameters, endTime)
    simulation.setUpTransitionMatrix(sde, parameters)
    stepByStepTiming = simulation.computeAllTimes(sde, parameters)

    PdfTraj = simulation.pdfTrajectory
    Meshes = simulation.meshTrajectory
    exactSolution = sde.exactSolution(Meshes[-1],  simulation.times[-1])

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(Meshes[-1], PdfTraj[-1], Meshes[-1], exactSolution, interpolate= True)
    print(L2wErrors)

    table = table + str(beta) + "&" +str("{:2e}".format(L2wErrors))+ "&" +str("{:2e}".format(L2Errors)) + "&" +str("{:2e}".format(L1Errors)) + "&" +str("{:2e}".format(LinfErrors))  + "&" + str(len(Meshes[-1])) + "\\\ \hline "



