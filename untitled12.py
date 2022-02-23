from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time
import sys
from Functions import get2DTrapezoidalMeshBasedOnLejaQuadratureSolutionMovingHill



# startup parameters
dimension = 2
radius = 2
h = 0.05
betaVals = [2.5]
betaToUseForMeshSizeOfTrapezoidalRule = 4
bufferVals = []
endTime = 8
spacingLQVals = [0.38]
spacingTRValsShort = []
spacingTRVals = []


# startup paramete

# dimension = 2
# radius = 2
# h = 0.05
# betaVals = [4]
# bufferVals = [0.5]
# endTime =10
# spacingLQVals = [0.38]
# spacingTRVals = [0.18]

# dimension = 2
# radius = 2
# h = 0.05
# betaVals = [3]
# bufferVals = []
# endTime = 40
# spacingLQVals = [0.38]
# spacingTRVals = []


# SDE creation
driftFunction = functionBank.oneDrift
diffusionFunction = functionBank.ptSixDiffusion
spatialDiff = False
adaptive = True
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
saveHistory = False

# Data Storage
ErrorsLQ = []
ErrorsTR = []
numPointsLQ = []
numPointsTR = []
timingArrayStorageLQ = []
timingArrayStorageTR = []

betaDict_errors = {}
betaDict_times = {}

bufferDict_errors = {}
bufferDict_times = {}


numIterations =1
original_stdout = sys.stdout # Save a reference to the original standard output
# with open('Output/outputInformationAllTimes.txt', 'w') as g:
    # sys.stdout = g
    # print("dimension: ", dimension, " Iterations: ", numIterations, " initial radius: ", radius, "endTime: ", endTime, "time step: ", h, "\n")

allTimingsArrayStorageLQ = []
allErrorsTimingArrayStorageLQ = []
for beta in betaVals:
    ErrorsLQ = []
    timingArrayStorageLQ = []
    for spacingLQ in spacingLQVals:
        print("\nLQ: beta= ", beta, ", spacing= ", spacingLQ, ":\n")
        timingPerRunArrayLQ = []
        errorsPerRunArrayLQ = []
        meshLengthsPerRunArrayLQ = []
        for iteration in range(numIterations):
            # SDE parameter creation
            parametersLQ = Parameters(sde, beta, radius, spacingLQ, spacingLQ, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ", saveHistory=saveHistory)
            simulationLQ = Simulation(sde, parametersLQ, endTime)

            startTimeLQ = time.time()
            simulationLQ.setUpTransitionMatrix(sde, parametersLQ)
            stepByStepTimingLQ = simulationLQ.computeAllTimes(sde, parametersLQ)
            totalTimeLQ = time.time() - startTimeLQ
            meshTrueSolnLQ = simulationLQ.meshTrajectory[-1]
            pdfTrueSolnLQ = sde.exactSolution(simulationLQ.meshTrajectory[-1],  simulationLQ.times[-1])

            LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationLQ.meshTrajectory[-1], simulationLQ.pdfTrajectory[-1], meshTrueSolnLQ, pdfTrueSolnLQ, interpolate=False)

            timingPerRunArrayLQ.append(np.copy(totalTimeLQ))
            errorsPerRunArrayLQ.append(np.copy(L2wErrors))
            meshLengthsPerRunArrayLQ.append(np.copy(simulationLQ.pdf.meshLength))

        print("Timing LQ: ", timingPerRunArrayLQ)
        print("Errors LQ: ", errorsPerRunArrayLQ)
        print("Mesh Size LQ: ", meshLengthsPerRunArrayLQ)

        allTimingsArrayStorageLQ.append(np.copy(np.asarray(timingPerRunArrayLQ)))
        allErrorsTimingArrayStorageLQ.append(np.copy(errorsPerRunArrayLQ))

        medianTimingLQ = np.median(np.asarray(timingPerRunArrayLQ))
        indx,  = np.where(timingPerRunArrayLQ == medianTimingLQ)[0]
        timingArrayStorageLQ.append(np.copy(medianTimingLQ))
        ErrorsLQ.append(np.copy(errorsPerRunArrayLQ[indx]))

        numPointsLQ.append(np.copy(meshLengthsPerRunArrayLQ[indx]))

    betaDict_times[beta] = np.copy(timingArrayStorageLQ)
    betaDict_errors[beta] = np.copy(ErrorsLQ)

    if beta == betaToUseForMeshSizeOfTrapezoidalRule:
        meshTrajectoryToUseForTRMeshSize =np.copy(simulationLQ.meshTrajectory)




