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


# startup parameters
dimension = 2
radius = 2
h = 0.05
betaVals = [3, 4]
bufferVals = [0, 0.5]
endTime =5
spacingLQVals = [0.38]
spacingTRVals = [0.25, 0.2, 0.18]


# dimension = 2
# radius = 2
# h = 0.05
# betaVals = [4]
# bufferVals = [0.5]
# endTime =20
# spacingLQVals = [0.38]
# spacingTRVals = [0.18]


# SDE creation
driftFunction = functionBank.oneDrift
diffusionFunction = functionBank.ptSixDiffusion
spatialDiff = False
adaptive = True
sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
saveHistory = False

# Data Storage
betaDict_L2werrors = {}
betaDict_L1errors = {}
betaDict_Linferrors = {}
betaDict_L2errors = {}
betaDict_times = {}
betaDict_NumPoints = {}

bufferDict_L2werrors = {}
bufferDict_L1errors = {}
bufferDict_Linferrors = {}
bufferDict_L2errors = {}
bufferDict_times = {}
bufferDict_NumPoints = {}



def get2DTrapezoidalMeshBasedOnLejaQuadratureSolution(simulationLQ, spacingTR, bufferVal = 0):
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


def runLQAdaptiveDTQ(beta, spacingLQ):
    '''setup'''
    parametersLQ = Parameters(sde, beta, radius, spacingLQ, spacingLQ, h,useAdaptiveMesh =adaptive, timeDiscretizationType = "EM", integratorType="LQ", saveHistory=saveHistory)
    simulationLQ = Simulation(sde, parametersLQ, endTime) # Also sets mesh

    '''Run Procedure and time'''
    startTimeLQ = time.time()
    simulationLQ.setUpTransitionMatrix(sde, parametersLQ)
    stepByStepTimingLQ = simulationLQ.computeAllTimes(sde, parametersLQ)
    totalTimeLQ = time.time() - startTimeLQ

    '''Compute Errors'''
    meshTrueSolnLQ = simulationLQ.meshTrajectory[-1]
    pdfTrueSolnLQ = sde.exactSolution(simulationLQ.meshTrajectory[-1], endTime)
    LinfError, L2Error, L1Error, L2wError = ErrorValsOneTime(simulationLQ.meshTrajectory[-1], simulationLQ.pdfTrajectory[-1], meshTrueSolnLQ, pdfTrueSolnLQ, interpolate=False)

    numPoints = len(simulationLQ.pdfTrajectory[-1])

    return LinfError, L2Error, L1Error, L2wError, totalTimeLQ , numPoints, simulationLQ

def runTRDTQ(buffer, spacingTR, simulationLQ):
    meshTR = get2DTrapezoidalMeshBasedOnLejaQuadratureSolution(simulationLQ, spacingTR, buffer)
    parametersTR = Parameters(sde, beta, radius, spacingTR, spacingTR, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = meshTR, saveHistory=saveHistory)

    simulationTR = Simulation(sde, parametersTR, endTime)

    startTimeTR = time.time()
    simulationTR.setUpTransitionMatrix(sde, parametersTR)
    stepByStepTimingTR = simulationTR.computeAllTimes(sde, parametersTR)
    totalTimeTR = time.time() - startTimeTR

    meshTrueSolnTR = simulationTR.meshTrajectory[-1]
    pdfTrueSolnTR = sde.exactSolution(simulationTR.meshTrajectory[-1], endTime)
    LinfError, L2Error, L1Error, L2wError = ErrorValsOneTime(simulationTR.meshTrajectory[-1], simulationTR.pdfTrajectory[-1], meshTrueSolnTR, pdfTrueSolnTR, interpolate=False)
    numPoints = len(simulationTR.pdfTrajectory[-1])

    return LinfError, L2Error, L1Error, L2wError, totalTimeTR , numPoints


numIterations =3
original_stdout = sys.stdout # Save a reference to the original standard output
# with open('Output/outputInformationAllTimes.txt', 'w') as g:
    # sys.stdout = g
    # print("dimension: ", dimension, " Iterations: ", numIterations, " initial radius: ", radius, "endTime: ", endTime, "time step: ", h, "\n")
for count in range(numIterations):
    for beta in betaVals:
        for spacingLQ in spacingLQVals:
            print("\nLQ: beta= ", beta, ", spacing= ", spacingLQ, ":\n")

            LinfError, L2Error, L1Error, L2wError, totalTimeLQ, numPoints, simulationLQ = runLQAdaptiveDTQ(beta, spacingLQ)

            if (beta, spacingLQ) not in betaDict_times:
                betaDict_times[(beta, spacingLQ)] = list()
                betaDict_L2werrors[(beta, spacingLQ)]= list()
                betaDict_L1errors[(beta, spacingLQ)]= list()
                betaDict_Linferrors[(beta, spacingLQ)]= list()
                betaDict_L2errors[(beta, spacingLQ)]= list()
                betaDict_NumPoints[(beta, spacingLQ)]= list()

            betaDict_times[(beta, spacingLQ)].append(np.copy(totalTimeLQ))
            betaDict_L2werrors[(beta, spacingLQ)].append(np.copy(L2wError))
            betaDict_L1errors[(beta, spacingLQ)].append(np.copy(L1Error))
            betaDict_Linferrors[(beta, spacingLQ)].append(np.copy(LinfError))
            betaDict_L2errors[(beta, spacingLQ)].append(np.copy(L2Error))
            betaDict_NumPoints[(beta, spacingLQ)].append(np.copy(numPoints))


    for buffer in bufferVals:
        for spacingTR in spacingTRVals:
            LinfError, L2Error, L1Error, L2wError, totalTime , numPoints = runTRDTQ(buffer, spacingTR, simulationLQ)

            if (buffer, spacingTR) not in bufferDict_times:
                bufferDict_times[(buffer, spacingTR)] =  list()
                bufferDict_L2werrors[(buffer, spacingTR)] = list()
                bufferDict_L1errors[(buffer, spacingTR)]=  list()
                bufferDict_Linferrors[(buffer, spacingTR)]=  list()
                bufferDict_L2errors[(buffer, spacingTR)]=  list()
                bufferDict_NumPoints[(buffer, spacingTR)]=  list()

            bufferDict_times[(buffer, spacingTR)].append(np.copy(totalTime))
            bufferDict_L2werrors[(buffer, spacingTR)].append(np.copy(L2wError))
            bufferDict_L1errors[(buffer, spacingTR)].append(np.copy(L1Error))
            bufferDict_Linferrors[(buffer, spacingTR)].append(np.copy(LinfError))
            bufferDict_L2errors[(buffer, spacingTR)].append(np.copy(L2Error))
            bufferDict_NumPoints[(buffer, spacingTR)].append(np.copy(numPoints))

ListToSave = [betaDict_L2werrors,
betaDict_L1errors,
betaDict_Linferrors ,
betaDict_L2errors ,
betaDict_times ,
betaDict_NumPoints ,
bufferDict_L2werrors,
bufferDict_L1errors ,
bufferDict_Linferrors ,
bufferDict_L2errors ,
bufferDict_times ,
bufferDict_NumPoints,dimension,
radius,h,
betaVals,
bufferVals,
endTime,
spacingLQVals,
spacingTRVals]

import pickle
timestr = time.strftime("%Y%m%d-%H%M%S")
f = open('Output/fileT40_'+str(timestr)+ "_" + str(endTime)+ '.pkl',"wb")
pickle.dump(ListToSave,f)
f.close()
