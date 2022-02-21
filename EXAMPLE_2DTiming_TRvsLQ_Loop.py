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
betaVals = [2.5, 3, 4, 5, 6]
betaToUseForMeshSizeOfTrapezoidalRule = 4
bufferVals = [0, 0.5]
endTime = 40
spacingLQVals = [0.38]
spacingTRValsShort = [0.25, 0.2, 0.18, 0.15]
spacingTRVals = [0.25, 0.2, 0.18]


# startup parameters
dimension = 2
radius = 2
h = 0.05
betaVals = [4]
betaToUseForMeshSizeOfTrapezoidalRule = 4
bufferVals = [0, 0.5]
endTime = 40
spacingLQVals = [0.38]
spacingTRValsShort = [0.18]
spacingTRVals = [0.18]


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


allTimingsArrayStorageTR = []
allErrorArrayStorageTR = []
for bufferVal in bufferVals:
    if bufferVal == 0:
        spacingTRValsToUse = spacingTRValsShort
    else:
        spacingTRValsToUse = spacingTRVals
    ErrorsTR = []
    timingArrayStorageTR = []
    for spacingTR in spacingTRValsToUse:
        print("\nTR: buffer= ", bufferVal, ", spacing= ", spacingTR, ":\n")
        timingPerRunArrayTR = []
        errorsPerRunArrayTR = []
        meshLengthsPerRunArrayTR = []
        for iteration in range(numIterations):
            # meshTR = get2DTrapezoidalMeshBasedOnLejaQuadratureSolutionMovingHill(meshTrajectoryToUseForTRMeshSize, spacingTR, bufferVal)
            meshTR = get2DTrapezoidalMeshBasedOnLejaQuadratureSolutionMovingHill(meshTrajectoryToUseForTRMeshSize, spacingTR, bufferVal)

            parametersTR = Parameters(sde, beta, radius, spacingTR, spacingTR, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = meshTR, saveHistory=saveHistory)

            simulationTR = Simulation(sde, parametersTR, endTime)
            startTimeTR = time.time()
            simulationTR.setUpTransitionMatrix(sde, parametersTR)

            stepByStepTimingTR = simulationTR.computeAllTimes(sde, parametersTR)
            totalTimeTR = time.time() - startTimeTR

            meshTrueSolnTR = simulationTR.meshTrajectory[-1]
            pdfTrueSolnTR = sde.exactSolution(simulationTR.meshTrajectory[-1],  simulationTR.times[-1])
            LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationTR.meshTrajectory[-1], simulationTR.pdfTrajectory[-1], meshTrueSolnTR, pdfTrueSolnTR, interpolate=False)
            allErrorArrayStorageTR.append(L2wErrors)

            timingPerRunArrayTR.append(np.copy(totalTimeTR))
            errorsPerRunArrayTR.append(np.copy(L2wErrors))
            meshLengthsPerRunArrayTR.append(np.copy(simulationTR.pdf.meshLength))

        print("Timing TR: ", timingPerRunArrayTR)
        print("Errors TR: ", errorsPerRunArrayTR)
        print("Mesh Size TR: ", meshLengthsPerRunArrayTR)

        allTimingsArrayStorageTR.append(np.copy(np.asarray(timingPerRunArrayTR)))
        medianTimingTR = np.median(np.asarray(timingPerRunArrayTR))

        numPointsTR.append(np.copy(simulationTR.pdf.meshLength))

        timingArrayStorageTR.append(np.copy(medianTimingTR))
        ErrorsTR.append(np.copy(errorsPerRunArrayTR[0]))


    bufferDict_times[bufferVal] = np.copy(timingArrayStorageTR)
    bufferDict_errors[bufferVal] = np.copy(ErrorsTR)


timestr = time.strftime("%Y%m%d-%H%M%S")
ListToSave = [betaDict_times, betaDict_errors, bufferDict_times, bufferDict_errors, betaVals, bufferVals, spacingLQVals, spacingTRVals, numPointsLQ, numPointsTR, h, radius, endTime, allTimingsArrayStorageLQ, allErrorsTimingArrayStorageLQ, allTimingsArrayStorageTR, allErrorArrayStorageTR]
import pickle

# define dictionary
# create a binary pickle file
f = open('Output/fileT40_'+str(timestr)+ "_" + str(endTime)+ '.pkl',"wb")
pickle.dump(ListToSave,f)
f.close()


original_stdout = sys.stdout # Save a reference to the original standard output
with open('Output/outputInformationSummaryT40_' +str(timestr)+ "_" +str(endTime)+ '.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("dimension: ", dimension, " Iterations: ", numIterations, " initial radius: ", radius, "endTime: ", endTime, "time step: ", h, "\n")
    print("Erorrs LQ", betaDict_errors)
    print("Errors TR", bufferDict_errors)
    print("LQ timing", betaDict_times)
    print("TR timing", bufferDict_times)
    print("# points LQ", numPointsLQ)
    print("# points TR", numPointsTR)
    sys.stdout = original_stdout # Reset the standard output to its original value


unitTime = np.asarray(betaDict_times[min(betaVals)])[0]
unitError = np.asarray(betaDict_errors[min(betaVals)])[0]
plt.figure()
plt.plot(unitError, unitTime/unitTime, "*k", markeredgewidth=1, markersize = "20",markerfacecolor="None", label = "Unit Time")

plt.semilogx(np.asarray(list(betaDict_errors.values())), np.asarray(list(betaDict_times.values()))/unitTime, "o-", label= "Adaptive LQ")
# for betaVal in betaVals:
#     if betaVal in betaDict_errors:
#         Errors = betaDict_errors[betaVal]
#         timing = betaDict_times[betaVal]
#         labelString = 'LQ, \u03B2 = %.2f' %betaVal
#         plt.semilogx(np.asarray(Errors), np.asarray(timing)/unitTime, "o", label= labelString)

for buff in bufferVals:
    if buff in bufferDict_errors:
        Errors = bufferDict_errors[buff]
        timing = bufferDict_times[buff]
        if buff == 0:
            labelString = 'Equispaced Trapezoidal Oracle, buffer = %d%%' %(buff*100)
        else:
            labelString = 'Equispaced Trapezoidal, buffer = %d%%' %(buff*100)
        plt.semilogx(np.asarray(Errors), np.asarray(timing)/unitTime, "-s", label= labelString)



plt.legend()
plt.xlabel(r'$L_{2w}$ Error')
plt.ylabel("Relative Running Time")

plt.savefig('Output/timingFigureT40_'+ str(timestr)+ "_" + str(endTime)+ '.png')

animate = False
if animate:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    simulation = simulationLQ
    if dimension ==1:
        def update_graph(num):
            graph.set_data(simulation.meshTrajectory[num], simulation.pdfTrajectory[num])
            return title, graph

        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = ax.set_title('2D Test')

        graph, = ax.plot(simulation.meshTrajectory[-1], simulation.pdfTrajectory[-1], linestyle="", marker=".")
        ax.set_xlim(-40, 40)
        ax.set_ylim(0, np.max(simulation.pdfTrajectory[0]))
        ani = animation.FuncAnimation(fig, update_graph, frames=len(simulation.pdfTrajectory), interval=50, blit=False)
        plt.show()

    if dimension ==2:
        Meshes = simulation.meshTrajectory
        PdfTraj = simulation.pdfTrajectory
        def update_graph(num):
            graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
            graph.set_3d_properties(PdfTraj[num])
            title.set_text('3D Test, time={}'.format(num))
            return title, graph

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('3D Test')

        graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
        ax.set_zlim(0, 0.5)
        ax.set_xlim(-55, 55)
        ax.set_ylim(-55, 55)


        ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=10, blit=False)
        plt.show()





