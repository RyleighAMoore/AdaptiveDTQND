from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import DriftDiffusionFunctionBank as functionBank
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact

dimension =2
driftFunction = functionBank.oneDrift
diffusionFunction = functionBank.oneDiffusion
spatialDiff = False


'''Initialization Parameters'''


betaVals = [1,2,3,4,5,6,7,8,9,10]
betaVals = [1,3,6,10]

endTime = 1.15
h=0.01
NumSteps = int(endTime/h)

times = np.asarray(np.arange(h,(NumSteps+0.5)*h,h))

L2ErrorArray = np.zeros((len(betaVals),len(times)))
LinfErrorArray = np.zeros((len(betaVals),len(times)))
L1ErrorArray = np.zeros((len(betaVals),len(times)))
L2wErrorArray = np.zeros((len(betaVals),len(times)))
timesArray = []
stepArray = []
count = 0
table = ""

a = 1
kstepMin = 0.2 # lambda
kstepMax = 0.2 # Lambda
# beta = 3
radius = 2 # R
SpatialDiff = False

for beta in betaVals:
    sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)
    parameters = Parameters(sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh =True, timeDiscretizationType = "EM", integratorType = "LQ")
    simulation = Simulation(sde, parameters, endTime)
    simulation.setUpTransitionMatrix(sde, parameters)
    stepByStepTiming = simulation.computeAllTimes(sde, parameters)

    surfaces = []
    PdfTraj = simulation.pdfTrajectory
    Meshes = simulation.meshTrajectory
    for ii in range(len(PdfTraj)):
        ana = TwoDdiffusionEquation(Meshes[ii],diffusionFunction(np.asarray([0,0]))[0,0], h*(ii+1), driftFunction(np.asarray([0,0]))[0,0])
        surfaces.append(ana)

    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, surfaces,h, plot=False)


    table = table + str(beta) + "&" +str("{:2e}".format(L2wErrors[-1]))+ "&" +str("{:2e}".format(L2Errors[-1])) + "&" +str("{:2e}".format(L1Errors[-1])) + "&" +str("{:2e}".format(LinfErrors[-1]))  + "&" + str(len(Meshes[-1])) + "\\\ \hline "
    L2ErrorArray[count,:] = np.asarray(L2Errors)
    LinfErrorArray[count,:] = np.asarray(LinfErrors)
    L1ErrorArray[count,:] = np.asarray(L1Errors)
    L2wErrorArray[count,:] = np.asarray(L2wErrors)
    for j in times:
        timesArray.append(j)
    stepArray.append(beta)
    count = count+1


X,Y = np.meshgrid(times,beta)
fig = plt.figure()
plt.semilogy(times, L2wErrorArray[-1], c='r', marker='.')
plt.semilogy(times, LinfErrorArray[-1], c='r', marker='.')
plt.show()

from matplotlib import rcParams
# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '12'
fontprops = {'fontweight': 'bold'}

plt.figure()
count = 0
for k in betaVals:
    print(count)
    # plt.semilogy(x, LinfErrorArray[k,:], label = 'Linf Error')
    # plt.semilogy(x, L2Errors[k,:], label = 'L2 Error')
    # plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
    plt.semilogy(times, L2wErrorArray[count,:], label = r'$\beta = %d$' %stepArray[count])
    plt.xlabel('Time')
    plt.ylabel(r'$L_{2w}$ Error')
    plt.legend()
    count = count+1