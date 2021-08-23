from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from NDFunctionBank import SimpleDriftSDE
import time

dimension =1
sde = SimpleDriftSDE(0,0.5,dimension)
mydrift = sde.Drift
mydiff = sde.Diff

# def mydrift(mesh):
#     # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return -1*mesh
#     return mesh*(4-mesh**2)
    
# def mydiff(mesh):
#     return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

timeStep = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
EndTime = 0.5

kstepMin = 0.05 # lambda
kstepMax = kstepMin + 0.005 # Lambda
beta = 20
radius = 4# R
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 10
numPointsForLejaCandidates = 50
numQuadFit = 50

ErrorsEM = []
timesEM = []
# for i in timeStep:
#     start = time.time()
#     NumSteps = int(EndTime/i)
#     h= i
#     par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
#     par.kstepMin = kstepMin
#     par.kstepMax = kstepMax
#     par.radius = radius
#     par.numPointsForLejaCandidates = numPointsForLejaCandidates
#     par.numQuadFit = numQuadFit
#     par.NumLejas = NumLejas
#     TSType = "EM"
    
#     Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True, TimeStepType= TSType)
#     end = time.time()
#     timesEM.append(end -start)

#     pc = []
#     for i in range(len(Meshes)-1):
#         l = len(Meshes[i])
#         pc.append(LPReuseArr[i]/l)
        
#     mean = np.mean(pc)
#     print("Leja Reuse: ", mean*100, "%")
    
#     pc = []
#     for i in range(len(Meshes)-1):
#         l = len(Meshes[i])
#         pc.append(AltMethod[i]/l)
        
#     mean2 = np.mean(pc)
#     print("Alt Method: ", mean2*100, "%")
    
#     trueSoln = []
#     from exactSolutions import OneDdiffusionEquation
#     for i in range(len(Meshes)):
#         truepdf = sde.Solution(Meshes[i], (i+1)*h)
#         # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
#         # truepdf = solution(xvec,-1,T)
#         trueSoln.append(np.squeeze(np.copy(truepdf)))
        
#     from Errors import ErrorValsExact
#     LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, plot=False)
#     ErrorsEM.append(L2wErrors[-1])
    
#     # def update_graph(num):
#     #     graph.set_data(Meshes[num], PdfTraj[num])
#     #     return title, graph

#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # title = ax.set_title('2D Test')
        
#     # graph, = ax.plot(Meshes[-1], PdfTraj[-1], linestyle="", marker=".")
#     # ax.set_xlim(-20, 20)
#     # ax.set_ylim(0, np.max(PdfTraj[0]))
    
    
#     # ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
#     # plt.show()

timeStepAM = [0.7, 0.9, 1.2]
timeStepAM = [0.1]

ErrorsAM = []
timesAM = []
for i in timeStepAM:
    start = time.time()

    NumSteps = int(EndTime/i)
    '''Discretization Parameters'''
    h= i
    
    par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
    par.kstepMax = kstepMax
    par.radius = radius
    par.numPointsForLejaCandidates = numPointsForLejaCandidates
    par.numQuadFit = numQuadFit
    par.NumLejas = NumLejas
    TSType = "AM"
    
    Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True, TimeStepType= TSType)
    end = time.time()
    timesAM.append(end - start)


    pc = []
    for i in range(len(Meshes)-1):
        l = len(Meshes[i])
        pc.append(LPReuseArr[i]/l)
        
    mean = np.mean(pc)
    print("Leja Reuse: ", mean*100, "%")
    
    pc = []
    for i in range(len(Meshes)-1):
        l = len(Meshes[i])
        pc.append(AltMethod[i]/l)
        
    mean2 = np.mean(pc)
    print("Alt Method: ", mean2*100, "%")
    
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(Meshes)):
        truepdf = sde.Solution(Meshes[i], (i+1)*h)
        # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
        
    from Errors import ErrorValsExact
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
    ErrorsAM.append(L2wErrors[-1])

plt.semilogy(np.asarray(timeStep), ErrorsEM, label="EM")
plt.semilogy(np.asarray(timeStepAM), ErrorsAM, '*', label="AM")
plt.xlabel("timestep")
plt.ylabel("Error")
plt.legend()

plt.plot(np.asarray(timeStep), timesEM, label="EM")
plt.plot(np.asarray(timeStepAM), timesAM, label="AM")
plt.xlabel("timestep")
plt.ylabel("Time (seconds)")
plt.legend()

