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
from DTQTensorized import ApproxExactSoln
import numpy as np
import Functions as fun
from scipy.spatial import Delaunay
import LejaQuadrature as LQ
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
import MeshUpdates2D as MeshUp
from pyopoly1.Scaling import GaussScale
import ICMeshGenerator as M
from pyopoly1.LejaPoints import getLejaSetFromPoints, getLejaPoints
import matplotlib.pyplot as plt


dimension =1
sde = SimpleDriftSDE(0.5,1,dimension)
mydrift = sde.Drift
mydiff = sde.Diff

def mydrift(mesh):
    # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return -1*mesh
    return mesh*(4-mesh**2)
    
def mydiff(mesh):
    return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

timeStep = [0.01, 0.05, 0.1]
EndTime = 1
kstepMin = 0.06 # lambda
kstepMax = kstepMin + 0.05 # Lambda
beta = 20
radius = 3.75# R
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 5
numPointsForLejaCandidates = 50
numQuadFit = 30

ErrorsEM = []
timesEM = []
LPE = []
AltE =[]
for i in timeStep:
    start = time.time()
    NumSteps = int(EndTime/i)
    h= i
    par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
    par.kstepMax = kstepMax
    par.radius = radius
    par.numPointsForLejaCandidates = numPointsForLejaCandidates
    par.numQuadFit = numQuadFit
    par.NumLejas = NumLejas
    TSType = "EM"
    
    Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True, TimeStepType= TSType)
    end = time.time()
    timesEM.append(end -start)

    pc = []
    for i in range(len(Meshes)-1):
        l = len(Meshes[i])
        pc.append(LPReuseArr[i]/l)
        
    mean = np.mean(pc)
    print("Leja Reuse: ", mean*100, "%")
    LPE.append(mean*100)

    
    pc = []
    for i in range(len(Meshes)-1):
        l = len(Meshes[i])
        pc.append(AltMethod[i]/l)
        
    mean2 = np.mean(pc)
    print("Alt Method: ", mean2*100, "%")
    AltE.append(mean2*100)

    
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(Meshes)):
        truepdf = sde.Solution(Meshes[i], (i+1)*h)
        # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
        
    from Errors import ErrorValsExact
    # LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
    Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
    LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
    ErrorsEM.append(L2wErrors[-1])
    
    

timeStepAM = timeStep
ErrorsAM = []
timesAM = []
LPA = []
AltA = []
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
    LPA.append(mean*100)

    
    pc = []
    for i in range(len(Meshes)-1):
        l = len(Meshes[i])
        pc.append(AltMethod[i]/l)
        
    mean2 = np.mean(pc)
    print("Alt Method: ", mean2*100, "%")
    AltA.append(mean2*100)
    
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(Meshes)):
        truepdf = sde.Solution(Meshes[i], (i+1)*h)
        # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
        
    from Errors import ErrorValsExact
    # LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
    Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
    LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
    ErrorsAM.append(L2wErrors[-1])  
    
    
    

ErrorsEMT = []
timesEMT = []
for i in timeStep:
    start = time.time()
    NumSteps = int(EndTime/i)
    h= i
    par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
    par.kstepMax = kstepMax
    par.radius = radius
    par.numPointsForLejaCandidates = numPointsForLejaCandidates
    par.numQuadFit = numQuadFit
    par.NumLejas = NumLejas
    TSType = "EM"
    
    mesh = M.NDGridMesh(dimension, kstepMin, radius, UseNoise = False)
    GMat = fun.GenerateEulerMarMatrix(len(mesh), mesh, h, mydrift, mydiff, SpatialDiff)
    scale = GaussScale(dimension)
    scale.setMu(h*mydrift(np.zeros(dimension)).T)
    scale.setCov((h*mydiff(np.zeros(dimension))*mydiff(np.zeros(dimension)).T).T)
    
    p = fun.Gaussian(scale, mesh)
    PdfTraj = []
    Meshes = []
    for i in range(NumSteps):
        Meshes.append(mesh)
        p = kstepMin*np.copy(GMat@p)
        PdfTraj.append(np.copy(p))
    
    end = time.time()
    timesEMT.append(end -start)
    
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(Meshes)):
        truepdf = sde.Solution(Meshes[i], (i+1)*h)
        # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
        
    from Errors import ErrorValsExact
    # LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
    Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
    LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
    ErrorsEMT.append(L2wErrors[-1])
    

ErrorsAMT = []
timesAMT = []
for i in timeStep:
    start = time.time()
    NumSteps = int(EndTime/i)
    h= i
    par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
    par.kstepMax = kstepMax
    par.radius = radius
    par.numPointsForLejaCandidates = numPointsForLejaCandidates
    par.numQuadFit = numQuadFit
    par.NumLejas = NumLejas
    TSType = "AM"
    
    mesh = M.NDGridMesh(dimension, kstepMin, radius, UseNoise = False)
    GMat = fun.GenerateAndersonMatMatrix(h, mydrift, mydiff, mesh, dimension, len(mesh), kstepMin)
    scale = GaussScale(dimension)
    scale.setMu(h*mydrift(np.zeros(dimension)).T)
    scale.setCov((h*mydiff(np.zeros(dimension))*mydiff(np.zeros(dimension)).T).T)
    
    p = fun.Gaussian(scale, mesh)
    PdfTraj = []
    Meshes = []
    for i in range(NumSteps):
        Meshes.append(mesh)
        p = kstepMin*np.copy(GMat@p)
        PdfTraj.append(np.copy(p))
    
    end = time.time()
    timesAMT.append(end -start)
    
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(Meshes)):
        truepdf = sde.Solution(Meshes[i], (i+1)*h)
        # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
        
    from Errors import ErrorValsExact
    # LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
    Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
    LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
    ErrorsAMT.append(L2wErrors[-1])
    
plt.figure()
plt.loglog(np.asarray(timeStep), ErrorsEM, '-o', label="EM")
plt.loglog(np.asarray(timeStepAM), ErrorsAM, '-o', label="AM")
plt.loglog(np.asarray(timeStep), ErrorsAMT, '-o', label="AM Trapezoidal")
plt.loglog(np.asarray(timeStep), ErrorsEMT, '-o', label="EM Trapezoidal")
plt.xlabel("timestep")
plt.ylabel("Error")
plt.legend()

plt.figure()
plt.plot(np.asarray(timeStep), timesEM, '-o', label="EM")
plt.plot(np.asarray(timeStepAM), timesAM, '-o', label="AM")
plt.plot(np.asarray(timeStep), timesAMT, '-o', label="AM Trapezoidal")
plt.plot(np.asarray(timeStep), timesEMT, '-o', label="EM Trapezoidal")
plt.xlabel("timestep")
plt.ylabel("Time (seconds)")
plt.legend()

plt.figure()
plt.plot(np.asarray(timeStep), LPE, '-o', label="EM Avg. Leja Reuse")
plt.plot(np.asarray(timeStepAM), LPA, '-o', label="AM Avg. Leja Reuse")
plt.plot(np.asarray(timeStep), AltE, '-o', label="EM Avg. Alt Method Use")
plt.plot(np.asarray(timeStepAM), AltA, '-o', label="AM Avg. Alt Method Use")
plt.xlabel("timestep")
plt.ylabel("Percent")
plt.legend()



def update_graph(num):
    graph.set_data(Meshes[num], PdfTraj[num])
    return title, graph

fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title('2D Test')
    
graph, = ax.plot(Meshes[-1], PdfTraj[-1], linestyle="", marker=".")
ax.set_xlim(-20, 20)
ax.set_ylim(0, np.max(PdfTraj[0]))


ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()

