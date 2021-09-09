from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Class_Parameters as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from NDFunctionBank import SimpleDriftSDE
import time
from DTQTensorized import ApproxExactSoln

dimension =1
sde = SimpleDriftSDE(0.5,0.5,dimension)
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

timeStep = [0.1]
EndTime = 1
kstepMin = 0.06 # lambda
kstepMax = kstepMin + 0.005 # Lambda
beta = 20
radius = 3# R
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 10
numPointsForLejaCandidates = 150
numQuadFit = 50

ErrorsEM = []
timesEM = []
LPReuses = []
AltMethodUses = []
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
    LPReuses.append(mean*100)
    
    pc = []
    for i in range(len(Meshes)-1):
        l = len(Meshes[i])
        pc.append(AltMethod[i]/l)
        
    mean2 = np.mean(pc)
    print("Alt Method: ", mean2*100, "%")
    AltMethodUses.append(mean2*100)
    
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(Meshes)):
        truepdf = sde.Solution(Meshes[i], (i+1)*h)
        # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
        
    
from Errors import ErrorValsExact
Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
LinfErrors, L2Errors, L1Errors, L2wErrors, approxPDF = ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)

for i in range(len(Meshes)):
    truepdf = sde.Solution(Meshes[i], (i+1)*h)
    trueSoln.append(np.squeeze(np.copy(truepdf)))

LinfErrorsT, L2ErrorsT, L1ErrorsT, L2wErrorsT = ErrorValsExact(Meshes, approxPDF, trueSoln, h, plot=True)
plt.semilogy(np.squeeze(np.asarray(Meshes[-1])), np.asarray(approxPDF[-1]))
plt.semilogy(np.squeeze(np.asarray(Meshes[-1])), np.asarray(trueSoln[-1])-np.asarray(approxPDF[-1]))
