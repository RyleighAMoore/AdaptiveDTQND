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
from DTQTensorized import ApproxExactSoln, ApproxExactSolnDense
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

kstepMin = [0.001, 0.005, 0.01, 0.05, 0.1]
# kstepMin = [0.001]

EndTime = 1
beta = 20
radius = 5# R
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 5
numPointsForLejaCandidates = 50
numQuadFit = 30
h = 0.0001


ErrorsEMT = []
timesEMT = []
for i in kstepMin:
    start = time.time()
    NumSteps = int(1/h)
    par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
    par.radius = radius
    par.numPointsForLejaCandidates = numPointsForLejaCandidates
    par.numQuadFit = numQuadFit
    par.NumLejas = NumLejas
    TSType = "EM"
    
    mesh = M.NDGridMesh(dimension, i, radius, UseNoise = False)
    GMat = fun.GenerateEulerMarMatrix(len(mesh), mesh, h, mydrift, mydiff, SpatialDiff)
    scale = GaussScale(dimension)
    scale.setMu(h*mydrift(np.zeros(dimension)).T)
    scale.setCov((h*mydiff(np.zeros(dimension))*mydiff(np.zeros(dimension)).T).T)

    p = fun.Gaussian(scale, mesh)
    PdfTraj = []
    Meshes = []
    for j in range(NumSteps):
        Meshes.append(mesh)
        p = i*np.copy(GMat@p)
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
    Times = []
    for i in range(len(PdfTraj)):
        Times.append((i+1)*h)
    Times = np.asarray(Times)
    LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, "EM", dimension, Meshes, PdfTraj, Times)
    ErrorsEMT.append(L2wErrors[-1])
    

plt.figure()
plt.loglog(np.asarray(kstepMin), np.asarray(ErrorsEMT))



kstepMin = 0.01
hstep = [0.0001, 0.001, 0.01, 0.1]
# kstepMin = [0.001]

EndTime = 1
beta = 20
radius = 5# R
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 5
numPointsForLejaCandidates = 50
numQuadFit = 30


ErrorsEMT = []
timesEMT = []
for j in hstep:
    h=j
    start = time.time()
    NumSteps = int(1/h)
    par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
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
    for j in range(NumSteps):
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
    Times = []
    for i in range(len(PdfTraj)):
        Times.append((i+1)*h)
    Times = np.asarray(Times)
    LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, "EM", dimension, Meshes, PdfTraj, Times)
    ErrorsEMT.append(L2wErrors[-1])
# plt.scatter(Meshes[-1], PdfTraj[-1])
# plt.scatter(Meshes[-1], dtqApprox)

plt.figure()
plt.loglog(np.asarray(hstep), np.asarray(ErrorsEMT), '.', label = "Error")
plt.loglog(np.asarray(hstep), np.asarray(hstep), label="slope 1")
plt.legend()
plt.xlabel("Time step size")
plt.ylabel("Error")




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

