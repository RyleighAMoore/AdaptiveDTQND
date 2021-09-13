from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Class_Parameters as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
# from NDFunctionBank import
import time
from DTQTensorized import ApproxExactSoln
import numpy as np
import Functions as fun
from scipy.spatial import Delaunay
import LejaQuadrature as LQ
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
import MeshUpdates2D as MeshUp
from pyopoly1.Class_Gaussian import GaussScale
import ICMeshGenerator as M
from pyopoly1.LejaPoints import getLejaSetFromPoints, getLejaPoints
import matplotlib.pyplot as plt
import numpy as np


import numpy as np

class Parameters:
   def __init__(self, fun, h, conditionNumForAltMethod, beta):
      self.conditionNumForAltMethod = conditionNumForAltMethod
      # self.NumLejas = NumLejas
      # self.numPointsForLejaCandidates = numPointsForLejaCandidates
      # self.numQuadFit = numQuadFit
      diffMax = np.max(fun.Diff(np.zeros(fun.dim)))
      self.h = h
      self.kstepMin = 0.09
      self.kstepMax = 0.95
      self.beta = beta
      self.radius = np.sqrt(diffMax*h)*6 #+0.5*np.exp(-fun.dim+1)+1)
      self.NumLejas = int(10*fun.dim)
      self.numPointsForLejaCandidates = int((1/self.kstepMin)**fun.dim/3)
      self.numQuadFit = int((1/self.kstepMin)**fun.dim/3)
      # self.numPointsForLejaCandidates = 350
      # self.numQuadFit = 350
      # self.maxDiff = None
      # self.minDiff = None


   # def set_kstepMin(self, dimension, diff, h):
   #     kstepMin = 0.08
   #     self.kstepMin = kstepMin


   # def set_kstepMax(self, dimension, diff, h):
   #     kstepMin = 0.09
   #     self.kstepMin = kstepMin

   # def set_radius(self, mesh, pdfTraj):
   #     self.radius = radius

   # def set_NumLejas(self, dimension, diff):
   #     self.NumLejas = None

   # def numPointsForLejaCandidates(self, dimension, diff):
   #     self.numPointsForLejaCandidates = None

   # def numQuadFit(self, dimension, diff):
   #     self.numQuadFit = None


class SimpleDriftSDE:
  def __init__(self, drift, diff, dimension):
    self.drift = drift
    self.diff = diff
    self.dim = dimension

  def Drift(self, mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = self.drift
    return dr

  def Diff(self, mesh):
      return self.diff*np.diag(np.ones(self.dim))

  def Solution(self, mesh, t):
    D = self.diff**2*0.5
    r = (mesh[:,0]-self.drift*t)**2
    for ii in range(1,self.dim):
        r += (mesh[:,ii])**2
    vals = np.exp(-r/(4*D*t))*(1/(4*np.pi*D*t))**(self.dim/2)
    return vals

dimension =1
sde = SimpleDriftSDE(1,1,dimension)
# mydrift = sde.Drift
# mydiff = sde.Diff

# def mydrift(mesh):
#       if mesh.ndim ==1:
#         mesh = np.expand_dims(mesh, axis=0)
#     # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return -1*mesh
#       return 0.2*mesh*(4-mesh**2)

# def mydiff(mesh):
#     return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

def mydrift(mesh):
      if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return -1*mesh
      return np.zeros(np.shape(mesh))

def mydiff(mesh):
    return np.expand_dims(np.asarray(1*np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

ApproxSoln = False
timeStep = [0.01]
EndTime =0.5
kstepMin = 0.06 # lambda
kstepMax = kstepMin # Lambda
beta = 3
radius = 2 # R
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
    par = Parameters(sde, h, conditionNumForAltMethod, beta)
    par.kstepMin = kstepMin
    par.kstepMax = kstepMax
    par.radius = radius
    par.numPointsForLejaCandidates = numPointsForLejaCandidates
    par.numQuadFit = numQuadFit
    par.NumLejas = NumLejas
    TSType = "EM"

    Meshes, PdfTraj, LPReuseArr, AltMethod, GMat= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True, TimeStepType= TSType, RetG = True)
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
        # truepdf = sde.Solution(Meshes[i], (i+1)*h)
        truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))

    from Errors import ErrorValsExact
    if not ApproxSoln:
        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
    else:
        Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
        LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
    ErrorsEM.append(L2wErrors[-1])



# timeStepAM= [0.05, 0.08]
# ErrorsAM = []
# timesAM = []
# LPA = []
# AltA = []
# for i in timeStepAM:
#     start = time.time()

#     NumSteps = int(EndTime/i)
#     '''Discretization Parameters'''
#     h= i

#     par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
#     par.kstepMin = kstepMin
#     par.kstepMax = kstepMax
#     par.radius = radius
#     par.numPointsForLejaCandidates = numPointsForLejaCandidates
#     par.numQuadFit = numQuadFit
#     par.NumLejas = NumLejas
#     TSType = "AM"

#     Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True, TimeStepType= TSType)
#     end = time.time()
#     timesAM.append(end - start)


#     pc = []
#     for i in range(len(Meshes)-1):
#         l = len(Meshes[i])
#         pc.append(LPReuseArr[i]/l)

#     mean = np.mean(pc)
#     print("Leja Reuse: ", mean*100, "%")
#     LPA.append(mean*100)


#     pc = []
#     for i in range(len(Meshes)-1):
#         l = len(Meshes[i])
#         pc.append(AltMethod[i]/l)

#     mean2 = np.mean(pc)
#     print("Alt Method: ", mean2*100, "%")
#     AltA.append(mean2*100)

#     trueSoln = []
#     from exactSolutions import OneDdiffusionEquation
#     for i in range(len(Meshes)):
#         truepdf = sde.Solution(Meshes[i], (i+1)*h)
#         # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
#         # truepdf = solution(xvec,-1,T)
#         trueSoln.append(np.squeeze(np.copy(truepdf)))

#     from Errors import ErrorValsExact
#     if not ApproxSoln:
#         LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
#     else:
#         Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
#         LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)

#     # plt.figure()
#     # plt.semilogy(Meshes[-1], PdfTraj[-1]- trueSoln[-1])
#     # plt.show()
#     ErrorsAM.append(L2wErrors[-1])


# radius = 6 # R


# # ErrorsEMT = []
# # timesEMT = []
# # for i in timeStep:
# #     start = time.time()
# #     NumSteps = int(EndTime/i)
# #     h= i
# #     par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
# #     par.kstepMin = kstepMin
# #     par.kstepMax = kstepMax
# #     par.radius = radius
# #     par.numPointsForLejaCandidates = numPointsForLejaCandidates
# #     par.numQuadFit = numQuadFit
# #     par.NumLejas = NumLejas
# #     TSType = "EM"

# #     mesh = M.NDGridMesh(dimension, kstepMin, 7.5, UseNoise = False)
# #     mean = 3
# #     delta = np.ones(np.shape(mesh))*mean
# #     mesh = np.asarray(mesh).T + delta.T
# #     mesh = mesh.T
# #     GMat = fun.GenerateEulerMarMatrix(len(mesh), mesh, h, mydrift, mydiff, SpatialDiff)
# #     scale = GaussScale(dimension)
# #     scale.setMu(h*mydrift(np.zeros(dimension)).T)
# #     scale.setCov((h*mydiff(np.zeros(dimension))*mydiff(np.zeros(dimension)).T).T)

# #     p = fun.Gaussian(scale, mesh)
# #     PdfTraj = []
# #     Meshes = []
# #     for i in range(NumSteps):
# #         Meshes.append(mesh)
# #         p = kstepMin*np.copy(GMat@p)
# #         PdfTraj.append(np.copy(p))

# #     end = time.time()
# #     timesEMT.append(end -start)

# #     trueSoln = []
# #     from exactSolutions import OneDdiffusionEquation
# #     for i in range(len(Meshes)):
# #         truepdf = sde.Solution(Meshes[i], (i+1)*h)
# #         # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
# #         # truepdf = solution(xvec,-1,T)
# #         trueSoln.append(np.squeeze(np.copy(truepdf)))

# #     from Errors import ErrorValsExact
# #     if not ApproxSoln:
# #         LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
# #     else:
# #         Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
# #         LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
# #     ErrorsEMT.append(L2wErrors[-1])


# # ErrorsAMT = []
# # timesAMT = []
# # for i in timeStep:
# #     start = time.time()
# #     NumSteps = int(EndTime/i)
# #     h= i
# #     par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
# #     par.kstepMin = kstepMin
# #     par.kstepMax = kstepMax
# #     par.radius = radius
# #     par.numPointsForLejaCandidates = numPointsForLejaCandidates
# #     par.numQuadFit = numQuadFit
# #     par.NumLejas = NumLejas
# #     TSType = "AM"

# #     mesh = M.NDGridMesh(dimension, kstepMin, 7.5, UseNoise = False)
# #     mean = 3
# #     delta = np.ones(np.shape(mesh))*mean
# #     mesh = np.asarray(mesh).T + delta.T
# #     mesh = mesh.T

# #     GMat = fun.GenerateAndersonMatMatrix(h, mydrift, mydiff, mesh, dimension, len(mesh), kstepMin, SpatialDiff)
# #     scale = GaussScale(dimension)
# #     scale.setMu(h*mydrift(np.zeros(dimension)).T)
# #     scale.setCov((h*mydiff(np.zeros(dimension))*mydiff(np.zeros(dimension)).T).T)

# #     p = fun.Gaussian(scale, mesh)
# #     PdfTraj = []
# #     Meshes = []
# #     for i in range(NumSteps):
# #         Meshes.append(mesh)
# #         p = kstepMin*np.copy(GMat@p)
# #         PdfTraj.append(np.copy(p))

# #     end = time.time()
# #     timesAMT.append(end -start)

# #     trueSoln = []
# #     from exactSolutions import OneDdiffusionEquation
# #     for i in range(len(Meshes)):
# #         truepdf = sde.Solution(Meshes[i], (i+1)*h)
# #         # truepdf = OneDdiffusionEquation(Meshes[i], mydiff(Meshes[i]), (i+1)*h, mydrift(Meshes[i]))
# #         # truepdf = solution(xvec,-1,T)
# #         trueSoln.append(np.squeeze(np.copy(truepdf)))

# #     from Errors import ErrorValsExact
# #     if not ApproxSoln:
# #         LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=False)
# #     else:
# #         Times = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
# #         LinfErrors, L2Errors, L1Errors, L2wErrors, dtqApprox= ApproxExactSoln(EndTime, mydrift, mydiff, TSType, dimension, Meshes, PdfTraj, Times)
# #     ErrorsAMT.append(L2wErrors[-1])

# plt.figure()
# plt.loglog(np.asarray(timeStep), ErrorsEM, '-o', label="EM")
# plt.loglog(np.asarray(timeStepAM), ErrorsAM, '-o', label="AM")
# # plt.loglog(np.asarray(timeStep), ErrorsAMT, '-o', label="AM Trapezoidal")
# # plt.loglog(np.asarray(timeStep), ErrorsEMT, '-o', label="EM Trapezoidal")
# plt.xlabel("timestep")
# plt.ylabel("Error")
# plt.legend()

# plt.figure()
# plt.plot(np.asarray(timeStep), timesEM, '-o', label="EM")
# plt.plot(np.asarray(timeStepAM), timesAM, '-o', label="AM")
# # plt.plot(np.asarray(timeStep), timesAMT, '-o', label="AM Trapezoidal")
# # plt.plot(np.asarray(timeStep), timesEMT, '-o', label="EM Trapezoidal")
# plt.xlabel("timestep")
# plt.ylabel("Time (seconds)")
# plt.legend()

# # plt.figure()
# # plt.plot(np.asarray(timeStep), LPE, '-o', label="EM Avg. Leja Reuse")
# # plt.plot(np.asarray(timeStepAM), LPA, '-o', label="AM Avg. Leja Reuse")
# # plt.plot(np.asarray(timeStep), AltE, '-o', label="EM Avg. Alt Method Use")
# # plt.plot(np.asarray(timeStepAM), AltA, '-o', label="AM Avg. Alt Method Use")
# # plt.xlabel("timestep")
# # plt.ylabel("Percent")
# # plt.legend()

# plt.figure()
# plt.semilogy(timesEM, ErrorsEM, '-o', label="EM")
# plt.semilogy(timesAM, ErrorsAM, '-o', label="AM")
# # plt.semilogy(timesAMT, ErrorsAMT, '-o', label="AM Trapezoidal")
# # plt.semilogy(timesEMT, ErrorsEMT, '-o', label="EM Trapezoidal")
# plt.xlabel("Runtime (Seconds)")
# plt.ylabel("Error")
# plt.legend()

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

