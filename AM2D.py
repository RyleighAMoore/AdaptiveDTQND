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
from DTQAdaptive import DTQ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ICMeshGenerator as MG
import QuadraticFit as QF



from NDFunctionBank import SimpleDriftSDE
dimension = 1
sde = SimpleDriftSDE(1,1,dimension)

# def driftfun(mesh):
#     # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return -1*mesh
#     return mesh*(4-mesh**2)
    
# def difffun(mesh):
#     return np.ones(np.size(mesh))
#     # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

driftfun = sde.Drift
difffun = sde.Diff


# simulation parameters
T =10
s = 0.75
h=0.01
numsteps = int(np.ceil(T/h))-1
k = h**s
k=0.01
k=0.03
# yM = k*(np.pi/(k**2))
# yM=15
# M = int(np.ceil(yM/k))

# xvec = k*np.linspace(-M,M,2*M+1)
xvec = MG.NDGridMesh(dimension, k, 1, UseNoise = False)


def alpha1(th):
    return(1/(2*th*(1-th)))

def alpha2(th):
  num = (1-th)**2 + th**2
  denom = 2*th*(1-th)
  return(num/denom)

def rho(x):
  v = x
  v[x<0] = 0
  return(v)

def rho2(x):
    # if np.linalg.det(x) <0:
        
    # v=x
    # if v<0:
    #     v=0
    return x

theta = 0.5
a1 = alpha1(theta)
a2 = alpha2(theta)

A= np.zeros((len(xvec),len(xvec)))

xjmat = np.repeat(xvec, len(xvec), axis=1)
xstarmat = xjmat.T
for i in range(len(xvec)):
    xjm1 = np.asarray([xvec[i]])
    mu1 = xjm1 + driftfun(xjm1)*theta*h
    sig1 = abs(difffun(xjm1))*np.sqrt(theta*h)
    scale = GaussScale(dimension)
    scale.setMu(np.asarray(mu1.T))
    scale.setCov(np.asarray(sig1**2))
    
    pvec = fun.Gaussian(scale, xvec)
 
    mu2 = xstarmat + (a1*driftfun(xstarmat) - a2*driftfun(xjm1))*(1-theta)*h
    sig2 = np.sqrt(rho(a1*difffun(xstarmat)**2 - a2*difffun(xjm1)**2))*np.sqrt((1-theta)*h)
    pmat = np.exp(-(xjmat-mu2)**2/(2*sig2*sig2))/(sig2*np.sqrt(2*np.pi))
    
    A[:,i] = k*(pmat @ pvec)

A2 = np.zeros((len(xvec),len(xvec)))

for i in range(len(xvec)):
    print(i)
    xrow = xvec[i]
    for j in range(len(xvec)):
        xcol = xvec[j]
        prow = []
        pvec = []
        for m,xm in enumerate(xvec):
            xsum = xm
            mu1 = xcol + driftfun(xcol)*theta*h
            sig1 = abs(difffun(xcol))*np.sqrt(theta*h)
            scale = GaussScale(dimension)
            scale.setMu(np.asarray(mu1.T))
            scale.setCov(np.asarray(sig1**2))
            
            N1 = fun.Gaussian(scale, xsum)
            pvec.append(N1)
            # print(N1)
            
            mu2 = xsum + (a1*driftfun(xsum) - a2*driftfun(xcol))*(1-theta)*h
            sig2 = np.sqrt(rho2(a1*difffun(xsum)**2 - a2*difffun(xcol)**2))*np.sqrt((1-theta)*h)
            
            scale2 = GaussScale(dimension)
            scale2.setMu(np.asarray(mu2.T))
            scale2.setCov(np.asarray(sig2**2))
            
            # N2 = np.exp(-(xrow-mu2)**2/(2*sig2*sig2))/(sig2*np.sqrt(2*np.pi))
            N2 = fun.Gaussian(scale2, xrow)
            # print(N2)
            prow.append(N2)
            
        A2[i,j]= k*np.asarray(prow)@np.asarray(pvec)
    
fullPDF = np.expand_dims(A2[100,:],1)

scale1, cc, Const, combinations = QF.leastSquares(xvec, fullPDF)

fullMesh = xvec
if np.size(fullMesh,1)==1:
    vals = np.exp(-(cc[0]*fullMesh**2+cc[1]*fullMesh+cc[2])).T/Const
    vals = vals*1/(np.sqrt(np.pi)*np.sqrt(scale1.cov))
else:
    L = np.linalg.cholesky((scale1.cov))
    JacFactor = np.prod(np.diag(L))
    # vals = 1/(np.pi*JacFactor)*np.exp(-(cc[0]*x**2+ cc[1]*y**2 + cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const

    vals2 = np.zeros(np.size(fullPDF)).T
    count = 0
    dimension = np.size(fullMesh,1)
    for i in range(dimension):
        vals2 += cc[count]*fullMesh[:,i]**2
        count +=1
    for i,k in combinations:
        vals2 += cc[count]*fullMesh[:,i]*fullMesh[:,k]
        count +=1
    for i in range(dimension):
        vals2 += cc[count]*fullMesh[:,i]
        count +=1
    vals2 += cc[count]*np.ones(np.shape(vals2))
    vals = 1/(np.sqrt(np.pi)**dimension*JacFactor)*np.exp(-(vals2))/Const

value = fullPDF/vals.T


plt.scatter(xvec, value, label="A-M")
# plt.scatter(xvec, fullPDF, label="A-M")


    
# init = np.zeros(dimension).T
# mymu = driftfun(init)*h
# mysigma = abs(difffun(init))*np.sqrt(h)
# scale = GaussScale(1)
# scale.setMu(np.asarray([mymu]))
# scale.setCov(np.asarray([mysigma**2]))
# phat = fun.Gaussian(scale, xvec)



from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from scipy.special import erf
from NDFunctionBank import SimpleDriftSDE

dimension = 1
fun = SimpleDriftSDE(2,0.5,dimension)
# mydrift = fun.Drift
mydrift = driftfun
mydiff = fun.Diff

'''Initialization Parameters'''
NumSteps = 1
'''Discretization Parameters'''
# a = 1
h=0.01
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
beta = 3
# radius = 0.55 # R
SpatialDiff = False
conditionNumForAltMethod = 8
# NumLejas = 30
# numPointsForLejaCandidates = 350
# numQuadFit = 350

par = Param.Parameters(fun, h, conditionNumForAltMethod, beta)
par.radius = 1
par.kstepMin = k
# par.minDistanceBetweenPoints = 0.09

Meshes, PdfTraj, LPReuseArr, AltMethod, GMat = DTQ(NumSteps, par.kstepMin, par.kstepMax, par.h, par.beta, par.radius, mydrift, mydiff, dimension, SpatialDiff, par, RetG=True)

fullPDF2 = np.expand_dims(GMat[100,:len(Meshes[0])],1)

scale1, cc, Const, combinations = QF.leastSquares(Meshes[0], fullPDF2)

fullMesh = Meshes[0]
if np.size(fullMesh,1)==1:
    vals = np.exp(-(cc[0]*fullMesh**2+cc[1]*fullMesh+cc[2])).T/Const
    vals = vals*1/(np.sqrt(np.pi)*np.sqrt(scale1.cov))
else:
    L = np.linalg.cholesky((scale1.cov))
    JacFactor = np.prod(np.diag(L))
    # vals = 1/(np.pi*JacFactor)*np.exp(-(cc[0]*x**2+ cc[1]*y**2 + cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const

    vals2 = np.zeros(np.size(fullPDF)).T
    count = 0
    dimension = np.size(fullMesh,1)
    for i in range(dimension):
        vals2 += cc[count]*fullMesh[:,i]**2
        count +=1
    for i,k in combinations:
        vals2 += cc[count]*fullMesh[:,i]*fullMesh[:,k]
        count +=1
    for i in range(dimension):
        vals2 += cc[count]*fullMesh[:,i]
        count +=1
    vals2 += cc[count]*np.ones(np.shape(vals2))
    vals = 1/(np.sqrt(np.pi)**dimension*JacFactor)*np.exp(-(vals2))/Const

value = fullPDF2/vals.T


plt.scatter(fullMesh, value, label="E-M")
# plt.scatter(fullMesh, fullPDF2, label="E-M")

plt.legend()
