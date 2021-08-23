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

driftfun = sde.Drift
difffun = sde.Diff


# simulation parameters
T =1
s = 0.75
h=0.01
numsteps = int(np.ceil(T/h))-1
k = h**s
xvec = MG.NDGridMesh(dimension, k, 1, UseNoise = False)

import ICMeshGenerator as M
import Functions as F


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
    
    
init = np.asarray([0])
mymu = init + driftfun(init)*h
mysigma = abs(difffun(init))*np.sqrt(h)
scale = GaussScale(1)
scale.setMu(np.asarray([mymu]))
scale.setCov(np.asarray([mysigma**2]))
phat = fun.Gaussian(scale, xvec)

PdfTraj =[]
PdfTraj.append(phat)
for i in range(numsteps): 
    phat = k*(A@phat)
    PdfTraj.append(phat)
    
    
trueSoln = []
from exactSolutions import OneDdiffusionEquation
for i in range(len(PdfTraj)):
    truepdf = OneDdiffusionEquation(np.expand_dims(xvec,1), difffun(xvec), (i+1)*h, 0)
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))
    
from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(xvec, PdfTraj, trueSoln, plot=False)

# compare solutions
plt.figure()
plt.plot(xvec,PdfTraj[-1],'o')
plt.plot(xvec,trueSoln[-1],'.r')

