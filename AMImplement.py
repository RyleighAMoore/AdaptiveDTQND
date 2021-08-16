import Functions as F
import numpy as np
import Functions as fun
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
from pyopoly1.Scaling import GaussScale
import ICMeshGenerator as M
import matplotlib.pyplot as plt
import QuadraticFit as QF
np.seterr(divide='ignore', invalid='ignore')
from NDFunctionBank import SimpleDriftSDE
import pyopoly1.QuadratureRules as QR


T =0.1
s = 0.75
h = 0.1
init = 0
numsteps = int(np.ceil(T/h))-1
k = h**s
# k=0.1
yM = k*(np.pi/(k**2))
yM=4
M = int(np.ceil(yM/k))
meshO = k*np.linspace(-M,M,2*M+1)
meshO = np.expand_dims(np.asarray(meshO),1)


dimension = 1
sde = SimpleDriftSDE(1,1,dimension)
theta = 0.5
a1 = F.alpha1(theta)
a2 = F.alpha2(theta)


poly = HermitePolynomials(rho=0)
d=dimension
# k = 40    
lambdas = indexing.total_degree_indices(d, 40)
poly.lambdas = lambdas

'''Generate Alt Leja Points'''
# lejaPointsFinal, new = getLejaPoints(10, np.zeros((dimension,1)), poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

# mesh = VT.map_from_canonical_space(lejaPointsFinal, scale1)
ALp = np.zeros((len(meshO), len(meshO)))
for i in range(len(meshO)):
    for j in range(len(meshO)):
        indexOfMesh = meshO[j]
        indexOfMesh2 = meshO[i]
        M2 = 5*h 
        M2 = abs(sde.Diff((indexOfMesh2 + indexOfMesh)/2))*np.sqrt(theta*h)
        M2 = 5*M2[0][0]
    
        mesh = np.linspace(-M2,M2,25) + (indexOfMesh2 + indexOfMesh)/2
        mesh = np.expand_dims(np.asarray(mesh),1)
        
        val, scaleComb = F.AndersonMattingly(indexOfMesh, indexOfMesh2, mesh, h, sde.Drift, sde.Diff, False, theta, a1, a2, dimension)
        val = np.expand_dims(val,1)
        val = np.where(val <= 0, np.min(val), val)
        if np.max(val) < 10**(-16):
            ALp[i,j] = 0
            continue
        
        scale1, LSFit, Const, combinations = QF.leastSquares(mesh, val)
        
        vals = QF.ComputeDividedOut(mesh, LSFit, Const, scale1, combinations)
        
        c, cond, ind = QR.QuadratureByInterpolationND(poly, scale1, mesh, val/vals.T, 10, sde.Diff, 20)
        print(c)
        print(cond)
        ALp[i,j] = c

driftfun = sde.Drift
difffun = sde.Diff
# A = A[2:-2,2:-2]
init = np.asarray([0])
xvec = meshO
mymu = init + driftfun(init)*h
mysigma = abs(difffun(init))*np.sqrt(h)
scale = GaussScale(1)
scale.setMu(np.asarray([mymu]))
scale.setCov(np.asarray([mysigma**2]))
phat = fun.Gaussian(scale, xvec)

PdfTraj =[]
PdfTraj.append(phat)
for i in range(14): 
    phat = k*(ALp@phat)
    PdfTraj.append(phat)
    
    
trueSoln = []
from exactSolutions import OneDdiffusionEquation
for i in range(len(PdfTraj)):
    truepdf = OneDdiffusionEquation(np.expand_dims(xvec,1), sde.Diff(xvec), (i+1)*h, sde.Drift(xvec))
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))
from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(xvec, PdfTraj, trueSoln, plot=False)

# compare solutions
plt.figure()
index = 1
plt.plot(xvec,PdfTraj[index],'o')
plt.plot(xvec,trueSoln[index],'.r')




