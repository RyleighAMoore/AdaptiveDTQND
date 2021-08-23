import numpy as np
from scipy.stats import multivariate_normal
import ICMeshGenerator as M

# Density, distribution ction, quantile ction and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def dnorm_partialx(x, mu, sigma):
    return np.divide(-x+mu,sigma**2)*np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def GVals(indexOfMesh, mesh, h):
    val = G(indexOfMesh,mesh,h)
    return val

def Gaussian(scaling, mesh):
    rv = multivariate_normal(scaling.mu.T[0], scaling.cov)        
    soln_vals = np.asarray([rv.pdf(mesh)]).T
    return np.squeeze(soln_vals)

def weightExp(scaling,mesh):
    if np.size(mesh,1) == 1:
        mu = scaling.mu
        cov = scaling.cov
        newvals = np.exp(-(mesh-mu)**2)*(1/cov)
        return np.squeeze(newvals)
        
    mu = scaling.mu
    D = mesh.shape[1]
    cov = scaling.cov
    soln_vals = np.empty(len(mesh))
    # const = 1/(np.sqrt((np.pi)**D*abs(np.linalg.det(cov))))
    invCov = np.linalg.inv(cov)
    for j in range(len(mesh)):
        x = np.expand_dims(mesh[j,:],1)
        Gs = np.exp(-(x-mu).T@invCov@(x-mu))
        soln_vals[j] = Gs
    return soln_vals


def covPart(Px, Py, mesh, cov):
    vals = []
    for i in range(len(mesh)):
        val = np.exp(-2*cov*(mesh[i,0]-Px)*(mesh[i,1]-Py))
        vals.append(np.copy(val))
    return np.asarray(vals)


def G(indexOfMesh,mesh, h, drift, diff, SpatialDiff):
    '''Changing mu and cov over each row'''
    x = mesh[indexOfMesh,:]
    D = mesh.shape[1]
    mean = mesh+drift(mesh)*h
    
    if D == 1:
        newpointVect = x*np.ones(np.shape(mesh))
        var = h*diff(mesh)**2
        newVals = 1/(np.sqrt((2*np.pi*var)))*np.exp(-(newpointVect-mean)**2/(2*var))
        return np.squeeze(newVals)
    
    if not SpatialDiff:
            cov = diff(x)@diff(x).T * h
            const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
            invCov = np.linalg.inv(cov)
        
    soln_vals = np.empty(len(mesh))
    for j in range(len(mesh)):
        if SpatialDiff:
            m = mesh[j,:]
            cov = diff(m)@diff(m).T * h
            const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
            invCov = np.linalg.inv(cov)
        mu = mean[j,:]
        Gs = np.exp(-1/2*((x-mu).T@invCov@(x.T-mu.T)))
        soln_vals[j] = Gs
    return soln_vals*const



def AddPointToG(mesh, newPointindex, h, GMat, drift, diff, SpatialDiff):
    newRow = G(newPointindex, mesh,h, drift, diff, SpatialDiff)
    GMat[newPointindex,:len(newRow)] = newRow

    if np.size(mesh,1) == 1:
        var = diff(mesh[newPointindex,:])**2*h
        mean = mesh[newPointindex,:]+ drift(mesh[newPointindex,:])*h
        newCol = 1/(np.sqrt(2*np.pi*var))*np.exp(-(mesh-mean)**2/(2*var))
        GMat[:len(newCol),newPointindex] = np.squeeze(newCol)
        return GMat
        
    #Now, add new column
    D = mesh.shape[1]
    # mu1 = mesh[-1,:]+drift(np.expand_dims(mesh[-1,:],axis=0))*h
    # mu1 = mu1[0]
    mu = mesh[newPointindex,:]+drift(np.expand_dims(mesh[newPointindex,:],axis=0))*h
    mu = mu[0]
    # assert(np.array_equal(mu1,mu))
    dfn = diff(mesh[newPointindex,:])
    cov = dfn@dfn.T*h
    newCol = np.empty(len(mesh))
    const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    covInv = np.linalg.inv(cov)
    for j in range(len(mesh)):
        x = mesh[j,:]
        Gs = np.exp(-1/2*((x-mu).T@covInv@(x.T-mu.T)))
        newCol[j] = (Gs)
    GMat[:len(newCol),newPointindex] = newCol*const
    return GMat

from pyopoly1.Scaling import GaussScale

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

def AndersonMattingly(yim1, yi, meshAM, h, driftfun, difffun, SpatialDiff, theta, a1, a2, dimension):
    mu1 = yim1 + driftfun(yim1)*theta*h
    sig1 = abs(difffun(yim1))*np.sqrt(theta*h)
    scale = GaussScale(dimension)
    scale.setMu(np.asarray(mu1.T))
    scale.setCov(np.asarray(sig1**2))

    N1 = Gaussian(scale, meshAM)
    
    xsum = []
    for i in meshAM:
        mu2 = i + (a1*driftfun(i) - a2*driftfun(yim1))*(1-theta)*h
        sig2 = np.sqrt(rho2(a1*difffun(i)**2 - a2*difffun(yim1)**2))*np.sqrt((1-theta)*h)
        scale2 = GaussScale(dimension)
        scale2. setMu(np.asarray(mu2.T))
        scale2.setCov(np.asarray(sig2**2))
        N2 = Gaussian(scale2, yi)
        xsum.append(N2)
    return N1*np.asarray(xsum), 0
        
import Functions as F
import QuadraticFit as QF
import pyopoly1.QuadratureRules as QR

# def AndersonMattinglyMatrix(meshOriginal, h, sde, theta, a1, a2, dimension, poly):
#     meshO = meshOriginal
#     ALp = np.zeros((len(meshO), len(meshO)))
#     for i in range(len(meshO)):
#         for j in range(len(meshO)):
#             indexOfMesh = meshO[j]
#             indexOfMesh2 = meshO[i]
#             M2 = 5*h
        
#             mesh = np.linspace(-M2,M2,10) + (indexOfMesh2 + indexOfMesh)/2
#             mesh = np.expand_dims(np.asarray(mesh),1)
            
            
#             val, scaleComb = F.AndersonMattingly(indexOfMesh, indexOfMesh2, mesh, h, sde.Drift, sde.Diff, False, theta, a1, a2, dimension)
#             val = np.expand_dims(val,1)
#             val = np.where(val <= 0, np.min(val), val)
#             # if np.max(val) < 10**(-16):
#             #     ALp[i-ii,j-ii] = 0
#             #     continue
            
#             scale1, LSFit, Const, combinations = QF.leastSquares(mesh, val)
            
#             vals = QF.ComputeDividedOut(mesh, LSFit, Const, scale1, combinations)
            
#             c, cond, ind = QR.QuadratureByInterpolationND(poly, scale1, mesh, val/vals.T, 10, sde.Diff, 5000)
#             # print(c)
#             # print(cond)
#             ALp[i,j] = c
            
#             return ALp
        
        
def GenerateEulerMarMatrix(maxDegFreedom, mesh, h, drift, diff, SpatialDiff):
    GMat = np.empty([maxDegFreedom, maxDegFreedom])*np.NaN
    for i in range(len(mesh)):
        v = G(i,mesh, h, drift, diff, SpatialDiff)
        GMat[i,:len(v)] = v
    return GMat


from pyopoly1 import variableTransformations as VT

# def ComputeAndersonMattingly(index1, index2, h, Drift, Diff, DTQMesh, dimension, poly, numPointsForLejaCandidates, minDistanceBetweenPoints):
#     theta = 0.5
#     a1 = F.alpha1(theta)
#     a2 = F.alpha2(theta)
#     meshO = DTQMesh
#     ALp = np.zeros((len(meshO), len(meshO)))
#     i = index1
#     j = index2
   
#     indexOfMesh = meshO[j]
#     indexOfMesh2 = meshO[i]
#     # M2 = 5*h 
#     # M2 = abs(Diff((indexOfMesh2 + indexOfMesh)/2))*np.sqrt(theta*h)
#     # M2 = 5*M2[0][0]


#     # mesh = np.linspace(-M2,M2,25) + (indexOfMesh2 + indexOfMesh)/2
#     # mesh = np.expand_dims(np.asarray(mesh),1)    
    
#     scale = GaussScale(dimension)
#     # scale.setMu(((indexOfMesh2 + indexOfMesh)/2).T)
#     # import operator
#     # index, value = max(enumerate(pdf), key=operator.itemgetter(1))
#     scale.setMu(((np.asarray([indexOfMesh + Drift(indexOfMesh)*theta*h]))).T)
#     scale.setCov(np.diag(np.ones(dimension)))

#     mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, 2, UseNoise = False)
#     mesh = VT.map_from_canonical_space(mesh, scale)    
    
#     val, scaleComb = AndersonMattingly(indexOfMesh, indexOfMesh2, mesh, h, Drift, Diff, False, theta, a1, a2, dimension)
#     val = np.expand_dims(val,1)
#     # val = np.where(val <= 10**(-16), 10**(-16), val)
    
#     if np.max(val) < 10**(-16):
#         c = [0]
#     else:
#         scale1, LSFit, Const, combinations = QF.leastSquares(mesh, val)
#         vals = QF.ComputeDividedOut(mesh, LSFit, Const, scale1, combinations)
#         c, cond, ind = QR.QuadratureByInterpolationND(poly, scale1, mesh, val/vals.T, 10, Diff, numPointsForLejaCandidates)
#     return c[0]
import matplotlib.pyplot as plt

def ComputeAndersonMattingly(index1, index2, h, Drift, Diff, DTQMesh, dimension, poly, numPointsForLejaCandidates, meshForAM, theta, a1, a2, minDistanceBetweenPoints):
    meshO = DTQMesh
    # ALp = np.zeros((len(meshO), len(meshO)))
    i = index1
    j = index2
    indexOfMesh = meshO[j]
    indexOfMesh2 = meshO[i]
    
    scale = GaussScale(dimension)
    scale.setMu(((np.asarray(indexOfMesh + Drift(indexOfMesh)*theta*h))).T)
    scale.setCov(np.diag(np.ones(dimension)))

    meshAM = VT.map_from_canonical_space(meshForAM, scale)
        
    
    val, scaleComb = AndersonMattingly(indexOfMesh, indexOfMesh2, meshAM, h, Drift, Diff, False, theta, a1, a2, dimension)
    # assert val[0] < 10**(-6), print(val[0])
    # assert val[-1] < 10**(-6), print(val[0])
    val2 = np.sum(minDistanceBetweenPoints**dimension*val)   
    
    return val2
    
from tqdm import trange
def GenerateAndersonMatMatrix(h, Drift, Diff, DTQMesh, dimension, poly, numPointsForLejaCandidates, maxDegFreedom, minDistanceBetweenPoints):
    theta = 0.5
    a1 = F.alpha1(theta)
    a2 = F.alpha2(theta)
    meshAM = M.NDGridMesh(dimension, minDistanceBetweenPoints, 2, UseNoise = False)
    meshO = DTQMesh
    ALp = np.empty([maxDegFreedom, maxDegFreedom])*np.NaN
    for i in trange(len(meshO)):
        for j in range(len(meshO)):
            c = ComputeAndersonMattingly(i, j, h, Drift, Diff, DTQMesh, dimension, poly, numPointsForLejaCandidates, meshAM, theta, a1, a2, minDistanceBetweenPoints)
            ALp[i,j] = c
    return ALp


def AddPointToGAndersonMat(mesh, newPointindex, h, GMat, Drift, Diff, SpatialDiff, dimension, poly, numPointsForLejaCandidates, minDistanceBetweenPoints):
    theta = 0.5
    a1 = F.alpha1(theta)
    a2 = F.alpha2(theta)
    meshAM = M.NDGridMesh(dimension, minDistanceBetweenPoints, 2, UseNoise = False)

    for j in range(len(mesh)):
        val = ComputeAndersonMattingly(newPointindex, j, h, Drift, Diff, mesh, dimension, poly, numPointsForLejaCandidates, meshAM, theta, a1, a2, minDistanceBetweenPoints)
        GMat[newPointindex, j] = val
       
    for i in range(len(mesh)):
        val = ComputeAndersonMattingly(i, newPointindex, h, Drift, Diff, mesh, dimension, poly, numPointsForLejaCandidates, meshAM, theta, a1, a2, minDistanceBetweenPoints)
        GMat[i,newPointindex] = val
    return GMat
    

def GAndersonMat(mesh, newPointindex, h, GMat, Drift, Diff, SpatialDiff, dimension, poly, numPointsForLejaCandidates, minDistanceBetweenPoints):
    theta = 0.5
    a1 = F.alpha1(theta)
    a2 = F.alpha2(theta)
    meshAM = M.NDGridMesh(dimension, minDistanceBetweenPoints, 2, UseNoise = False)

    vals = []
    for j in range(len(mesh)):
        val = ComputeAndersonMattingly(newPointindex, j, h, Drift, Diff, mesh, dimension, poly, numPointsForLejaCandidates,meshAM, theta, a1, a2, minDistanceBetweenPoints)
        vals.append(val)
    return vals
        

