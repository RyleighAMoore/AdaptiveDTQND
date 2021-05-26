
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from pyopoly1.Scaling import GaussScale
import Functions as fun

def leastSquares(QuadMesh, pdf):
    dimension = np.size(QuadMesh,1)
    numLSBasis = ncr(dimension+2, 2)
    M, comboList = buildVMatForLinFit(QuadMesh, dimension, numLSBasis)
    
    MT = M.T
    const = -1*np.linalg.inv(MT@M)@(MT@np.log(pdf))
    c=const.T[0]
    
    A = np.diag(c[:dimension])
    for ind,i in enumerate(comboList):
        A[i[0],i[1]] = 1/2*c[dimension+ind]
        A[i[1],i[0]] = 1/2*c[dimension+ind]
        
    # A = np.asarray([[c[0], 1/2*c[2]],[1/2*c[2],c[1]]])    

    B = np.expand_dims(c[dimension+ind+1:numLSBasis-1],1)
    # B = np.expand_dims(np.asarray([c[3], c[4]]),1)
    
    if np.linalg.det(A)<= 0:
         return float('nan'),float('nan'),float('nan')
         
    sigma = np.linalg.inv(A)
    Lam, U = np.linalg.eigh(A)
    if np.min(Lam) <= 0:
        return float('nan'),float('nan'),float('nan')
    
    La = np.diag(Lam)
    mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T    
    Const = np.exp(-c[-1]+1/4*B.T@U@np.linalg.inv(La)@U.T@B)
    
    if math.isfinite(mu[0][0]) and math.isfinite(mu[1][0]) and math.isfinite(np.sqrt(sigma[0,0])) and math.isfinite(np.sqrt(sigma[1,1])):
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[mu[0][0],mu[1][0]]]).T)
        scaling.setCov(sigma)
    else:
        return float('nan'),float('nan'),float('nan')
    # cc=pred_params
    # x,y = xy   
    # vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const[0][0]
    
    return scaling, c, Const, comboList



from itertools import combinations
def buildVMatForLinFit(QuadMesh, dimension, numLSBasis):
    M = np.zeros((len(QuadMesh), numLSBasis))
    size = 0
    for i in range(dimension):
        M[:,size] = QuadMesh[:,i]**2
        size+=1
    
    # dimension =4
    comb = combinations(list(range(dimension)), 2)
    comboList= list(comb)
    for i in comboList:
        # print(i)
        vals = np.ones(np.size(QuadMesh,0))
        for j in range(2):
            # print(j)
            vals = vals*QuadMesh[:,j]  
        M[:,size] = vals
        size +=1
        
    for i in range(dimension):
        M[:,size] = QuadMesh[:,i]
        size+=1
    M[:,size] = np.ones(np.size(QuadMesh,0))
    return M, comboList 
        
import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

    
    


