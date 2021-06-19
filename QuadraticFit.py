
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
        
    if np.linalg.det(A)<= 0:
         return float('nan'),float('nan'),float('nan'), float('nan')
     
    if dimension == 1:
        cov = c[0]
        mean = -c[1]/(2*c[0])
        con = 1/np.exp(-c[1]**2/(4*c[0])+c[2]) #7.052369794346946
        # con = 12.499999999999986
        if math.isfinite(mean) and math.isfinite(np.sqrt(cov)):
            scaling = GaussScale(1)
            scaling.setMu(mean)
            scaling.setCov(1/cov)
        else:
            return float('nan'),float('nan'),float('nan'), float('nan')
        
        # plt.figure()
        # plt.plot(QuadMesh, pdf, 'o')
        # plt.plot(QuadMesh.T, np.exp(-(c[0]*QuadMesh**2+c[1]*QuadMesh+c[2])).T, '.r')
        # plt.plot(QuadMesh.T, 1/(np.sqrt(np.pi)*np.sqrt(scaling.cov))*np.exp(-(c[0]*QuadMesh**2+c[1]*QuadMesh+c[2])).T, '.k')

        return scaling, c, con, comboList
        
        # B = np.expand_dims(c[dimension+1:numLSBasis-1],1)
        
     
    B = np.expand_dims(c[dimension+ind+1:numLSBasis-1],1)
    # B = np.expand_dims(np.asarray([c[3], c[4]]),1)
         
    sigma = np.linalg.inv(A)
    Lam, U = np.linalg.eigh(A)
    if np.min(Lam) <= 0:
        return float('nan'),float('nan'),float('nan'), float('nan')
    
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
            vals = vals*QuadMesh[:,i[j]]  
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


# import ICMeshGenerator as M
# x = M.getICMeshRadius(0.25, 0.05, 0.01, 1)
# # x = np.expand_dims(np.linspace(-0.25, 0.25),1)  
# s = 0.1
# mu = 0
# import numpy as np
# import Functions as fun
# from pyopoly1.Scaling import GaussScale
# import ICMeshGenerator as M
# import matplotlib.pyplot as plt


# def drift(mesh):
#     return np.asarray(np.zeros((np.shape(mesh))))
    
# def diff(mesh):
#     return np.expand_dims(np.asarray(0.8*np.asarray(np.ones((np.size(mesh))))),1)

# h=0.01
# scale = GaussScale(1)
# scale.setMu(h*drift(np.zeros(1)).T)
# scale.setCov((h*diff(np.zeros(1))*diff(np.zeros(1)).T).T)

# # from watchpoints import watch
# pdf = fun.Gaussian(scale, x)**2
# # pdf = 1/(s*np.sqrt(np.pi))*np.exp(-(x-mu)**2/(s**2))

# scaling, cc, Const, combinations = leastSquares(x, np.expand_dims(pdf,1))
# print(scaling.mu)
# print(np.sqrt(scaling.cov))

# fullMesh = x
# fullPDF = pdf

# vals2 = np.zeros(np.size(fullPDF)).T
# count = 0
# dimension = np.size(fullMesh,1)
# for i in range(dimension):
#     vals2 += cc[count]*fullMesh[:,i]**2
#     count +=1
# for i,k in combinations:
#     vals2 += cc[count]*fullMesh[:,i]*fullMesh[:,k]
#     count +=1
# for i in range(dimension):
#     vals2 += cc[count]*fullMesh[:,i]
#     count +=1
    
# vals2 += cc[count]*np.ones(np.shape(vals2))

# vals = 1/(np.sqrt(scaling.cov)*np.sqrt(np.pi)**dimension)*np.exp(-(vals2))/Const
# vals = np.expand_dims(vals,1).T
# # vals = np.exp(-(vals2))/Const

# # vals = np.exp(-(cc[0]*x**2+cc[1]*x+cc[2]))
# vals = 1/(np.sqrt(scaling.cov)*np.sqrt(np.pi)**dimension)*np.exp(-(cc[0]*fullMesh**2+cc[1]*fullMesh+cc[2]*np.ones(np.shape(fullMesh)))).T/Const

# plt.plot(fullMesh,vals.T, '*')
# plt.plot(fullMesh,pdf, '.')
# plt.plot(fullMesh,np.expand_dims(pdf,1)/vals.T, 'o')
        
# print(np.expand_dims(pdf,1)/vals.T)


