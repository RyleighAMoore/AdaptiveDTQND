"""
Created on Fri Apr  3 12:44:33 2020
@author: Rylei
"""
from pyopoly1 import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
from pyopoly1 import opolynd
from mpl_toolkits.mplot3d import Axes3D
from Functions import *
from pyopoly1.Class_Gaussian import GaussScale
from pyopoly1.Plotting import productGaussians2D
import UnorderedMesh as UM
from pyopoly1.families import HermitePolynomials
import pyopoly1.indexing
import pyopoly1.LejaPoints as LP
from QuadraticFit import leastSquares, ComputeDividedOut
from scipy.interpolate import griddata
import math
np.seterr(divide='ignore', invalid='ignore')



def QuadratureByInterpolation_Simple(poly, scaling, mesh, pdf):
    '''Quadrature rule with no change of variables. Must pass in mesh you want to use.'''
    u = VT.map_to_canonical_space(mesh, scaling)
    
    numSamples = len(u)          
    V = opolynd.opolynd_eval(u, poly.lambdas[:numSamples,:], poly.ab, poly)
    vinv = np.linalg.inv(V)
    c = np.matmul(vinv[0,:], pdf)
    
    return c, np.sum(np.abs(vinv[0,:]))
    
  
def QuadratureByInterpolationND(poly, scaling, mesh, pdf, NumLejas, diff, numPointsForLejaCandidates):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    u = VT.map_to_canonical_space(mesh, scaling)
    
    dimension = np.size(mesh,1)
    normScale = GaussScale(dimension)
    normScale.setMu(np.zeros(dimension).T)
    normScale.setCov(np.eye(dimension))
    
    mesh2, pdfNew, indices = LP.getLejaSetFromPoints(normScale, u, NumLejas, poly, pdf, diff, numPointsForLejaCandidates)
    if math.isnan(indices[0]):
        return [10000], 10000, 10000
    assert np.max(indices) < len(mesh)

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    try:
        vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError as err: 
        if 'Singular matrix' in str(err):
            return [1000], 1000, indices
    c = np.matmul(vinv[0,:], pdfNew)
    return c, np.sum(np.abs(vinv[0,:])), indices


def QuadratureByInterpolationND_KnownLP(poly, scaling, mesh, pdf, LejaIndices):
    '''Quadrature rule with change of variables for nonzero covariance. 
    Used by QuadratureByInterpolationND_DivideOutGaussian
    Selects a Leja points subset of the passed in mesh'''
    LejaMesh = mesh[LejaIndices]
    mesh2 = VT.map_to_canonical_space(LejaMesh, scaling)
    pdfNew = pdf[LejaIndices]

    numSamples = len(mesh2)          
    V = opolynd.opolynd_eval(mesh2, poly.lambdas[:numSamples,:], poly.ab, poly)
    try:
        vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError as err: 
        if 'Singular matrix' in str(err):
        # print("Singular******************")
            return 100000, 100000
    c = np.matmul(vinv[0,:], pdfNew)
    return c, np.sum(np.abs(vinv[0,:]))



def QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, fullMesh, fullPDF, LPMat, LPMatBool, index, NumLejas, numQuadPoints,diff, numPointsForLejaCandidates):
    '''Divides out Gaussian using a quadratic fit. Then computes the update using a Leja Quadrature rule.'''
    if not LPMatBool[index][0]: # Do not have points for quadratic fit
        mesh, distances, ii = UM.findNearestKPoints(scaling.mu, fullMesh,numQuadPoints, getIndices = True)
        mesh =  mesh[:numQuadPoints]
        pdf = fullPDF[ii[:numQuadPoints]]
        scale1, LSFit, Const, combinations = leastSquares(mesh, pdf)
        
    else:
        QuadPoints = LPMat[index,:].astype(int)
        mesh = fullMesh[QuadPoints]
        pdf = fullPDF[QuadPoints]
        scale1, LSFit, Const, combinations = leastSquares(mesh, pdf)

    if not math.isnan(Const): # succeeded fitting Gaussian
        # if np.size(fullMesh,1)==1:
        #     vals = np.exp(-(cc[0]*fullMesh**2+cc[1]*fullMesh+cc[2])).T/Const
        #     vals = vals*1/(np.sqrt(np.pi)*np.sqrt(scale1.cov))
        # else:
        vals = ComputeDividedOut(fullMesh, LSFit, Const, scale1, combinations)
            # cc = LSFit
            # L = np.linalg.cholesky((scale1.cov))
            # JacFactor = np.prod(np.diag(L))
            # # vals = 1/(np.pi*JacFactor)*np.exp(-(cc[0]*x**2+ cc[1]*y**2 + cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const
        
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
            # vals = 1/(np.sqrt(np.pi)**dimension*JacFactor)*np.exp(-(vals2))/Const
        # plt.figure()
        # plt.plot(fullMesh,fullPDF, 'o')
        # plt.plot(fullMesh,vals.T, '.')
        
        # vals1 = vals*(1/np.sqrt(np.pi**2*np.linalg.det(scale1.cov)))
        # vals2 = Gaussian(scale1, fullMesh)
        # vals2 = weightExp(scale1,fullMesh)
        # vals = np.expand_dims(vals,0)
        # assert np.isclose(np.max(np.abs(vals-vals3)),0)
        np.seterr(divide='ignore', invalid='ignore')
        pdf2 = fullPDF/vals.T
        
        if LPMatBool[index][0]: # Don't Need LejaPoints
            LejaIndices = LPMat[index,:].astype(int)
            value, condNum = QuadratureByInterpolationND_KnownLP(poly, scale1, fullMesh, pdf2, LejaIndices)
            if condNum > 1.1:
                LPMatBool[index]=False
            else:
                return value[0], condNum, scale1, LPMat, LPMatBool, 1
            
        if not LPMatBool[index][0]: # Need Leja points.
            value, condNum, indices = QuadratureByInterpolationND(poly, scale1, fullMesh, pdf2,NumLejas, diff, numPointsForLejaCandidates)
            LPMat[index, :] = np.asarray(indices)
            if condNum < 1.1:
                LPMatBool[index] = True
            else: 
                LPMatBool[index] = False
            return value[0], condNum, scale1, LPMat, LPMatBool,0
    return float('nan'), float('nan'), float('nan'), LPMat, LPMatBool, 0
        
        
        
        
        
