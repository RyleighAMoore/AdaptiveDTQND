import numpy as np
import math
from itertools import combinations
import operator as op
from functools import reduce

from Class_Gaussian import GaussScale


class LaplaceApproximation:
    def __init__(self, dimension):
        '''
        Used to approimate the log of the local PDF via Lapalce approximation
        of a quadratic form.

        dimenion: dimension of the SDE
        '''
        self.scalingForGaussian = None
        self.leastSqauresFit= None
        self.constantOfGaussian= None
        self.combinationOfBasisFunctionsList = None
        self.numLSBasis = self.computeShapeOfMatrix(dimension+2, 2)

    def computeShapeOfMatrix(self, n, r):
        '''Compute necessary shape for matrix in Laplace approximation given dimension of SDE'''
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom

    def buildVMatForLinFit(self, dimension, QuadMesh, laplaceFitPdf):
        '''Fill out matrix for Laplace approximation'''
        self.scalingForGaussian = None
        self.leastSqauresFit= None
        self.constantOfGaussian= None
        self.combinationOfBasisFunctionsList = None

        M = np.zeros((len(QuadMesh), self.numLSBasis))
        size = 0
        for i in range(dimension):
            M[:,size] = QuadMesh[:,i]**2
            size+=1
        comboList = list(combinations(list(range(dimension)), 2))
        for i in comboList:
            vals = np.ones(np.size(QuadMesh,0))
            for j in range(2):
                vals = vals*QuadMesh[:,i[j]]
            M[:,size] = vals
            size +=1

        for i in range(dimension):
            M[:,size] = QuadMesh[:,i]
            size+=1
        M[:,size] = np.ones(np.size(QuadMesh,0))
        return M, comboList

    def computeleastSquares(self, QuadMesh, laplaceFitPdf, dimension):
        '''Compute least squares fit for Laplace approximation'''
        M, comboList = self.buildVMatForLinFit(dimension, QuadMesh, laplaceFitPdf)
        # MT = M.T
        try:
            const, residuals, rank,s = np.linalg.lstsq(-M, np.log(laplaceFitPdf), rcond = None)
            # const = -1*np.linalg.inv(MT@M)@(MT@np.log(laplaceFitPdf))
        except:
            return

        '''Result of leas squares fit'''
        c=const.T


        '''For 1D: Compute corresponding mean, covariance, and constant for Gaussian
        corresponding to the fit. Return without setting Gaussian if failure occurs.
        '''
        if dimension == 1:
            cov = c[0]
            mean = -c[1]/(2*c[0])
            con = 1/np.exp(-c[1]**2/(4*c[0])+c[2])
            if math.isfinite(mean) and math.isfinite(np.sqrt(cov)):
                scaling = GaussScale(1)
                scaling.setMu(np.asarray([[mean]]))
                scaling.setCov(np.asarray([[1/cov]]))
                self.scalingForGaussian = scaling
                self.leastSqauresFit= c
                self.constantOfGaussian= con
                return
            else:
                return

        '''For ND: Compute corresponding mean, covariance, and constant for Gaussian
        corresponding to the fit. Return without setting Gaussian if failure occurs.'''
        A = np.diag(c[:dimension])
        for ind,i in enumerate(comboList):
            A[i[0],i[1]] = 1/2*c[dimension+ind]
            A[i[1],i[0]] = 1/2*c[dimension+ind]

        if np.linalg.det(A)<= 0:
            return

        B = np.expand_dims(c[dimension+ind+1:self.numLSBasis-1],1)

        sigma = np.linalg.inv(A)
        Lam, U = np.linalg.eigh(A)

        if np.min(Lam) <= 0:
            return

        La = np.diag(Lam)
        mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T

        try:
            Const = np.exp(-c[-1]+1/4*B.T@U@np.linalg.inv(La)@U.T@B)
        except:
            return

        if np.isfinite(mu).all() and np.isfinite(sigma).all():
            '''Everything looks good. Set scaling and return values'''
            scaling = GaussScale(dimension)
            scaling.setMu(mu)
            scaling.setCov(sigma)
            self.scalingForGaussian = scaling
            self.leastSqauresFit= c
            self.constantOfGaussian= Const
            self.combinationOfBasisFunctionsList = comboList
        else:
            return


    def ComputeDividedOut(self, meshCoordinates, dimension):
        '''We use this to divide the weight function out of the Gaussian for the integration.'''
        if dimension == 1:
            vals = np.exp(-(self.leastSqauresFit[0]*meshCoordinates**2+self.leastSqauresFit[1]*meshCoordinates+self.leastSqauresFit[2])).T/self.constantOfGaussian
            vals = vals*1/(np.sqrt(np.pi)*np.sqrt(self.scalingForGaussian.cov))
        else:
            L = np.linalg.cholesky((self.scalingForGaussian.cov))
            JacFactor = np.prod(np.diag(L))
            vals2 = np.zeros(np.size(meshCoordinates,0)).T
            count = 0
            for i in range(dimension):
                vals2 += self.leastSqauresFit[count]*meshCoordinates[:,i]**2
                count +=1
            for i,k in self.combinationOfBasisFunctionsList:
                vals2 += self.leastSqauresFit[count]*meshCoordinates[:,i]*meshCoordinates[:,k]
                count +=1
            for i in range(dimension):
                vals2 += self.leastSqauresFit[count]*meshCoordinates[:,i]
                count +=1
            vals2 += self.leastSqauresFit[count]*np.ones(np.shape(vals2))
            vals = 1/(np.sqrt(np.pi)**dimension*JacFactor)*np.exp(-(vals2))/self.constantOfGaussian
        return np.squeeze(vals).T


