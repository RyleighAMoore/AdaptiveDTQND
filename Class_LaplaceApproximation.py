
import numpy as np
import math
from Class_Gaussian import GaussScale
from itertools import combinations
import operator as op
from functools import reduce

class LaplaceApproximation:
    def __init__(self, dimension):
        self.scalingForGaussian = None
        self.leastSqauresFit= None
        self.constantOfGaussian= None
        self.combinationOfBasisFunctionsList = None
        self.numLSBasis = self.computeShapeOfMatrix(dimension+2, 2)

    def computeShapeOfMatrix(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom  # or / in Python 2

    # @profile
    def buildVMatForLinFit(self, dimension, QuadMesh, laplaceFitPdf):
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

    # @profile
    def computeleastSquares(self, QuadMesh, laplaceFitPdf, dimension):
        M, comboList = self.buildVMatForLinFit(dimension, QuadMesh, laplaceFitPdf)

        MT = M.T
        const = -1*np.linalg.inv(MT@M)@(MT@np.log(laplaceFitPdf))
        c=const.T

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
                self.combinationOfBasisFunctionsList = None
                return
            else:
                self.scalingForGaussian = None
                self.leastSqauresFit= None
                self.constantOfGaussian= None
                self.combinationOfBasisFunctionsList = None
                return

        A = np.diag(c[:dimension])

        for ind,i in enumerate(comboList):
            A[i[0],i[1]] = 1/2*c[dimension+ind]
            A[i[1],i[0]] = 1/2*c[dimension+ind]

        if np.linalg.det(A)<= 0:
            self.scalingForGaussian = None
            self.leastSqauresFit= None
            self.constantOfGaussian= None
            self.combinationOfBasisFunctionsList = None
            return

        B = np.expand_dims(c[dimension+ind+1:self.numLSBasis-1],1)

        sigma = np.linalg.inv(A)
        Lam, U = np.linalg.eigh(A)
        if np.min(Lam) <= 0:
            self.scalingForGaussian = None
            self.leastSqauresFit= None
            self.constantOfGaussian= None
            self.combinationOfBasisFunctionsList = None
            return

        La = np.diag(Lam)
        mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T
        Const = np.exp(-c[-1]+1/4*B.T@U@np.linalg.inv(La)@U.T@B)

        if math.isfinite(mu[0][0]) and math.isfinite(mu[1][0]) and math.isfinite(np.sqrt(sigma[0,0])) and math.isfinite(np.sqrt(sigma[1,1])):
            scaling = GaussScale(dimension)
            scaling.setMu(mu)
            scaling.setCov(sigma)
        else:
            self.scalingForGaussian = None
            self.leastSqauresFit= None
            self.constantOfGaussian= None
            self.combinationOfBasisFunctionsList = None
            return

        self.scalingForGaussian = scaling
        self.leastSqauresFit= c
        self.constantOfGaussian= Const
        self.combinationOfBasisFunctionsList = comboList

    def ComputeDividedOut(self, pdf, dimension):
        if dimension == 1:
            vals = np.exp(-(self.leastSqauresFit[0]*pdf.meshCoordinates**2+self.leastSqauresFit[1]*pdf.meshCoordinates+self.leastSqauresFit[2])).T/self.constantOfGaussian
            vals = vals*1/(np.sqrt(np.pi)*np.sqrt(self.scalingForGaussian.cov))
        else:
            L = np.linalg.cholesky((self.scalingForGaussian.cov))
            JacFactor = np.prod(np.diag(L))
            # vals = 1/(np.pi*JacFactor)*np.exp(-(cc[0]*x**2+ cc[1]*y**2 + cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const

            vals2 = np.zeros(np.size(pdf.meshCoordinates,0)).T
            count = 0
            for i in range(dimension):
                vals2 += self.leastSqauresFit[count]*pdf.meshCoordinates[:,i]**2
                count +=1
            for i,k in self.combinationOfBasisFunctionsList:
                vals2 += self.leastSqauresFit[count]*pdf.meshCoordinates[:,i]*pdf.meshCoordinates[:,k]
                count +=1
            for i in range(dimension):
                vals2 += self.leastSqauresFit[count]*pdf.meshCoordinates[:,i]
                count +=1
            vals2 += self.leastSqauresFit[count]*np.ones(np.shape(vals2))
            vals = 1/(np.sqrt(np.pi)**dimension*JacFactor)*np.exp(-(vals2))/self.constantOfGaussian
        return np.squeeze(vals).T


    def ComputeDividedOutAM(self, pdf, dimension, mesh):
       temp = pdf.meshCoordinates
       pdf.meshCoordinates = mesh
       vals = self.ComputeDividedOut(pdf, dimension)
       pdf.meshCoordinates = temp
       return vals

