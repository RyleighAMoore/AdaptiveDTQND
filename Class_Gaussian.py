
import numpy as np
from scipy.stats import multivariate_normal


class GaussScale:
    def __init__(self, dimension):
        self.dimension = dimension
        self.mu = np.zeros((dimension,1))
        self.cov = np.zeros((dimension, dimension))
        self.invCov = None
        self.const = None
        self.invCovR = None

    def getSigma(self):
        return np.sqrt(np.diagonal(self.cov))

    def setMu(self, muVals):
        # assert np.shape(muVals) == np.shape(self.mu), print(np.shape(muVals), 'Should be', np.shape(self.mu))
        self.mu = muVals

    def setCov(self, covMat):
        # assert np.shape(covMat) == np.shape(self.cov)
        self.cov = covMat
        try:
            self.invCov = np.linalg.inv(covMat)
        except np.linalg.LinAlgError:
            self.invCov = np.linalg.inv(np.asarray(covMat))
        self.invCovR = 1/np.sqrt(2)*np.linalg.cholesky(self.invCov).T

    def setSigma(self, sigmas):
        # assert np.size(sigmas) == self.numVars
        for i in range(len(sigmas)):
            self.cov[i,i] = sigmas[i]**2

    def ComputeGaussian(self, mesh, dimension):
        if self.const == None:
            const = 1/(np.sqrt((2*np.pi)**dimension*abs(np.linalg.det(self.cov))))
            self.const = const

        diff = (mesh.T - self.mu)
        step1 = self.invCovR @ diff
        step2and3 = -1 * self.normSpecialSquared(step1, axis=0)
        step4 = np.exp(step2and3)
        soln = self.const * step4
        #soln = self.const * np.exp(-1*np.squeeze(np.sqrt(np.add.reduce(((self.invCovR @ diff).conj() * (self.invCovR @ diff)).real, axis=0, keepdims=True)))**2)
        # soln = self.const * np.exp(-1*np.linalg.norm(self.invCovR @ diff, axis=0)**2)
        return soln

    def normSpecialSquared(self, x, axis):
        axis = (axis,)
        s = (x.conj() * x).real
        return np.add.reduce(s, axis=axis)

    def combineTwoGaussians(self, scale2):
        if self.dimension ==1:
            muCombined = scale2.cov*1/(self.cov + scale2.cov)*self.mu + self.cov*1/(self.cov + scale2.cov)*scale2.mu
            covCombined = self.cov*1/(self.cov + scale2.cov)*scale2.cov
        else:
            muCombined = scale2.cov@np.linalg.inv(self.cov + scale2.cov)@self.mu + self.cov@np.linalg.inv(self.cov + scale2.cov)@scale2.mu
            covCombined = self.cov@np.linalg.inv(self.cov + scale2.cov)@scale2.cov
        return muCombined, covCombined



