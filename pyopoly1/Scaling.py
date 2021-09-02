
import numpy as np 

class GaussScale:
    def __init__(self, numVars):
        self.numVars = numVars
        self.mu = np.zeros((numVars,1))
        self.cov = np.zeros((numVars, numVars))
        self.invCov = float('NaN')
        
    def getSigma(self):
        return np.sqrt(np.diagonal(self.cov))

    def setMu(self, muVals):
        # assert np.shape(muVals) == np.shape(self.mu), print(np.shape(muVals), 'Should be', np.shape(self.mu))
        self.mu = muVals
        
    def setCov(self, covMat):
        # assert np.shape(covMat) == np.shape(self.cov)
        self.cov = covMat
        self.invCov = np.linalg.inv(covMat)

        
    # def setInvCov(self, covMat):
    #     # assert np.shape(covMat) == np.shape(self.cov)
    #     self.invCov = np.linalg.inv(covMat)
    
    def setSigma(self, sigmas):
        # assert np.size(sigmas) == self.numVars
        for i in range(len(sigmas)):
            self.cov[i,i] = sigmas[i]**2