
import numpy as np 

class GaussScale:
    def __init__(self, dimension):
        self.dimension = dimension
        self.mu = np.zeros((dimension,1))
        self.cov = np.zeros((dimension, dimension))
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
    
    def setSigma(self, sigmas):
        # assert np.size(sigmas) == self.numVars
        for i in range(len(sigmas)):
            self.cov[i,i] = sigmas[i]**2
            
    def ComputeGaussian(self, mesh):
        mu = np.repeat(self.mu, len(mesh.vals),axis = 0)
        invCov= self.invCov
        norm = np.zeros(len(mesh.vals))
        for dim in range(self.dimension):
            norm += (mesh.vals[:,dim] - mu[:,dim])**2
        const = 1/(np.sqrt((2*np.pi)**self.dimension*abs(np.linalg.det(self.cov))))
        soln = const*np.exp(-1/2*invCov*norm).T
        return np.squeeze(soln)