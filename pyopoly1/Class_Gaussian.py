
import numpy as np
from scipy.stats import multivariate_normal


class GaussScale:
    def __init__(self, dimension):
        self.dimension = dimension
        self.mu = np.zeros((dimension,1))
        self.cov = np.zeros((dimension, dimension))
        self.invCov = None

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

    # def ComputeGaussian(self, mesh, sde):
    #     if sde.dimension == 1:
    #         mu = np.repeat(self.mu, len(mesh),axis = 0)
    #         norm = (mesh - mu)**2
    #         const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(self.cov))))
    #         soln = np.squeeze(const*np.exp(-1/2*self.invCov*norm).T)
    #     else:
    #         vals = []
    #         const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(self.cov))))
    #         diff = (mesh-self.mu.T)
    #         for i, x in enumerate(mesh):
    #             x = np.expand_dims(x,1)
    #             temp = diff[i,:]
    #             val = temp.T@self.invCov@temp
    #             vals.append(val)
    #         soln = const*np.exp(-1/2*np.asarray(vals)).T
    #     return soln


    def ComputeGaussian(self, mesh, sde):
        # same mean and cov for all points.
        if sde.dimension == 1:
            mu = np.repeat(self.mu, len(mesh),axis = 0)
            norm = (mesh - mu)**2
            const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(self.cov))))
            soln = np.squeeze(const*np.exp(-1/2*self.invCov*norm).T)

        if sde.dimension == 2:
            norm1 = np.subtract(mesh[:,0],self.mu[0])
            norm2 = np.subtract(mesh[:,1],self.mu[1])
            norm12 = norm1*norm2

            const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(self.cov))))
            first = np.exp(-1/2*self.invCov[0,0]*norm1**2).T
            second = np.exp(-1/2*self.invCov[0,1]*norm12).T
            third = np.exp(-1/2*self.invCov[1,0]*norm12).T
            fourth = np.exp(-1/2*self.invCov[1,1]*norm2**2).T

            soln = np.squeeze(const*first*second*third*fourth)
            # rv = multivariate_normal(self.mu.T[0], self.cov)
            # soln_vals = np.asarray([rv.pdf(mesh)]).T
            # soln= np.squeeze(soln_vals)

        else:
            const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(self.cov))))
            diff = np.subtract(mesh,self.mu.T)
            val = diff@self.invCov@diff.T
            val = np.diagonal(val)
            soln = const*np.exp(-1/2*val).T
        return soln



