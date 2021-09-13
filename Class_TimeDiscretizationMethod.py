import Class_PDF
import numpy as np
from Functions import G, alpha1, alpha2
class TimeDiscretizationMethod():
    def __init__(self):
        self
    def computeTransitionMatrix(self):
        pass

    def AddPointToG(self):
        pass

    def RemovePoints(self):
        pass

class EulerMaruyamaTimeDiscretizationMethod(TimeDiscretizationMethod):
    def __init__(self, pdf):
        self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength*5

    # def removePoints(self, index):
    #     self.TransitionMatrix = np.delete(self.TransitionMatrix, index,0)
    #     self.TransitionMatrix = np.delete(self.TransitionMatrix, index,1)


    def computeTransitionMatrix(self, pdf, sde, h):
        GMat = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for i in range(pdf.meshLength):
            v = G(i,pdf.meshCoordinates, h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
            GMat[i,:len(v)] = v
        return GMat

    def AddPointToG(self, meshPartial, newPointindex, parameters, sde, pdf, integrator):
        newRow = G(newPointindex, meshPartial, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
        integrator.TransitionMatrix[newPointindex,:len(newRow)] = newRow
        if sde.dimension == 1:
            var = sde.diffusionFunction(meshPartial[newPointindex,:])**2*parameters.h
            mean = meshPartial[newPointindex,:]+ sde.driftFunction(pdf.meshCoordinates[newPointindex,:])*parameters.h
            newCol = 1/(np.sqrt(2*np.pi*var))*np.exp(-(meshPartial-mean)**2/(2*var))
            integrator.TransitionMatrix[:len(newCol),newPointindex] = np.squeeze(newCol)
            return
            #Now, add new column
        mu = meshPartial[newPointindex,:]+sde.driftFunction(np.expand_dims(meshPartial[newPointindex,:],axis=0))*parameters.h
        mu = mu[0]
        dfn = sde.diffusionFunction(meshPartial[newPointindex,:])
        cov = dfn@dfn.T*parameters.h
        newCol = np.empty(pdf.meshLength)
        const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(cov))))
        covInv = np.linalg.inv(cov)
        for j in range(len(meshPartial)):
            x = meshPartial[j,:]
            Gs = np.exp(-1/2*((x-mu).T@covInv@(x.T-mu.T)))
            newCol[j] = (Gs)
        integrator.TransitionMatrix[:len(newCol),newPointindex] = newCol*const


from Class_PDF import nDGridMeshCenteredAtOrigin
# from pyopoly1.Scaling import GaussScale
from Functions import Gaussian
class AndersonMattinglyTimeDiscretizationMethod(TimeDiscretizationMethod):
    ## TODO: RECHECK THAT RHO ISNT NEEDED, Combine the N2 computations
    def __init__(self):
        self.meshSpacingAM = 0.05
        self.meshAMPadding = 3.5
        self.theta = 0.5
        self.a1 = alpha1(self.theta)
        self.a2 = alpha2(self.theta)
        self.meshAM = None
        self.N2s = None


    def setAndersonMattinglyMeshForComputingTransitionProbability(self, pdf, sde):
        radius = int(max(int(np.ceil(np.max(pdf.meshCoordinates)-np.min(pdf.meshCoordinates))),2)+self.meshAMPadding)/2
        meshAM = nDGridMeshCenteredAtOrigin(sde.dimension, self.meshSpacingAM, radius, UseNoise = False)
        mean = (np.max(pdf.meshCoordinates)+np.min(pdf.meshCoordinates))/2
        delta = np.ones(np.shape(meshAM))*mean
        meshAM = np.asarray(meshAM).T + delta.T
        meshAM = meshAM.T
        self.meshAM = meshAM

    def computeN2s(self, pdf, sde, h):
        N2Complete = []
        count = 0
        for yim1 in sde.meshCoordinates:
            count = count+1
            N2All = []
            scale2 = GaussScale(sde.spatialDiff)
            if sde.spatialDiff == False:
                sig2 = np.sqrt(self.a1*sde.difffun(self.meshAM[0])**2 - self.a2*sde.difffun(self.meshAM[0])**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(np.asarray(sig2**2))

            mu2s = self.meshAM + (self.a1*sde.driftfun(self.meshAM) - self.a2*sde.driftfun(yim1))*(1-self.theta)*h
            for count, i in enumerate(self.meshAM):
                mu2 = np.expand_dims(mu2s[count],1)
                scale2.setMu(np.asarray(mu2.T))
                if sde.spatialDiff == True:
                    sig2 = np.sqrt(self.a1*sde.difffun(i)**2 - self.a2*sde.difffun(yim1)**2)*np.sqrt((1-self.theta)*h)
                    scale2.setCov(np.asarray(sig2**2))
                N2 = Gaussian(scale2, pdf.meshCoordinates)
                N2All.append(np.copy(N2))
            N2Complete.append(np.copy(N2All))
        self.N2s = np.asarray(N2Complete)

    def computeTransitionProbability(self, sde, yi, yim1, h, N2= None):
        mu1 = yim1 + sde.driftfun(yim1)*self.theta*h
        sig1 = abs(sde.difffun(yim1))*np.sqrt(self.theta*h)
        scale = GaussScale(sde.dimension)
        scale.setMu(np.asarray(mu1.T))
        scale.setCov(np.asarray(sig1**2))
        N1 = Gaussian(scale, self.meshAM)

        if N2 == None:
            N2 = []
            scale2 = GaussScale(sde.dimension)
            if sde.spatialDiff == False:
                sig2 = np.sqrt(self.a1*sde.difffun(self.meshAM[0])**2 - self.a2*sde.difffun(self.meshAM[0])**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(np.asarray(sig2**2))
            mu2s = self.meshAM + (self.a1*sde.driftfun(self.meshAM) - self.a2*sde.driftfun(yim1))*(1-self.theta)*h
            for count, i in enumerate(self.meshAM):
                mu2 = np.expand_dims(mu2s[count],1)
                if sde.spatialDiff == True:
                    sig2 = np.sqrt(self.a1*sde.difffun(i)**2 - self.a2*sde.difffun(yim1)**2)*np.sqrt((1-self.theta)*h)
                    scale2.setCov(np.asarray(sig2**2))
                scale2.setMu(np.asarray(mu2.T))
                N2a = Gaussian(scale2, np.asarray([yi])) # depends on yi, yim1, i
                N2.append(np.copy(N2a))

        val = N1*np.asarray(N2)
        transitionProb = np.sum(self.meshSpacingAM**sde.dimension*val)
        return transitionProb

    def computeTransitionMatrix(self, pdf, sde, h):
        sizeMatrix = pdf.meshLength
        matrix = np.empty([sizeMatrix, sizeMatrix])*np.NaN
        self.setMeshForComputingTransitionProbability(pdf, sde)
        self.computeN2s(self, pdf, sde, h)
        for i in range(len(pdf.meshCoordinates)):
            for j in range(len(pdf.meshCoordinates)):
                N2 = self.N2s[j][:,i]
                transitionProb = self.computeTransitionProbability(sde, i, j, h, N2= N2)
                matrix[i,j] = transitionProb
        return matrix








# class TransitionMatrix:
#     def __init__(self):
#         self.transitionMatrix = []









