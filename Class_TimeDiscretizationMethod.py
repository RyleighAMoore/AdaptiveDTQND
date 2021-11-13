import Class_PDF
import numpy as np
from Functions import G, alpha1, alpha2
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


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
    def __init__(self, pdf, parameters):
        if parameters.useAdaptiveMesh:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength*3
        else:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength

    # def removePoints(self, index):
    #     self.TransitionMatrix = np.delete(self.TransitionMatrix, index,0)
    #     self.TransitionMatrix = np.delete(self.TransitionMatrix, index,1)


    # def computeTransitionMatrix(self, pdf, sde, parameters):
    #     GMat = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     for indexOfMesh in trange(pdf.meshLength):
    #         # '''Changing mu and cov over each row'''
    #         # x = pdf.meshCoordinates[indexOfMesh,:]
    #         # D = sde.dimension
    #         # mean = pdf.meshCoordinates+sde.driftFunction(pdf.meshCoordinates)*parameters.h

    #         # if D == 1:
    #         #     newpointVect = x*np.ones(np.shape(pdf.meshCoordinates))
    #         #     var = parameters.h*sde.diffusionFunction(pdf.meshCoordinates)**2
    #         #     newVals = 1/(np.sqrt((2*np.pi*var)))*np.exp(-(newpointVect-mean)**2/(2*var))
    #         #     v1 = np.squeeze(newVals)
    #         # else:
    #         #     if not sde.spatialDiff:
    #         #             cov = sde.diffusionFunction(x)@sde.diffusionFunction(x).T * parameters.h
    #         #             const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    #         #             invCov = np.linalg.inv(cov)

    #         #     soln_vals = np.zeros(pdf.meshLength)
    #         #     for j in range(pdf.meshLength):
    #         #         if sde.spatialDiff:
    #         #             m = pdf.meshCoordinates[indexOfMesh,:]
    #         #             cov = sde.diffusionFunction(m)@sde.diffusionFunction(m).T * parameters.h
    #         #             const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    #         #             invCov = np.linalg.inv(cov)
    #         #         mu = mean[j,:]
    #         #         Gs = np.exp(-1/2*((x-mu).T@invCov@(x.T-mu.T)))
    #         #         soln_vals[j] = Gs
    #         #     v1 = soln_vals*const


    #         v = G(indexOfMesh,pdf.meshCoordinates, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
    #         GMat[indexOfMesh,:len(v)] = v
    #         # assert np.all(v==v1)
    #     return GMat

    def computeTransitionMatrix(self, pdf, sde, parameters):
        # test = self.computeTransitionMatrixO(pdf, sde, parameters)
        # GMat2 = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        # for indexOfMesh in trange(pdf.meshLength):
        #     v = G(indexOfMesh,pdf.meshCoordinates, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
        #     GMat2[indexOfMesh,:len(v)] = v
        GMat = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for indexOfMesh in range(pdf.meshLength):
            m = pdf.meshCoordinates[indexOfMesh,:]
            D = sde.dimension
            scale1 = GaussScale(sde.dimension)
            mu = m+sde.driftFunction(m)*parameters.h
            scale1.setMu(np.asarray(mu.T))
            if D == 1:
                var = parameters.h*sde.diffusionFunction(m)**2
                scale1.setCov(np.asarray(var))
            else:
                cov = sde.diffusionFunction(m)@sde.diffusionFunction(m).T * parameters.h
                scale1.setCov(cov)

            soln_vals = np.zeros(pdf.meshLength)
            vals = scale1.ComputeGaussian(pdf.meshCoordinates, sde.dimension)

            GMat[:len(vals), indexOfMesh] = vals
        return GMat

    def AddPointToG(self, meshPartial, newPointindex, parameters,sde, pdf, integrator):
        m = pdf.meshCoordinates[newPointindex,:]
        D = sde.dimension
        scale1 = GaussScale(sde.dimension)
        mu = m+sde.driftFunction(m)*parameters.h
        scale1.setMu(np.asarray(mu.T))
        if D == 1:
            var = parameters.h*sde.diffusionFunction(m)**2
            scale1.setCov(np.asarray(var))
        else:
            cov = sde.diffusionFunction(m)@sde.diffusionFunction(m).T * parameters.h
            scale1.setCov(cov)

        soln_vals = np.zeros(pdf.meshLength)
        vals = scale1.ComputeGaussian(pdf.meshCoordinates, sde.dimension)
        integrator.TransitionMatrix[:len(vals), newPointindex] = vals

        newRow = G(newPointindex, meshPartial, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
        integrator.TransitionMatrix[newPointindex, :len(newRow)] = newRow




    # def AddPointToG(self, meshPartial, newPointindex, parameters, sde, pdf, integrator):
    #     integrator.TransitionMatrix = self.computeTransitionMatrix(pdf, sde, parameters)
    #     newRow = G(newPointindex, meshPartial, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
    #     integrator.TransitionMatrix[newPointindex,:len(newRow)] = newRow
    #     if sde.dimension == 1:
    #         var = sde.diffusionFunction(meshPartial[newPointindex,:])**2*parameters.h
    #         mean = meshPartial[newPointindex,:]+ sde.driftFunction(pdf.meshCoordinates[newPointindex,:])*parameters.h
    #         newCol = 1/(np.sqrt(2*np.pi*var))*np.exp(-(meshPartial-mean)**2/(2*var))
    #         integrator.TransitionMatrix[:len(newCol),newPointindex] = np.squeeze(newCol)
    #         return
    #         #Now, add new column
    #     mu = meshPartial[newPointindex,:]+sde.driftFunction(np.expand_dims(meshPartial[newPointindex,:],axis=0))*parameters.h
    #     mu = mu[0]
    #     dfn = sde.diffusionFunction(meshPartial[newPointindex,:])
    #     cov = dfn@dfn.T*parameters.h
    #     newCol = np.empty(pdf.meshLength)
    #     const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(cov))))
    #     covInv = np.linalg.inv(cov)
    #     for j in range(len(meshPartial)):
    #         x = meshPartial[j,:]
    #         Gs = np.exp(-1/2*((x-mu).T@covInv@(x.T-mu.T)))
    #         newCol[j] = (Gs)
    #     integrator.TransitionMatrix[:len(newCol),newPointindex] = newCol*const


from Class_PDF import nDGridMeshCenteredAtOrigin
from Class_Gaussian import GaussScale
from tqdm import tqdm

class AndersonMattinglyTimeDiscretizationMethod(TimeDiscretizationMethod):
    ## TODO: RECHECK THAT RHO ISNT NEEDED, Combine the N2 computations
    def __init__(self, pdf, parameters):
        if parameters.useAdaptiveMesh:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength*3
        else:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength
        self.meshSpacingAM = parameters.AMMeshSpacing
        self.theta = 0.5
        self.a1 = alpha1(self.theta)
        self.a2 = alpha2(self.theta)
        self.meshAM = None
        self.N2s = None

    def setAndersonMattinglyMeshAroundPoint(self, point, sde, radius):
        if sde.dimension ==1:
            radius =5*radius
        else:
            radius = 5*radius

        meshAM = nDGridMeshCenteredAtOrigin(sde.dimension, radius,self.meshSpacingAM, useNoiseBool = False)
        mean = point
        delta = np.ones(np.shape(meshAM))*mean
        meshAM = np.asarray(meshAM).T + delta.T
        meshAM = meshAM.T
        self.meshAM = meshAM


    # @profile
    def computeN2(self, pdf, sde, h, yim1):
        count1 = 0
        s = np.size(self.meshAM,0)
        N2Complete2 = np.zeros((pdf.meshLength,s))

        scale2 = GaussScale(sde.dimension)
        if sde.spatialDiff == False:
            sig2 = np.sqrt(self.a1*sde.diffusionFunction(self.meshAM[0])**2 - self.a2*sde.diffusionFunction(self.meshAM[0])**2)*np.sqrt((1-self.theta)*h)
            scale2.setCov(sig2**2)

        mu2s = self.meshAM + (self.a1*sde.driftFunction(self.meshAM) - self.a2*sde.driftFunction(yim1))*(1-self.theta)*h
        for count, i in enumerate(self.meshAM):
            mu2 = mu2s[[count],:]
            scale2.setMu(mu2.T)
            if sde.spatialDiff == True:
                sig2 = np.sqrt(self.a1*sde.diffusionFunction(i)**2 - self.a2*sde.diffusionFunction(yim1)**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(sig2**2)
            # N2 = Gaussian(scale2, pdf.meshCoordinates)
            N2 = scale2.ComputeGaussian(pdf.meshCoordinates, sde.dimension)
            N2Complete2[:,count] = N2
        return N2Complete2

    # @profile
    def computeN2Paritial(self, pdf, sde, h, yim1, meshNew):

        s = np.size(self.meshAM,0)
        N2Complete2 = np.zeros((len(meshNew),s))

        scale2 = GaussScale(sde.dimension)
        if sde.spatialDiff == False:
            sig2 = np.sqrt(self.a1*sde.diffusionFunction(self.meshAM[0])**2 - self.a2*sde.diffusionFunction(self.meshAM[0])**2)*np.sqrt((1-self.theta)*h)
            scale2.setCov(sig2**2)

        mu2s = self.meshAM + (self.a1*sde.driftFunction(self.meshAM) - self.a2*sde.driftFunction(yim1))*(1-self.theta)*h
        for count, i in enumerate(self.meshAM):
            mu2 = mu2s[[count],:]
            scale2.setMu(mu2.T)
            if sde.spatialDiff == True:
                sig2 = np.sqrt(self.a1*sde.diffusionFunction(i)**2 - self.a2*sde.diffusionFunction(yim1)**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(sig2**2)
            N2 = scale2.ComputeGaussian(meshNew, sde.dimension)
            N2Complete2[:,count] = N2
        return N2Complete2

    def computeN1(self, sde, yim1, h, scale):
        mu1 = yim1 + sde.driftFunction(np.asarray([yim1]))*self.theta*h
        sig1 = abs(sde.diffusionFunction(np.asarray([yim1])))*np.sqrt(self.theta*h)
        scale = GaussScale(sde.dimension)
        scale.setMu(np.asarray(mu1.T))
        scale.setCov(np.asarray(sig1**2))
        # N1 = Gaussian(scale, self.meshAM)
        N1 = scale.ComputeGaussian(self.meshAM, sde)

        return N1

    def computeTransitionProbability(self, sde, yim1, h, N2):
        N1 = self.computeN1(sde, yim1, h)
        val = N1*np.asarray(N2)
        transitionProb = np.sum(self.meshSpacingAM**sde.dimension*val)
        return transitionProb


    def computeTransitionMatrix(self, pdf, sde, parameters):
        matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for j in trange(pdf.meshLength):
            mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1**2))

            self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
            N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])
            N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)

            val = self.meshSpacingAM**sde.dimension*N2@np.expand_dims(N1,1)

            # fig = pyplot.figure()
            # ax = Axes3D(fig)
            # ax.scatter(self.meshAM[:,0], self.meshAM[:,1], N1)
            # # ax.scatter(pdf.meshCoordinates[:,0], pdf.meshCoordinates[:,1], val)

            # pyplot.show()
            matrix[:len(val),j] = np.squeeze(val)
        return matrix

    # def AddPointToG(self, pdf, newPointindices, parameters, integrator, sde):
    #     # aaa= self.computeTransitionMatrix(pdf, sde, parameters.h)
    #     # self.TransitionMatrix[:pdf.meshLength, :pdf.meshLength] = aaa
    #     self.computeN2s(pdf, sde, parameters.h)
    #     newPointsCount = len(newPointindices)

    #     for i in range(pdf.meshLength-newPointsCount, pdf.meshLength):
    #         for j in range(pdf.meshLength):
    #             N2 = self.N2s[j][:,i]
    #             transitionProb = self.computeTransitionProbability(sde, pdf.meshCoordinates[i], pdf.meshCoordinates[j], parameters.h, N2= N2)
    #             integrator.TransitionMatrix[i,j] = transitionProb

    #     for i in range(pdf.meshLength):
    #         for j in range(pdf.meshLength-newPointsCount, pdf.meshLength):
    #             N2 = self.N2s[j][:,i]
    #             transitionProb = self.computeTransitionProbability(sde, pdf.meshCoordinates[i], pdf.meshCoordinates[j], parameters.h, N2= N2)
    #             integrator.TransitionMatrix[i,j] = transitionProb
    #     t=0

    # @profile
    def AddPointToG(self, pdf, newPointindices, parameters, integrator, sde):
        # if len(newPointindices) < 5:
        #     pass
        for index, point in enumerate(pdf.meshCoordinates[newPointindices]):
            mu1= point+sde.driftFunction(np.asarray([point]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([point]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1**2))
            self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))

            # Add column
            N2 = self.computeN2(pdf, sde, parameters.h, point)
            N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
            vals = self.meshSpacingAM**sde.dimension*N2@np.expand_dims(N1,1)
            integrator.TransitionMatrix[:len(pdf.meshCoordinates),newPointindices[index]] = np.squeeze(vals)

        # Add row
        count = 0
        for index in range(pdf.meshLength):
            point = pdf.meshCoordinates[count]
            mu1= point+sde.driftFunction(np.asarray([point]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([point]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1**2))

            self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
            N22 = self.computeN2Paritial(pdf, sde, parameters.h, pdf.meshCoordinates[count], pdf.meshCoordinates[newPointindices])
            N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
            vals = self.meshSpacingAM**sde.dimension*N22@np.expand_dims(N1,1)

            integrator.TransitionMatrix[pdf.meshLength-len(newPointindices):pdf.meshLength, count] = np.squeeze(vals)
            count = count +1
        # matrix = self.computeTransitionMatrix(pdf, sde, parameters.h)
        # print(np.nanmax(abs(matrix - integrator.TransitionMatrix[:pdf.meshLength,:pdf.meshLength])))

        # assert np.isclose(10**(-16),np.max(abs(matrix - integrator.TransitionMatrix[:pdf.meshLength,:pdf.meshLength])))
        t=0











