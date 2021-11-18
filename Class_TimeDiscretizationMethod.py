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

    def computeTransitionMatrix(self, pdf, sde, parameters):
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

    def AddPointToG(self, meshPartial, newPointindex, parameters,sde, pdf, integrator, simulation):
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
        simulation.TransitionMatrix[:len(vals), newPointindex] = vals

        newRow = G(newPointindex, meshPartial, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff)
        simulation.TransitionMatrix[newPointindex, :len(newRow)] = newRow



from Class_PDF import nDGridMeshCenteredAtOrigin
from Class_Gaussian import GaussScale
from tqdm import tqdm
from LejaPoints import getLejaPoints
from Class_Integrator import IntegratorLejaQuadrature
from variableTransformations import map_to_canonical_space, map_from_canonical_space
import opolynd


class AndersonMattinglyTimeDiscretizationMethod(TimeDiscretizationMethod):
    ## TODO: RECHECK THAT RHO ISNT NEEDED, Combine the N2 computations
    def __init__(self, pdf, parameters, dimension):
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
        self.integrator = IntegratorLejaQuadrature(dimension, parameters)


    def setAndersonMattinglyMeshAroundPoint(self, point, sde, radius):
        if sde.dimension ==1:
            radius =6*radius
        else:
            radius = 6*radius

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


    def computeN2Row(self, pdf, sde, h, yim1, meshAMr):
        count1 = 0
        s = np.size(self.meshAM,0)
        N2Complete2 = np.zeros((len(meshAMr), 1))

        scale2 = GaussScale(sde.dimension)
        if sde.spatialDiff == False:
            sig2 = np.sqrt(self.a1*sde.diffusionFunction(meshAMr[0])**2 - self.a2*sde.diffusionFunction(meshAMr)**2)*np.sqrt((1-self.theta)*h)
            scale2.setCov(sig2**2)

        mu2s = meshAMr + (self.a1*sde.driftFunction(meshAMr) - self.a2*sde.driftFunction(yim1))*(1-self.theta)*h
        for count, i in enumerate(meshAMr):
            mu2 = mu2s[[count],:]
            scale2.setMu(mu2.T)
            if sde.spatialDiff == True:
                sig2 = np.sqrt(self.a1*sde.diffusionFunction(i)**2 - self.a2*sde.diffusionFunction(yim1)**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(sig2**2)
            # N2 = Gaussian(scale2, pdf.meshCoordinates)
            N2 = scale2.ComputeGaussian(yim1, sde.dimension)
            N2Complete2[count] = N2
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


    def computeTransitionMatrixLQ3(self, pdf, sde, parameters):
        self.meshSpacingAM = 0.1
        matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for j in trange(pdf.meshLength):
            mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1**2))

            self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
            N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
            N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])

            product = N1

            self.integrator.laplaceApproximation.computeleastSquares(self.meshAM, product, sde.dimension)
            # print(self.integrator.laplaceApproximation.scalingForGaussian.cov)
            if self.integrator.laplaceApproximation.scalingForGaussian == None:
                value = 0
                condNumber = 1
            else:
                self.meshAM = map_from_canonical_space(self.integrator.altMethodLejaPoints, self.integrator.laplaceApproximation.scalingForGaussian)
                N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
                N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])
                # plt.scatter(self.meshAM, N2[i,:]*N1)
                for i in range(pdf.meshLength):
                    pdf.setIntegrandBeforeDividingOut(N2[i,:]*N1)

                    vals = self.integrator.laplaceApproximation.ComputeDividedOutAM(pdf, sde.dimension, self.meshAM)
                    pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
                    V = opolynd.opolynd_eval(self.integrator.altMethodLejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
                    vinv = np.linalg.inv(V)
                    if sde.dimension > 1:
                        L = np.linalg.cholesky((scale1.cov))
                        JacFactor = np.prod(np.diag(L))
                    if sde.dimension ==1:
                        L = np.sqrt(scale1.cov)
                        JacFactor = np.squeeze(L)

                    value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut)
                    condNumber = np.sum(np.abs(vinv[0,:]))
                    # print(condNumber)
                    matrix[i,j] = value
        # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
        return matrix

    def computeTransitionMatrixLQ2(self, pdf, sde, parameters):
        self.meshSpacingAM = 0.2

        matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for j in trange(pdf.meshLength):
            mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1**2))

            self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
            N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
            N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])

            for i in range(pdf.meshLength):
                # plt.scatter(self.meshAM, N2[i,:]*N1)
                product = N2[i,:]*N1
                self.integrator.laplaceApproximation.computeleastSquares(self.meshAM, product, sde.dimension)
                # print(self.integrator.laplaceApproximation.scalingForGaussian.mu)
                if self.integrator.laplaceApproximation.scalingForGaussian == None:
                    value = 0
                    condNumber = 1
                else:
                    meshAMr = map_from_canonical_space(self.integrator.altMethodLejaPoints, self.integrator.laplaceApproximation.scalingForGaussian)
                    N1r = scale1.ComputeGaussian(meshAMr, sde.dimension)
                    N2r = self.computeN2Row(pdf, sde, parameters.h, np.expand_dims(pdf.meshCoordinates[i],1).T, meshAMr)
                    # N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])

                    # plt.scatter(self.meshAM, N2[i,:]*N1)

                    pdf.setIntegrandBeforeDividingOut(N2r.T*N1r)

                    vals = self.integrator.laplaceApproximation.ComputeDividedOutAM(pdf, sde.dimension, meshAMr)
                    pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
                    V = opolynd.opolynd_eval(self.integrator.altMethodLejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
                    vinv = np.linalg.inv(V)
                    if sde.dimension > 1:
                        L = np.linalg.cholesky((scale1.cov))
                        JacFactor = np.prod(np.diag(L))
                    if sde.dimension ==1:
                        L = np.sqrt(scale1.cov)
                        JacFactor = np.squeeze(L)

                    value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut.T)
                    condNumber = np.sum(np.abs(vinv[0,:]))
                    # print(condNumber)
                matrix[i,j] = value
        # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
        return matrix


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

            # self.integrator.laplaceApproximation.copmuteleastSquares(self.meshCoordinates, val, sde.dimension)
            # print(self.integrator.laplaceApproximation.scalingForGaussian)
            # fig = pyplot.figure()
            # ax = Axes3D(fig)
            # ax.scatter(self.meshAM[:,0], self.meshAM[:,1], N1)
            # # ax.scatter(pdf.meshCoordinates[:,0], pdf.meshCoordinates[:,1], val)

            # pyplot.show()
            matrix[:len(val),j] = np.squeeze(val)
        return matrix


    def AddPointToG(self, simulation, newPointindices, parameters, integrator, sde):
        pdf = simulation.pdf
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
            simulation.TransitionMatrix[:len(pdf.meshCoordinates),newPointindices[index]] = np.squeeze(vals)

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

            simulation.TransitionMatrix[pdf.meshLength-len(newPointindices):pdf.meshLength, count] = np.squeeze(vals)
            count = count +1
        # matrix = self.computeTransitionMatrix(pdf, sde, parameters.h)
        # print(np.nanmax(abs(matrix - integrator.TransitionMatrix[:pdf.meshLength,:pdf.meshLength])))

        # assert np.isclose(10**(-16),np.max(abs(matrix - integrator.TransitionMatrix[:pdf.meshLength,:pdf.meshLength])))
        t=0











