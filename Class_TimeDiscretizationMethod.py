import numpy as np
from tqdm import trange

from Class_PDF import nDGridMeshCenteredAtOrigin
from Class_Gaussian import GaussScale
from Class_Integrator import IntegratorLejaQuadrature
from variableTransformations import map_to_canonical_space, map_from_canonical_space
import opolynd


class TimeDiscretizationMethod():
    def __init__(self):
        pass

    def computeTransitionMatrix(self):
        pass

    def AddPointToG(self):
        pass

    def RemovePoints(self):
        pass

class EulerMaruyamaTimeDiscretizationMethod(TimeDiscretizationMethod):
    def __init__(self, pdf, parameters):
        '''
        Manages the time discretization for the Euler-Maruyama method

        Parameters:
        pdf: manages the probability density function of the solution of the SDE (PDF class object)
        parameters: parameters for the simulation (class object)
        '''
        if parameters.useAdaptiveMesh:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength*10
        else:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength

    def computeTransitionMatrix(self, pdf, sde, parameters):
        GMat = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for indexOfMesh in trange(pdf.meshLength):
            y = pdf.meshCoordinates[indexOfMesh,:]
            GMat[:pdf.meshLength, indexOfMesh] = self.computeTransitionMatrixColumn(y, sde, parameters, pdf)
        return GMat

    def computeTransitionMatrixColumn(self, y, sde, parameters, pdf):
        scale1 = GaussScale(sde.dimension)
        mu = y+sde.driftFunction(y)*parameters.h
        scale1.setMu(np.asarray(mu.T))

        diff_y = sde.diffusionFunction(y)
        cov = diff_y@diff_y.T * parameters.h
        scale1.setCov(cov)
        vals = scale1.ComputeGaussian(pdf.meshCoordinates, sde.dimension)
        return vals

    def computeTransitionMatrixRow(self, point, mesh, h, sde):
        mean = mesh+sde.driftFunction(mesh)*h
        if sde.dimension == 1:
            newpointVect = point*np.ones(np.shape(mesh))
            var = h*sde.diffusionFunction(mesh)**2
            newVals = 1/(np.sqrt((2*np.pi*var)))*np.exp(-(newpointVect-mean)**2/(2*var))
            return np.squeeze(newVals)

        if not sde.spatialDiff:
            '''diff(y) = diff(x) since diffusion constant. So we can set it once beforehand.'''
            diff = sde.diffusionFunction(point)
            cov = diff@diff.T * h
            const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(cov))))
            invCov = np.linalg.inv(cov)

        soln_vals = np.empty(len(mesh))
        for j in range(len(mesh)):
            '''Changing mu and cov over each column as needed.'''
            if sde.spatialDiff:
                y = mesh[j,:]
                diff_y = sde.diffusionFunction(y)
                cov = diff_y@diff_y.T * h
                const = 1/(np.sqrt((2*np.pi)**sde.dimension*abs(np.linalg.det(cov))))
                invCov = np.linalg.inv(cov)
            mu = mean[j,:]
            Gs = np.exp(-1/2*((point-mu).T@invCov@(point.T-mu.T)))
            soln_vals[j] = Gs
        return soln_vals*const

    def AddPointToG(self, meshPartial, newPointindex, parameters,sde, pdf, integrator, simulation):
        y = pdf.meshCoordinates[newPointindex,:]
        simulation.TransitionMatrix[:pdf.meshLength, newPointindex] = self.computeTransitionMatrixColumn(y, sde, parameters, pdf)

        newRow = self.computeTransitionMatrixRow(meshPartial[newPointindex], meshPartial, parameters.h,sde)
        simulation.TransitionMatrix[newPointindex, :len(newRow)] = newRow




class AndersonMattinglyTimeDiscretizationMethod(TimeDiscretizationMethod):
    ## TODO: RECHECK THAT RHO ISNT NEEDED, Combine the N2 computations
    ##TODO: Finish this method
    def __init__(self, pdf, parameters, dimension):
        '''
        This class is still in development.

        Manages the time discretization for the Anderson-Mattingly method

        Parameters:
        pdf: manages the probability density function of the solution of the SDE (PDF class object)
        parameters: parameters: parameters defined by the user (Parameters class object)
        '''
        if parameters.useAdaptiveMesh:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength*3
        else:
            self.sizeTransitionMatrixIncludingEmpty =  pdf.meshLength
        self.meshSpacingAM = parameters.AMMeshSpacing
        self.theta = 0.5
        self.a1 = self.alpha1(self.theta)
        self.a2 = self.alpha2(self.theta)
        self.meshAM = None
        self.N2s = None
        self.integrator = IntegratorLejaQuadrature(dimension, parameters, parameters.timeDiscretizationType)

    def alpha1(self, theta):
        return(1/(2*theta*(1-theta)))

    def alpha2(self, theta):
      num = (1-theta)**2 + theta**2
      denom = 2*theta*(1-theta)
      return(num/denom)


    def setAndersonMattinglyMeshAroundPoint(self, point, sde, radius, Noise = False):
        if sde.dimension ==1:
            radius =20*radius
        else:
            radius = 6*radius

        meshAM = nDGridMeshCenteredAtOrigin(sde.dimension, radius,self.meshSpacingAM, useNoiseBool = Noise)
        mean = point
        delta = np.ones(np.shape(meshAM))*mean
        meshAM = np.asarray(meshAM).T + delta.T
        meshAM = meshAM.T
        self.meshAM = meshAM


    def computeN2(self, pdf, sde, h, yim1):
        count1 = 0
        s = np.size(self.meshAM,0)
        N2Complete2 = np.zeros((len(pdf.meshCoordinates),s))

        scale2 = GaussScale(sde.dimension)
        if sde.spatialDiff == False:
            sig2 = np.sqrt(self.a1*sde.diffusionFunction(self.meshAM[0])**2 - self.a2*sde.diffusionFunction(self.meshAM[0])**2)*np.sqrt((1-self.theta)*h)
            scale2.setCov(sig2@sig2)

        mu2s = self.meshAM + (self.a1*sde.driftFunction(self.meshAM) - self.a2*sde.driftFunction(yim1))*(1-self.theta)*h
        for count, i in enumerate(self.meshAM):
            mu2 = mu2s[[count],:]
            scale2.setMu(mu2.T)
            if sde.spatialDiff == True:
                sig2 = np.sqrt(self.a1*sde.diffusionFunction(i)**2 - self.a2*sde.diffusionFunction(yim1)**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(sig2@sig2)
            # N2 = Gaussian(scale2, pdf.meshCoordinates)
            N2 = scale2.ComputeGaussian(pdf.meshCoordinates, sde.dimension)
            N2Complete2[:,count] = N2
        return N2Complete2


    def computeN2Row(self, pdf, sde, h, yim1, meshAMr):
        N2Complete2 = np.zeros((len(meshAMr), 1))

        scale2 = GaussScale(sde.dimension)
        if sde.spatialDiff == False:
            sig2 = np.sqrt(self.a1*sde.diffusionFunction(meshAMr[0])**2 - self.a2*sde.diffusionFunction(meshAMr)**2)*np.sqrt((1-self.theta)*h)
            scale2.setCov(sig2@sig2)

        mu2s = meshAMr + (self.a1*sde.driftFunction(meshAMr) - self.a2*sde.driftFunction(yim1))*(1-self.theta)*h
        for count, i in enumerate(meshAMr):
            mu2 = mu2s[[count],:]
            scale2.setMu(mu2.T)
            if sde.spatialDiff == True:
                sig2 = np.sqrt(self.a1*sde.diffusionFunction(i)**2 - self.a2*sde.diffusionFunction(yim1)**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(sig2@sig2)
            # N2 = Gaussian(scale2, pdf.meshCoordinates)
            N2 = scale2.ComputeGaussian(yim1, sde.dimension)
            N2Complete2[count] = N2
        return N2Complete2

    def computeN2Paritial(self, sde, h, yim1, meshNew):
        s = np.size(self.meshAM,0)
        N2Complete2 = np.zeros((len(meshNew),s))

        scale2 = GaussScale(sde.dimension)
        if sde.spatialDiff == False:
            sig2 = np.sqrt(self.a1*sde.diffusionFunction(self.meshAM[0])**2 - self.a2*sde.diffusionFunction(self.meshAM[0])**2)*np.sqrt((1-self.theta)*h)
            scale2.setCov(sig2@sig2)

        mu2s = self.meshAM + (self.a1*sde.driftFunction(self.meshAM) - self.a2*sde.driftFunction(yim1))*(1-self.theta)*h
        for count, i in enumerate(self.meshAM):
            mu2 = mu2s[[count],:]
            scale2.setMu(mu2.T)
            if sde.spatialDiff == True:
                sig2 = np.sqrt(self.a1*sde.diffusionFunction(i)**2 - self.a2*sde.diffusionFunction(yim1)**2)*np.sqrt((1-self.theta)*h)
                scale2.setCov(sig2@sig2)
            N2 = scale2.ComputeGaussian(meshNew, sde.dimension)
            N2Complete2[:,count] = N2
        return N2Complete2

    def computeN1(self, sde, yim1, h, scale):
        mu1 = yim1 + sde.driftFunction(np.asarray([yim1]))*self.theta*h
        sig1 = abs(sde.diffusionFunction(np.asarray([yim1])))*np.sqrt(self.theta*h)
        scale = GaussScale(sde.dimension)
        scale.setMu(np.asarray(mu1.T))
        scale.setCov(np.asarray(sig1@sig1))
        # N1 = Gaussian(scale, self.meshAM)
        N1 = scale.ComputeGaussian(self.meshAM, sde)

        return N1

    def computeTransitionProbability(self, sde, yim1, h, N2):
        N1 = self.computeN1(sde, yim1, h)
        val = N1*np.asarray(N2)
        transitionProb = np.sum(self.meshSpacingAM**sde.dimension*val)
        return transitionProb

    def computeTransitionMatrix(self, pdf, sde, parameters):
        '''See MiscCodeNotForProcedure/AMDiscretization.py for more drafts of code to compute the AM Kernel'''
        matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
        for j in trange(pdf.meshLength):
            mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1@sig1))

            self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
            N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])
            N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)

            val = self.meshSpacingAM**sde.dimension*N2@np.expand_dims(N1,1)
            matrix[:len(val),j] = np.squeeze(val)
        return matrix

    def computeTransitionMatrixRow(self, point, mesh, h, sde, fullMesh = None, newPointIndices_AM = None):
        mu1= point+sde.driftFunction(np.asarray([point]))*self.theta*h
        sig1 = abs(sde.diffusionFunction(np.asarray([point]))*np.sqrt(self.theta*h))
        scale1 = GaussScale(sde.dimension)
        scale1.setMu(np.asarray(mu1.T))
        scale1.setCov(np.asarray(sig1@sig1))

        self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
        N22 = self.computeN2Paritial(sde, h, point,fullMesh[newPointIndices_AM])
        N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
        vals = self.meshSpacingAM**sde.dimension*N22@np.expand_dims(N1,1)
        return vals

    def AddPointToG(self, simulation, newPointindices, parameters, integrator, sde):
        pdf = simulation.pdf
        for index, point in enumerate(pdf.meshCoordinates[newPointindices]):
            mu1= point+sde.driftFunction(np.asarray([point]))*self.theta*parameters.h
            sig1 = abs(sde.diffusionFunction(np.asarray([point]))*np.sqrt(self.theta*parameters.h))
            scale1 = GaussScale(sde.dimension)
            scale1.setMu(np.asarray(mu1.T))
            scale1.setCov(np.asarray(sig1@sig1))
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
            vals = self.computeTransitionMatrixRow(point, pdf.meshCoordinates, parameters.h, sde, fullMesh = pdf.meshCoordinates, newPointIndices_AM = newPointindices)
            simulation.TransitionMatrix[pdf.meshLength-len(newPointindices):pdf.meshLength, count] = np.squeeze(vals)
            count = count +1



def findNearestKPoints(Coord, AllPoints, numNeighbors, getIndices = False):
    normList = np.zeros(np.size(AllPoints,0))
    size = np.size(AllPoints,0)
    for i in range(np.size(AllPoints,1)):
        normList += (Coord[i]*np.ones(size) - AllPoints[:,i])**2
    idx = np.argsort(normList)
    if getIndices:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]], idx[:numNeighbors]
    else:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]]




