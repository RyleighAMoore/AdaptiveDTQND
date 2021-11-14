from Class_Gaussian import GaussScale
import numpy as np
from Class_LaplaceApproximation import LaplaceApproximation
import math
from variableTransformations import map_to_canonical_space, map_from_canonical_space
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import opolynd
from LejaPoints import getLejaPoints
from scipy.interpolate import griddata
from Functions import weightExp, G


class Integrator:
    def __init__(self, simulation, sde, parameters, pdf):
        pass

    def computeTimeStep(self, sde, parameters, simulation):
        pass


class IntegratorTrapezoidal(Integrator):
    def __init__(self, dimension, parameters):
        print("Warning: You must have an equispaced mesh!")
        if parameters.useAdaptiveMesh == True:
            print("Updates with the Trapezoidal rule are not currently supported.")
            parameters.useAdaptiveMesh = False
        self.stepSize = parameters.minDistanceBetweenPoints

    def computeTimeStep(self, sde, parameters, simulation):
        vals= np.asarray(self.stepSize**sde.dimension*simulation.TransitionMatrix[:simulation.pdf.meshLength, :simulation.pdf.meshLength]@simulation.pdfTrajectory[-1])
        return np.squeeze(vals)


class IntegratorLejaQuadrature(Integrator):
    def __init__(self, dimension, parameters):
        self.lejaPoints = None
        self.conditionNumberForAcceptingLejaPointsAtNextTimeStep = 1.1
        self.setIdentityScaling(dimension)
        self.setUpPolnomialFamily(dimension)
        self.altMethodLejaPoints, temp = getLejaPoints(parameters.numLejas, np.zeros((dimension,1)), self.poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
        self.laplaceApproximation = LaplaceApproximation(dimension)

    def setUpPolnomialFamily(self, dimension):
        self.poly = HermitePolynomials(rho=0)
        d=dimension
        k = 40
        lambdas = indexing.total_degree_indices(d, k)
        self.poly.lambdas = lambdas

    def setIdentityScaling(self, dimension):
         self.identityScaling = GaussScale(dimension)
         self.identityScaling.setMu(np.zeros(dimension).T)
         self.identityScaling.setCov(np.eye(dimension))

    def findQuadraticFit(self, sde, simulation, parameters, index):
        pdf = simulation.pdf
        ## TODO: Update this so I don't recompute drift and diff everytime
        pdf.setIntegrandBeforeDividingOut(simulation.TransitionMatrix[index,:pdf.meshLength]*pdf.pdfVals)
        if not simulation.LejaPointIndicesBoolVector[index]: # Do not have good Leja points
            orderedPoints, distances, indicesOfOrderedPoints = findNearestKPoints(pdf.meshCoordinates[index], pdf.meshCoordinates, parameters.numQuadFit, getIndices=True)
            quadraticFitMeshPoints = orderedPoints[:parameters.numQuadFit]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[indicesOfOrderedPoints]
            self.laplaceApproximation.copmuteleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints, sde.dimension)
        else:
            quadraticFitMeshPoints = pdf.meshCoordinates[simulation.LejaPointIndicesMatrix[index,:].astype(int)]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[simulation.LejaPointIndicesMatrix[index,:].astype(int)]
            self.laplaceApproximation.copmuteleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints,sde.dimension)

        if np.any(self.laplaceApproximation.constantOfGaussian)==None: # Fit failed
            return False
        else:
            return True

    def divideOutGaussianAndSetIntegrand(self, pdf, sde, index):
        gaussianToDivideOut = self.laplaceApproximation.ComputeDividedOut(pdf, sde.dimension)
        pdf.setIntegrandAfterDividingOut(pdf.integrandBeforeDividingOut/gaussianToDivideOut)


    def setLejaPoints(self, simulation, index, parameters, sde):
        pdf = simulation.pdf
        self.divideOutGaussianAndSetIntegrand(pdf, sde, index)
        if simulation.LejaPointIndicesBoolVector[index]: # Already have LejaPoints
            LejaIndices = simulation.LejaPointIndicesMatrix[index,:].astype(int)
            mesh2 = map_to_canonical_space(pdf.meshCoordinates, self.laplaceApproximation.scalingForGaussian)
            self.lejaPoints = mesh2[LejaIndices,:]
            self.lejaPointsPdfVals = pdf.integrandAfterDividingOut[LejaIndices]
            self.indicesOfLejaPoints = LejaIndices
        else: # Need Leja points.
            mappedMesh = map_to_canonical_space(pdf.meshCoordinates, self.laplaceApproximation.scalingForGaussian)
            self.lejaPoints, self.lejaPointsPdfVals,self.indicesOfLejaPoints,self.lejaSuccess = LP.getLejaSetFromPoints(self.identityScaling, mappedMesh, parameters.numLejas, self.poly, pdf.integrandAfterDividingOut, sde.diffusionFunction, parameters.numPointsForLejaCandidates)
            if self.lejaSuccess ==False: # Failed to get Leja points
                self.lejaPoints = None
                self.lejaPointsPdfVals = None
                self.idicesOfLejaPoints = None

    def computeUpdateWithInterpolatoryQuadrature(self, parameters, pdf, index, sde):
        self.lejaPointsPdfVals = pdf.integrandAfterDividingOut[self.indicesOfLejaPoints]
        V = opolynd.opolynd_eval(self.lejaPoints, self.poly.lambdas[:parameters.numLejas,:], self.poly.ab, self.poly)
        vinv = np.linalg.inv(V)
        value = np.matmul(vinv[0,:], self.lejaPointsPdfVals)
        condNumber = np.sum(np.abs(vinv[0,:]))
        return value, condNumber


    def computeUpdateWithAlternativeMethod(self, sde, parameters, pdf, index):
        self.AltMethodUseCount = self.AltMethodUseCount  + 1
        scaling = GaussScale(sde.dimension)
        scaling.setMu(np.asarray(pdf.meshCoordinates[index,:]+parameters.h*sde.driftFunction(pdf.meshCoordinates[index,:])).T)
        scaling.setCov((parameters.h*sde.diffusionFunction(scaling.mu.T*sde.diffusionFunction(scaling.mu.T))))
        mesh12 = map_from_canonical_space(self.altMethodLejaPoints, scaling)
        meshNearest, distances, indx = findNearestKPoints(scaling.mu, pdf.meshCoordinates,parameters.numQuadFit, getIndices = True)
        pdfNew = pdf.pdfVals[indx]

        pdf12 = np.asarray(griddata(np.squeeze(meshNearest), pdfNew, np.squeeze(mesh12), method='linear', fill_value=np.min(pdf.pdfVals)))
        pdf12[pdf12 < 0] = np.min(pdf.pdfVals)

        ## TDOD: Implement AM option
        transitionMatrixRow = np.expand_dims(G(0,mesh12, parameters.h, sde.driftFunction, sde.diffusionFunction, sde.spatialDiff),1)
        transitionMatrixRow = np.squeeze(transitionMatrixRow)
        if sde.dimension > 1:
            L = np.linalg.cholesky((scaling.cov))
            JacFactor = np.prod(np.diag(L))
        if sde.dimension ==1:
            L = np.sqrt(scaling.cov)
            JacFactor = np.squeeze(L)

        g = weightExp(scaling,mesh12)*1/(np.pi*JacFactor)

        testing = (pdf12*transitionMatrixRow)/g
        u = map_to_canonical_space(mesh12, scaling)
        numSamples = len(u)
        V = opolynd.opolynd_eval(u, self.poly.lambdas[:numSamples,:], self.poly.ab, self.poly)
        vinv = np.linalg.inv(V)
        c = np.matmul(vinv[0,:], testing)
        # print("Use Alt Method")
        return c, np.sum(np.abs(vinv[0,:]))


    def computeTimeStep(self, sde, parameters, simulation):
        LPReuseCount = 0
        self.AltMethodUseCount = 0
        newPdf = np.zeros(simulation.pdf.meshLength)
        pdf = simulation.pdf
        for index, point in enumerate(pdf.meshCoordinates):
            useLejaIntegrationProcedure = self.findQuadraticFit(sde, simulation, parameters, index)
            if not useLejaIntegrationProcedure: # Failed Quadratic Fit
                value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
            else: # Get Leja points
                self.setLejaPoints(simulation, index, parameters,sde)
                if self.lejaSuccess == False: #Getting Leja points failed
                     value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
                     print("failed Leja")
                else: # Continue with integration, try to use Leja points from last step
                    value, condNumber = self.computeUpdateWithInterpolatoryQuadrature(parameters,pdf, index, sde)
                    if condNumber < 1.1:
                        LPReuseCount = LPReuseCount +1
                        simulation.LejaPointIndicesBoolVector[index] = True
                        simulation.LejaPointIndicesMatrix[index,:] = self.indicesOfLejaPoints
                    else: # Continue with integration, use new leja points
                        simulation.LejaPointIndicesBoolVector[index] = False
                        self.setLejaPoints(simulation, index, parameters,sde)
                        if self.lejaSuccess == False:
                            value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
                        else:
                            value, condNumber = self.computeUpdateWithInterpolatoryQuadrature(parameters,pdf, index, sde)
                            if condNumber < 1.1: # Leja points worked really well, likely okay for next time step
                                simulation.LejaPointIndicesBoolVector[index] = True
                                simulation.LejaPointIndicesMatrix[index,:] = self.indicesOfLejaPoints
                            if condNumber > parameters.conditionNumForAltMethod or value < 0: # Nothing worked, use alt method
                                value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
            newPdf[index] =value
        print(LPReuseCount/pdf.meshLength*100, "% Leja Reuse")
        print(self.AltMethodUseCount/pdf.meshLength*100, "% Alt method Use")
        return newPdf



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



