import numpy as np
import indexing
from LejaPoints import getLejaPoints
from scipy.interpolate import griddata

from Class_Gaussian import GaussScale
from Class_LaplaceApproximation import LaplaceApproximation
from variableTransformations import map_to_canonical_space, map_from_canonical_space
from families import HermitePolynomials
import LejaPoints as LP
import opolynd

np.seterr(divide='ignore', invalid='ignore')

class Integrator:
    def __init__(self, simulation, sde, parameters, pdf):
        pass

    def computeTimeStep(self, sde, parameters, simulation):
        pass


class IntegratorTrapezoidal(Integrator):
    def __init__(self, dimension, parameters):
        print("Warning: Please make sure you have an equispaced mesh. Otherwise, this equispaced trapezoidal rule is not accurate.")
        if parameters.useAdaptiveMesh == True:
            print("Warning: Mesh updates with the Trapezoidal rule are not currently supported.")
            parameters.useAdaptiveMesh = False
        self.stepSize = parameters.minDistanceBetweenPoints

    def computeTimeStep(self, sde, parameters, simulation):
        vals= np.asarray(self.stepSize**sde.dimension*simulation.TransitionMatrix@simulation.pdf.pdfVals)
        return np.squeeze(vals), 0, 0


class IntegratorLejaQuadrature(Integrator):
    def __init__(self, dimension, parameters, timeDiscretiazationMethod):
        self.lejaPoints = None
        self.setIdentityScaling(dimension)
        self.setUpPolnomialFamily(dimension)
        self.altMethodLejaPoints, temp = getLejaPoints(parameters.numLejas, np.zeros((dimension,1)), self.poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
        self.laplaceApproximation = LaplaceApproximation(dimension)
        self.timeDiscretiazationMethod = timeDiscretiazationMethod

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
        pdf.setIntegrandBeforeDividingOut(simulation.TransitionMatrix[index,:pdf.meshLength]*pdf.pdfVals)

        if not simulation.LejaPointIndicesBoolVector[index]:
            '''Do not have Leja points.'''
            orderedPoints, distances, indicesOfOrderedPoints = findNearestKPoints(pdf.meshCoordinates[index], pdf.meshCoordinates, parameters.numQuadFit, getIndices=True)
            laplaceApproximationMeshPoints = orderedPoints[:parameters.numQuadFit]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[indicesOfOrderedPoints]
            self.laplaceApproximation.computeleastSquares(laplaceApproximationMeshPoints, pdfValuesOfQuadraticFitPoints, sde.dimension)

            '''Have Leja points. Compute fit on Leja points'''
        else:
            laplaceApproximationMeshPoints = pdf.meshCoordinates[simulation.LejaPointIndicesMatrix[index,:].astype(int)]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[simulation.LejaPointIndicesMatrix[index,:].astype(int)]
            self.laplaceApproximation.computeleastSquares(laplaceApproximationMeshPoints, pdfValuesOfQuadraticFitPoints,sde.dimension)

        '''We don't want a 0 value since we need to take the log.'''
        if np.min(pdf.integrandBeforeDividingOut)==0:
            pdf.integrandBeforeDividingOut[pdf.integrandBeforeDividingOut ==0] = 1e-16
        if np.any(self.laplaceApproximation.constantOfGaussian)==None:
            return False # Fit failed
        else:
            return True # Fit succeeded

    def divideOutGaussianAndSetIntegrand(self, pdf, sde, index):
        gaussianToDivideOut = self.laplaceApproximation.ComputeDividedOut(pdf.meshCoordinates[self.indicesOfLejaPoints], sde.dimension)
        integrand = pdf.integrandBeforeDividingOut[self.indicesOfLejaPoints]/gaussianToDivideOut
        pdf.setIntegrandAfterDividingOut(integrand)

    def reuseLejaPoints(self, simulation, index, parameters, sde):
        assert simulation.LejaPointIndicesBoolVector[index], "Leja points weren't valid." # Check that we have LejaPoints
        LejaIndices = simulation.LejaPointIndicesMatrix[index,:].astype(int)
        mesh2 = map_to_canonical_space(simulation.pdf.meshCoordinates[LejaIndices,:], self.laplaceApproximation.scalingForGaussian)
        self.lejaPoints = mesh2
        self.indicesOfLejaPoints = LejaIndices

    def computeLejaPoints(self, simulation, index, parameters, sde):
        mappedMesh = map_to_canonical_space(simulation.pdf.meshCoordinates, self.laplaceApproximation.scalingForGaussian)
        self.lejaPoints,self.indicesOfLejaPoints, self.lejaSuccess = LP.getLejaSetFromPoints(self.identityScaling, mappedMesh, parameters.numLejas, self.poly, parameters.numPointsForLejaCandidates)
        if self.lejaSuccess ==False: # Failed to get Leja points
            self.lejaPoints = None
            self.idicesOfLejaPoints = None

    def computeUpdateWithInterpolatoryQuadrature(self, parameters, pdf, index, sde):
        self.divideOutGaussianAndSetIntegrand(pdf, sde, index)
        lejaPointsPdfVals = pdf.integrandAfterDividingOut
        V = opolynd.opolynd_eval(self.lejaPoints, self.poly.lambdas[:parameters.numLejas,:], self.poly.ab, self.poly)
        try:
            vinv = np.linalg.inv(V)
        except:
            return -1, 100000 # For catching failure
        value = np.matmul(vinv[0,:], lejaPointsPdfVals)
        condNumber = np.sum(np.abs(vinv[0,:]))
        return value, condNumber

    def computeUpdateWithAlternativeMethod(self, sde, parameters, pdf, index):

        scaling = GaussScale(sde.dimension)
        scaling.setMu(np.asarray(pdf.meshCoordinates[index,:]+parameters.h*sde.driftFunction(pdf.meshCoordinates[index,:])).T)
        cov = sde.diffusionFunction(scaling.mu.T)
        scaling.setCov((parameters.h*cov@cov.T))

        transformedAltMethodLejaPoints = map_from_canonical_space(self.altMethodLejaPoints, scaling)

        meshNearest, distances, indx = findNearestKPoints(scaling.mu, pdf.meshCoordinates,parameters.numQuadFit, getIndices = True)
        pdfNew = pdf.pdfVals[indx]

        transformedAltMethodLejaPointsPdfVals = np.exp(np.asarray(griddata(np.squeeze(meshNearest), np.log(pdfNew), np.squeeze(transformedAltMethodLejaPoints), method='linear', fill_value=np.log(pdf.minPdfValue))))

        if parameters.timeDiscretizationType == "EM":
            transitionMatrixRowAltMethodLejaPoints = np.expand_dims(self.timeDiscretiazationMethod.computeTransitionMatrixRow(transformedAltMethodLejaPoints[0],transformedAltMethodLejaPoints, parameters.h, sde ),1)

        else: #For Anderson-Mattingly
            indices = list(range(len(transformedAltMethodLejaPoints)))
            transitionMatrixRowAltMethodLejaPoints = self.timeDiscretiazationMethod.computeTransitionMatrixRow(transformedAltMethodLejaPoints[0],transformedAltMethodLejaPoints, parameters.h, sde, fullMesh =transformedAltMethodLejaPoints, newPointIndices_AM = indices)

        transitionMatrixRowAltMethodLejaPoints = np.squeeze(transitionMatrixRowAltMethodLejaPoints)

        if sde.dimension > 1:
            L = np.linalg.cholesky((scaling.cov))
            JacFactor = np.prod(np.diag(L))
        elif sde.dimension ==1:
            L = np.sqrt(scaling.cov)
            JacFactor = np.squeeze(L)

        weightToDivideOut = self.weightExp(scaling,transformedAltMethodLejaPoints)*1/(np.pi*JacFactor)

        testing = (transformedAltMethodLejaPointsPdfVals*transitionMatrixRowAltMethodLejaPoints)/weightToDivideOut

        u = map_to_canonical_space(transformedAltMethodLejaPoints, scaling)
        V = opolynd.opolynd_eval(u, self.poly.lambdas[:parameters.numLejas,:], self.poly.ab, self.poly)
        vinv = np.linalg.inv(V)
        value = np.matmul(vinv[0,:], testing)
        condNumber = np.sum(np.abs(vinv[0,:]))
        return value, condNumber


    def weightExp(self, scaling, mesh):
        '''Compute weight for alternative procedure'''
        if np.size(mesh,1) == 1:
            newvals = np.exp(-(mesh-scaling.mu)**2)*(1/scaling.cov)
            return np.squeeze(newvals)

        soln_vals = np.empty(len(mesh))
        invCov = np.linalg.inv(scaling.cov)
        for j in range(len(mesh)):
            x = np.expand_dims(mesh[j,:],1)
            Gs = np.exp(-(x-scaling.mu).T@invCov@(x-scaling.mu))
            soln_vals[j] = Gs
        return soln_vals

    def checkPdfValue(self, value, valueReplace):
        '''Adjust PDF value if less than or equal to 0.'''
        if value <= 0:
            value = valueReplace
        return value

    def manageLejaReuse(self, index, condNumber, simulation, parameters):
        '''Used for Leja reuse determination.'''
        if condNumber < parameters.conditionNumberForAcceptingLejaPointsAtNextTimeStep: # Finished, set Leja values for next time step
            simulation.LejaPointIndicesBoolVector[index] = True
            simulation.LejaPointIndicesMatrix[index,:] = self.indicesOfLejaPoints
        else:
            simulation.LejaPointIndicesBoolVector[index] = False


    def computeTimeStep(self, sde, parameters, simulation):
        LPReuseCount = 0
        AltMethodUseCount = 0
        newPdf = np.zeros(simulation.pdf.meshLength)
        pdf = simulation.pdf
        valueReplace = pdf.minPdfValue

        for index, point in enumerate(pdf.meshCoordinates):
            value = None
            condNumber = None
            quadraticFitSuccess = self.findQuadraticFit(sde, simulation, parameters, index)

            if quadraticFitSuccess:
                '''Reuse Leja points if available'''
                if simulation.LejaPointIndicesBoolVector[index]:
                    self.reuseLejaPoints(simulation, index, parameters,sde)
                    value, condNumber = self.computeUpdateWithInterpolatoryQuadrature(parameters,pdf, index, sde)
                    if condNumber < parameters.conditionNumForAltMethod: # Finished, housekeeping
                        LPReuseCount = LPReuseCount +1
                        self.manageLejaReuse(index, condNumber, simulation, parameters)
                        newPdf[index] =self.checkPdfValue(value, valueReplace)
                        continue

                '''Try with newly computed leja points'''
                self.computeLejaPoints(simulation, index, parameters, sde)
                if self.lejaSuccess:
                    value, condNumber = self.computeUpdateWithInterpolatoryQuadrature(parameters,pdf, index, sde)
                    if condNumber < parameters.conditionNumForAltMethod: # Finished, housekeeping
                        self.manageLejaReuse(index, condNumber, simulation, parameters)
                        newPdf[index] =self.checkPdfValue(value, valueReplace)
                        continue

            '''Nothing worked, use alterative method'''
            value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
            newPdf[index] =self.checkPdfValue(value, valueReplace)
            AltMethodUseCount = AltMethodUseCount+ 1

        return newPdf, LPReuseCount, AltMethodUseCount


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



