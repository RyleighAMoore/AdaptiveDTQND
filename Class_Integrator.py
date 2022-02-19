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
        print("Warning: You must have an equispaced mesh!")
        if parameters.useAdaptiveMesh == True:
            print("Updates with the Trapezoidal rule are not currently supported.")
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

        if not simulation.LejaPointIndicesBoolVector[index]: # Do not have Leja points
            orderedPoints, distances, indicesOfOrderedPoints = findNearestKPoints(pdf.meshCoordinates[index], pdf.meshCoordinates, parameters.numQuadFit, getIndices=True)
            quadraticFitMeshPoints = orderedPoints[:parameters.numQuadFit]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[indicesOfOrderedPoints]
            self.laplaceApproximation.computeleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints, sde.dimension)
        else: # Have Leja points to use
            quadraticFitMeshPoints = pdf.meshCoordinates[simulation.LejaPointIndicesMatrix[index,:].astype(int)]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[simulation.LejaPointIndicesMatrix[index,:].astype(int)]
            self.laplaceApproximation.computeleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints,sde.dimension)

        if np.min(pdf.integrandBeforeDividingOut)==0:
            pdf.integrandBeforeDividingOut[pdf.integrandBeforeDividingOut ==0] = 1e-16
        if np.any(self.laplaceApproximation.constantOfGaussian)==None:
            return False # Fit failed
        else:
            return True # Fit succeeded

    def divideOutGaussianAndSetIntegrand(self, pdf, sde, index):
        gaussianToDivideOut = self.laplaceApproximation.ComputeDividedOut(pdf, sde.dimension)

        if np.min(gaussianToDivideOut)<=0:
            gaussianToDivideOut[gaussianToDivideOut <=0] = min([x for x in gaussianToDivideOut if x !=0])
        try:
            integrand = pdf.integrandBeforeDividingOut/gaussianToDivideOut
            if np.isnan(integrand):
                integrand = np.nan_to_num(integrand, nan=np.nanmin(integrand))
        except:
            t=0
            # print(np.min(gaussianToDivideOut))
        pdf.setIntegrandAfterDividingOut(integrand)

    def reuseLejaPoints(self, simulation, index, parameters, sde):
        self.divideOutGaussianAndSetIntegrand(simulation.pdf, sde, index)
        assert simulation.LejaPointIndicesBoolVector[index] # Already have LejaPoints
        LejaIndices = simulation.LejaPointIndicesMatrix[index,:].astype(int)
        mesh2 = map_to_canonical_space(simulation.pdf.meshCoordinates[LejaIndices,:], self.laplaceApproximation.scalingForGaussian)
        self.lejaPoints = mesh2
        self.lejaPointsPdfVals = simulation.pdf.integrandAfterDividingOut[LejaIndices]
        self.indicesOfLejaPoints = LejaIndices

    def computeLejaPoints(self, simulation, index, parameters, sde):
        self.divideOutGaussianAndSetIntegrand(simulation.pdf, sde, index)
        mappedMesh = map_to_canonical_space(simulation.pdf.meshCoordinates, self.laplaceApproximation.scalingForGaussian)
        self.lejaPoints,self.indicesOfLejaPoints, self.lejaSuccess = LP.getLejaSetFromPoints(self.identityScaling, mappedMesh, parameters.numLejas, self.poly, parameters.numPointsForLejaCandidates)
        self.lejaPointsPdfVals = simulation.pdf.pdfVals[self.indicesOfLejaPoints]
        if self.lejaSuccess ==False: # Failed to get Leja points
            self.lejaPoints = None
            self.lejaPointsPdfVals = None
            self.idicesOfLejaPoints = None



    def computeUpdateWithInterpolatoryQuadrature(self, parameters, pdf, index, sde):
        self.lejaPointsPdfVals = pdf.integrandAfterDividingOut[self.indicesOfLejaPoints]
        V = opolynd.opolynd_eval(self.lejaPoints, self.poly.lambdas[:parameters.numLejas,:], self.poly.ab, self.poly)
        try:
            vinv = np.linalg.inv(V)
        except:
            # plt.figure()
            # plt.scatter(pdf.meshCoordinates[:,0], pdf.meshCoordinates[:,1])
            # plt.scatter(self.lejaPoints[:,0], self.lejaPoints[:,1])
            # plt.show()
            return -1, 100000
        value = np.matmul(vinv[0,:], self.lejaPointsPdfVals)
        condNumber = np.sum(np.abs(vinv[0,:]))
        return value, condNumber


    def computeUpdateWithAlternativeMethod(self, sde, parameters, pdf, index):
        return pdf.minPdfValue, 1

        '''Old Alternative method, works but does more than necessary'''
        # scaling = GaussScale(sde.dimension)
        # scaling.setMu(np.asarray(pdf.meshCoordinates[index,:]+parameters.h*sde.driftFunction(pdf.meshCoordinates[index,:])).T)
        # cov = sde.diffusionFunction(scaling.mu.T)
        # scaling.setCov((parameters.h*cov@cov.T))

        # mesh12 = map_from_canonical_space(self.altMethodLejaPoints, scaling)
        # meshNearest, distances, indx = findNearestKPoints(scaling.mu, pdf.meshCoordinates,parameters.numQuadFit, getIndices = True)
        # pdfNew = pdf.pdfVals[indx]

        # pdf12 = np.asarray(griddata(np.squeeze(meshNearest), np.log(pdfNew), np.squeeze(mesh12), method='linear', fill_value=np.log(np.min(pdf.pdfVals))))
        # pdf12 = np.exp(pdf12)

        # if parameters.timeDiscretizationType == "EM":
        #     transitionMatrixRow = np.expand_dims(self.timeDiscretiazationMethod.computeTransitionMatrixRow(mesh12[0],mesh12, parameters.h, sde ),1)
        # else:
        #     indices = list(range(len(mesh12)))
        #     transitionMatrixRow = self.timeDiscretiazationMethod.computeTransitionMatrixRow(mesh12[0],mesh12, parameters.h, sde, fullMesh =mesh12, newPointIndices_AM = indices)

        # transitionMatrixRow = np.squeeze(transitionMatrixRow)
        # if sde.dimension > 1:
        #     L = np.linalg.cholesky((scaling.cov))
        #     JacFactor = np.prod(np.diag(L))
        # elif sde.dimension ==1:
        #     L = np.sqrt(scaling.cov)
        #     JacFactor = np.squeeze(L)

        # g = self.weightExp(scaling,mesh12)*1/(np.pi*JacFactor)

        # testing = (pdf12*transitionMatrixRow)/g
        # u = map_to_canonical_space(mesh12, scaling)
        # V = opolynd.opolynd_eval(u, self.poly.lambdas[:parameters.numLejas,:], self.poly.ab, self.poly)
        # vinv = np.linalg.inv(V)
        # value = np.matmul(vinv[0,:], testing)
        # condNumber = np.sum(np.abs(vinv[0,:]))
        # return value, condNumber


    def weightExp(self, scaling, mesh):
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
        if value <= 0:
            value = valueReplace
        return value

    def manageLejaReuse(self, index, condNumber, simulation, parameters):
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
            AltMethodUseCount += 1


        # print(LPReuseCount/pdf.meshLength*100, "% Leja Reuse")
        # print(self.AltMethodUseCount/pdf.meshLength*100, "% Alt method Use")
        # assert self.AltMethodUseCount/pdf.meshLength*100 < 10, "WARNING: Alt method use is high*************"
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



