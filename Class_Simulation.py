# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:07:18 2021

@author: Rylei
"""
from Class_TimeDiscretizationMethod import EulerMaruyamaTimeDiscretizationMethod, AndersonMattinglyTimeDiscretizationMethod
from Class_PDF import PDF
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Class_MeshUpdater import MeshUpdater

class Simulation():
    def __init__(self, sde, parameters, endTime):
        self.timeDiscretizationMethod = None
        self.pdf = PDF(sde, parameters)
        self.endTime = endTime
        self.pdfTrajectory = []
        self.meshTrajectory = []
        self.setTimeDiscretizationDriver(parameters, self.pdf)
        self.meshUpdater = MeshUpdater(parameters, self.pdf, sde.dimension)
        self.integrator = Integrator(self, sde, parameters, self.pdf)
        self.computeAllTimes(sde, self.pdf, parameters)


    def setTimeDiscretizationDriver(self, parameters, pdf):
        if parameters.timeDiscretizationType == "EM":
            self.timeDiscretizationMethod = EulerMaruyamaTimeDiscretizationMethod(pdf)
        if parameters.timeDiscretizationType == "AM":
            self.timeDiscretizationMethod = AndersonMattinglyTimeDiscretizationMethod()


    def computeTimestep(self, sde, pdf, parameters):
        pdf.pdfVals = self.integrator.computeTimeStep(sde, parameters, pdf)
        self.pdfTrajectory.append(np.copy(pdf.pdfVals))
        self.meshTrajectory.append(np.copy(pdf.meshCoordinates))

    def computeAllTimes(self, sde, pdf, parameters):
        self.pdfTrajectory.append(np.copy(pdf.pdfVals))
        self.meshTrajectory.append(np.copy(pdf.meshCoordinates))
        numSteps = int(self.endTime/parameters.h)
        for i in range(numSteps):
            if i>2:
                self.meshUpdater.addPointsToMeshProcedure(pdf, parameters, self, sde)
                # self.meshUpdater.removePointsFromMeshProcedure(pdf, self, parameters, sde)
            self.computeTimestep(sde, pdf, parameters)





from pyopoly1.Class_Gaussian import GaussScale
import numpy as np
from QuadraticFit import LaplaceApproximation
import math
from pyopoly1.variableTransformations import map_to_canonical_space, map_from_canonical_space
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
from pyopoly1 import LejaPoints as LP
from pyopoly1 import opolynd
from pyopoly1.LejaPoints import getLejaPoints
from scipy.interpolate import griddata
from Functions import weightExp, G

class Integrator:
    def __init__(self, simulation, sde, parameters, pdf):
        self.lejaPoints = None
        self.TransitionMatrix = simulation.timeDiscretizationMethod.computeTransitionMatrix(pdf, sde, parameters.h)
        self.LejaPointIndicesMatrix = np.zeros((simulation.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty, parameters.numLejas))
        self.LejaPointIndicesBoolVector = np.zeros(simulation.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty)
        self.conditionNumberForAcceptingLejaPointsAtNextTimeStep = 1.1
        self.setIdentityScaling(sde.dimension)
        self.setUpPolnomialFamily(sde.dimension)
        self.altMethodLejaPoints, temp = getLejaPoints(10, np.zeros((sde.dimension,1)), self.poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
        self.laplaceApproximation = LaplaceApproximation(sde)


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

    def findQuadraticFit(self, sde, pdf, parameters, index):
        ## TODO: Update this so I don't recompute drift and diff everytime
        pdf.setIntegrandBeforeDividingOut(self.TransitionMatrix[index,:pdf.meshLength]*pdf.pdfVals)
        if not self.LejaPointIndicesBoolVector[index]: # Do not have good Leja points
            orderedPoints, distances, indicesOfOrderedPoints = findNearestKPoints(pdf.meshCoordinates[index], pdf.meshCoordinates, parameters.numQuadFit, getIndices=True)
            quadraticFitMeshPoints = orderedPoints[:parameters.numQuadFit]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[indicesOfOrderedPoints]
            self.laplaceApproximation.copmuteleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints, sde.dimension)
        else:
            quadraticFitMeshPoints = pdf.meshCoordinates[self.LejaPointIndicesMatrix[index,:].astype(int)]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[self.LejaPointIndicesMatrix[index,:].astype(int)]
            self.laplaceApproximation.copmuteleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints,sde.dimension)

        if np.any(self.laplaceApproximation.constantOfGaussian)==None: # Fit failed
            return False
        else:
            return True

    def divideOutGaussianAndSetIntegrand(self, pdf, sde, index):
        gaussianToDivideOut = self.laplaceApproximation.ComputeDividedOut(pdf, sde.dimension)
        pdf.setIntegrandAfterDividingOut(pdf.integrandBeforeDividingOut/gaussianToDivideOut)


    def setLejaPoints(self, pdf, index, parameters, sde):
        self.divideOutGaussianAndSetIntegrand(pdf, sde, index)
        if self.LejaPointIndicesBoolVector[index]: # Already have LejaPoints
            LejaIndices = self.LejaPointIndicesMatrix[index,:].astype(int)
            self.lejaPoints = pdf.meshCoordinates[LejaIndices,:]
            self.lejaPointsPdfVals = pdf.integrandAfterDividingOut[LejaIndices]
            self.indicesOfLejaPoints = LejaIndices
        else: # Need Leja points.
            mappedMesh = map_to_canonical_space(pdf.meshCoordinates, self.laplaceApproximation.scalingForGaussian)
            self.lejaPoints, self.lejaPointsPdfVals,self.indicesOfLejaPoints = LP.getLejaSetFromPoints(self.identityScaling, mappedMesh, parameters.numLejas, self.poly, pdf.integrandAfterDividingOut, sde.diffusionFunction, parameters.numPointsForLejaCandidates)
            if math.isnan(self.lejaPoints[0]): # Failed to get Leja points
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
        scaling = GaussScale(sde.dimension)
        scaling.setMu(pdf.meshCoordinates[index,:]+parameters.h*sde.driftFunction(pdf.meshCoordinates[index,:]))
        scaling.setCov((parameters.h*sde.diffusionFunction(scaling.mu*sde.diffusionFunction(scaling.mu).T).T))
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


    def computeTimeStep(self, sde, parameters, pdf):
        newPdf = []
        for index, point in enumerate(pdf.meshCoordinates):
            useLejaIntegrationProcedure = self.findQuadraticFit(sde, pdf, parameters, index)
            if not useLejaIntegrationProcedure: # Failed Quadratic Fit
                value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
            else: # Get Leja points
                self.setLejaPoints(pdf, index, parameters,sde)
                if any(self.lejaPoints) == None: #Getting Leja points failed
                     value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
                     print("failed Leja")
                else: # Continue with integration, try to use Leja points from last step
                    value, condNumber = self.computeUpdateWithInterpolatoryQuadrature(parameters,pdf, index, sde)
                    if condNumber < 1.1: # Leja points worked really well, likely okay for next time step
                        self.LejaPointIndicesBoolVector[index] = True
                        self.LejaPointIndicesMatrix[index,:] = self.indicesOfLejaPoints
                    else: # Continue with integration, use new leja points
                        self.LejaPointIndicesBoolVector[index] = False
                        self.setLejaPoints(pdf, index, parameters,sde)
                        value, condNumber = self.computeUpdateWithInterpolatoryQuadrature(parameters,pdf, index, sde)
                        if condNumber < 1.1: # Leja points worked really well, likely okay for next time step
                            self.LejaPointIndicesBoolVector[index] = True
                            self.LejaPointIndicesMatrix[index,:] = self.indicesOfLejaPoints
                        if condNumber > parameters.conditionNumForAltMethod or value < 0: # Nothing worked, use alt method
                            value,condNumber = self.computeUpdateWithAlternativeMethod(sde, parameters, pdf, index)
            newPdf.append(np.copy(value))
        return np.asarray(newPdf)



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





