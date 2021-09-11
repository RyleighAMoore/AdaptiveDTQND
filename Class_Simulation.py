# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:07:18 2021

@author: Rylei
"""
from Class_TimeDiscretizationMethod import EulerMaruyamaTimeDiscretizationMethod, AndersonMattinglyTimeDiscretizationMethod
from Class_PDF import PDF
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Simulation():
    def __init__(self, sde, parameters, endTime):
        self.pdf = PDF(sde, parameters)
        self.endTime = endTime
        self.pdfTrajectory = []
        self.meshTrajectory = []
        self.setTimeDiscretizationDriver(parameters)
        self.integrator = Integrator(self, sde, parameters, self.pdf)
        self.computeAllTimes(sde, self.pdf, parameters)


    def setTimeDiscretizationDriver(self, parameters):
        if parameters.timeDiscretizationType == "EM":
            self.timeDiscretizationMethod = EulerMaruyamaTimeDiscretizationMethod()
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
            self.computeTimestep(sde, pdf, parameters)





from pyopoly1.Class_Gaussian import GaussScale
import numpy as np
from QuadraticFit import LaplaceApproximation
import math
from pyopoly1.variableTransformations import map_to_canonical_space
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
from pyopoly1 import LejaPoints as LP
from pyopoly1 import opolynd

class Integrator:
    def __init__(self, simulation, sde, parameters, pdf):
        self.lejaPoints = None
        self.TransitionMatrix = simulation.timeDiscretizationMethod.computeTransitionMatrix(pdf, sde, parameters.h)
        self.LejaPointIndicesMatrix = np.zeros((simulation.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty, parameters.numLejas))
        self.LejaPointIndicesBoolVector = np.zeros(simulation.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty)
        self.conditionNumberForAcceptingLejaPointsAtNextTimeStep = 1.1
        self.setIdentityScaling(sde)
        self.setUpPolnomialFamily(sde)
        self.laplaceApproximation = LaplaceApproximation(sde)

    def setUpPolnomialFamily(self, sde):
        self.poly = HermitePolynomials(rho=0)
        d=sde.dimension
        k = 40
        lambdas = indexing.total_degree_indices(d, k)
        self.poly.lambdas = lambdas

    def setIdentityScaling(self, sde):
         self.identityScaling = GaussScale(sde.dimension)
         self.identityScaling.setMu(np.zeros(sde.dimension).T)
         self.identityScaling.setCov(np.eye(sde.dimension))

    def findQuadraticFit(self, sde, pdf, parameters, index):
        ## TODO: Update this so I don't recompute drift and diff everytime
        pdf.setIntegrandBeforeDividingOut(self.TransitionMatrix[index,:pdf.meshLength]*pdf.pdfVals)
        if not self.LejaPointIndicesBoolVector[index]:
            scaling = GaussScale(sde.dimension)
            scaling.setMu(pdf.meshCoordinates[index,:]+parameters.h*sde.driftFunction(pdf.meshCoordinates[index,:]))
            orderedPoints, distances, indicesOfOrderedPoints = findNearestKPoints(pdf.meshCoordinates[index], pdf.meshCoordinates, parameters.numQuadFit, getIndices=True)
            quadraticFitMeshPoints = orderedPoints[:parameters.numQuadFit]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[indicesOfOrderedPoints]
            self.laplaceApproximation.copmuteleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints, pdf, sde, parameters)
        else:
            quadraticFitMeshPoints = pdf.meshCoordinates[self.LejaPointIndicesMatrix[index,:].astype(int)]
            pdfValuesOfQuadraticFitPoints = pdf.integrandBeforeDividingOut[self.LejaPointIndicesMatrix[index,:].astype(int)]
            self.laplaceApproximation.copmuteleastSquares(quadraticFitMeshPoints, pdfValuesOfQuadraticFitPoints, pdf, sde, parameters)

        if np.any(self.laplaceApproximation.constantOfGaussian)==None: # Fit failed
            return False
        else:
            return True

    def divideOutGaussianAndSetIntegrand(self, pdf, sde, index):
        gaussianToDivideOut = self.laplaceApproximation.ComputeDividedOut(pdf, sde)
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

        # transformedMesh = map_to_canonical_space(self.lejaPoints, self.laplaceApproximation.scalingForGaussian)
        V = opolynd.opolynd_eval(self.lejaPoints, self.poly.lambdas[:parameters.numLejas,:], self.poly.ab, self.poly)
        vinv = np.linalg.inv(V)
        value = np.matmul(vinv[0,:], self.lejaPointsPdfVals)
        condNumber = np.sum(np.abs(vinv[0,:]))
        return value, condNumber


    def computeUpdateWithAlternativeMethod(self):
        pass

    def computeTimeStep(self, sde, parameters, pdf):
        newPdf = []
        for index, point in enumerate(pdf.meshCoordinates):
            useLejaIntegrationProcedure = self.findQuadraticFit(sde, pdf, parameters, index)
            if not useLejaIntegrationProcedure: # Failed Quadratic Fit
                self.computeUpdateWithAlternativeMethod()
            else: # Get Leja points
                self.setLejaPoints(pdf, index, parameters,sde)
                if any(self.lejaPoints) == None: #Getting Leja points failed
                     self.computeUpdateWithAlternativeMethod()
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
                        if condNumber > parameters.conditionNumForAltMethod: # Nothing worked, use alt method
                            self.computeUpdateWithAlternativeMethod()

            newPdf.append(np.copy(value))
        return np.asarray(newPdf)



def findNearestKPoints(Coord, AllPoints, numNeighbors, getIndices = False):
    # xCoord = Coord[0]
    # yCoord= Coord[1]
    # normList1 = (xCoord*np.ones(len(AllPoints)) - AllPoints[:,0])**2 + (yCoord*np.ones(len(AllPoints)) - AllPoints[:,1])**2

    normList = np.zeros(np.size(AllPoints,0))
    size = np.size(AllPoints,0)
    for i in range(np.size(AllPoints,1)):
        normList += (Coord[i]*np.ones(size) - AllPoints[:,i])**2

    idx = np.argsort(normList)


    if getIndices:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]], idx[:numNeighbors]
    else:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]]





