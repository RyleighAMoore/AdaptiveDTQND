# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:07:18 2021

@author: Rylei
"""
import numpy as np
from Class_TimeDiscretizationMethod import EulerMaruyamaTimeDiscretizationMethod, AndersonMattinglyTimeDiscretizationMethod
from Class_PDF import PDF
from Class_MeshUpdater import MeshUpdater
from Class_Integrator import IntegratorLejaQuadrature, IntegratorTrapezoidal
from tqdm import trange
import time

class Simulation():
    def __init__(self, sde, parameters, endTime):
        '''
        Manages computing the solution of the SDE.

        Parameters:
        sde: stochastic differential equation to solve (class object)
        parameters: parameters for the simulation (class object)
        endTime: ending time of simulation
        '''
        self.pdf = PDF(sde, parameters)
        self.endTime = endTime
        self.pdfTrajectory = []
        self.meshTrajectory = []
        self.times = []
        self.LPReuseCount = []
        self.AltMethodUseCount = []

    def removePoints(self, index):
        '''Remove point from Transition matrix (row and column)'''
        self.TransitionMatrix = np.delete(self.TransitionMatrix, index,0)
        self.TransitionMatrix = np.delete(self.TransitionMatrix, index,1)


    def checkIncreaseSizeStorageMatrices(self, parameters):
        '''Manages size of matrices for new points'''
        sizer = 2*self.pdf.meshLength
        if self.pdf.meshLength*2 >= np.size(self.TransitionMatrix,1):
            GMat2 = np.empty([2*sizer, 2*sizer])*np.NaN
            GMat2[:self.pdf.meshLength, :self.pdf.meshLength]= self.TransitionMatrix[:self.pdf.meshLength, :self.pdf.meshLength]
            self.TransitionMatrix = GMat2

            LPMat2 = np.empty([2*sizer, parameters.numLejas])*np.NaN
            LPMat2[:self.pdf.meshLength, :self.pdf.meshLength]= self.LejaPointIndicesMatrix[:self.pdf.meshLength, :parameters.numLejas]
            self.LejaPointIndicesMatrix = LPMat2

            LPMatBool2 = np.zeros((2*sizer,1), dtype=bool)
            LPMatBool2[:self.pdf.meshLength]= self.LejaPointIndicesBoolVector[:self.pdf.meshLength]
            self.LejaPointIndicesBoolVector = LPMatBool2


    def houseKeepingStorageMatrices(self, indices):
        '''Adjusts Leja point index tracker in the case points are removed'''
        largerLPMat = np.zeros(np.shape(self.LejaPointIndicesMatrix))
        for ii in indices:
            LPUpdateList = np.where(self.LejaPointIndicesMatrix == ii)[0]
            for i in LPUpdateList:
                self.LejaPointIndicesBoolVector[i] = False # Need to recopmute the Leja points
            largerLP = self.LejaPointIndicesMatrix >= ii
            largerLPMat = largerLPMat + largerLP
        self.LejaPointIndicesMatrix = self.LejaPointIndicesMatrix - largerLPMat
        '''Removes points'''
        self.LejaPointIndicesBoolVector = np.delete(self.LejaPointIndicesBoolVector, indices,0)
        self.LejaPointIndicesMatrix = np.delete(self.LejaPointIndicesMatrix, indices, 0)


    def setUpTransitionMatrix(self, sde, parameters):
        '''Initialize pieces needed for simulation'''
        self.setTimeDiscretizationDriver(parameters, sde)
        self.setIntegrator(sde, parameters)
        self.meshUpdater = MeshUpdater(parameters, self.pdf, sde.dimension)

        '''Set up matrices'''
        self.TransitionMatrix = self.timeDiscretizationMethod.computeTransitionMatrix(self.pdf, sde, parameters)
        self.LejaPointIndicesMatrix = np.zeros((self.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty, parameters.numLejas))
        self.LejaPointIndicesBoolVector = np.zeros((self.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty,1))

    def setTimeDiscretizationDriver(self, parameters, sde):
        if parameters.timeDiscretizationType == "EM":
            self.timeDiscretizationMethod = EulerMaruyamaTimeDiscretizationMethod(self.pdf, parameters)
        if parameters.timeDiscretizationType == "AM":
            self.timeDiscretizationMethod = AndersonMattinglyTimeDiscretizationMethod(self.pdf, parameters, sde.dimension)

    def setIntegrator(self, sde, parameters):
        if parameters.integratorType == "LQ":
            self.integrator = IntegratorLejaQuadrature(sde.dimension, parameters, self.timeDiscretizationMethod)
        if parameters.integratorType == "TR":
            self.integrator = IntegratorTrapezoidal(sde.dimension, parameters)

    def StepForwardInTime(self, sde, parameters):
        '''Iterates solution one time step and saves history if needed.'''
        self.pdf.pdfVals, LPReuseCount, AltMethodUseCount = self.integrator.computeTimeStep(sde, parameters, self)
        if parameters.saveHistory:
            self.LPReuseCount.append(np.copy(LPReuseCount))
            self.AltMethodUseCount.append(np.copy(AltMethodUseCount))

    def computeAllTimes(self, sde, parameters):
        # if parameters.integratorType == "LQ" and not parameters.saveHistory:
        #     self.pdfTrajectory.append(np.copy(self.pdf.pdfVals))
        #     self.meshTrajectory.append(np.copy(self.pdf.meshCoordinates))
        if parameters.saveHistory:
            self.pdfTrajectory.append(np.copy(self.pdf.pdfVals))
            self.meshTrajectory.append(np.copy(self.pdf.meshCoordinates))
        self.times.append(parameters.h)
        numSteps = int(self.endTime/parameters.h)
        timing = []
        timeStart = time.time()
        for i in trange(1, numSteps):
            if parameters.useAdaptiveMesh:
                if i>=parameters.eligibleToAddPointsTimeStep and i%parameters.addPointsEveryNSteps==0:
                    self.checkIncreaseSizeStorageMatrices(parameters)
                    self.meshUpdater.addPointsToMeshProcedure(self.pdf, parameters, self, sde)
                if i>=parameters.eligibleToRemovePointsTimeStep and i%parameters.removePointsEveryNSteps==0:
                    self.meshUpdater.removePointsFromMeshProcedure(self.pdf, self, parameters, sde)
                    # self.meshUpdater.removeOutlierPoints(self.pdf, self, parameters, sde)
            self.StepForwardInTime(sde, parameters)
            if i==numSteps-1 or parameters.saveHistory:
                self.pdfTrajectory.append(np.copy(self.pdf.pdfVals))
                self.meshTrajectory.append(np.copy(self.pdf.meshCoordinates))
            self.times.append((i+1)*parameters.h)
            timing.append(time.time()- timeStart)
        return timing







