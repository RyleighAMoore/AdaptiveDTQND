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
        self.setTimeDiscretizationDriver(parameters, sde)
        self.setIntegrator(sde, parameters)
        # self.setUpTransitionMatrix(self.pdf, sde, parameters)
        self.meshUpdater = MeshUpdater(parameters, self.pdf, sde.dimension)

    def removePoints(self, index):
        self.TransitionMatrix = np.delete(self.TransitionMatrix, index,0)
        self.TransitionMatrix = np.delete(self.TransitionMatrix, index,1)
        # self.LejaPointIndicesMatrix = np.delete(self.LejaPointIndicesMatrix, index,0)
        # self.LejaPointIndicesBoolVector = np.delete(self.LejaPointIndicesBoolVector, index)


    def checkIncreaseSizeStorageMatrices(self, parameters):
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
        largerLPMat = np.zeros(np.shape(self.LejaPointIndicesMatrix))
        for ii in indices:
            LPUpdateList = np.where(self.LejaPointIndicesMatrix == ii)[0]
            for i in LPUpdateList:
                self.LejaPointIndicesBoolVector[i] = False
            largerLP = self.LejaPointIndicesMatrix >= ii
            largerLPMat = largerLPMat + largerLP
        self.LejaPointIndicesMatrix = self.LejaPointIndicesMatrix - largerLPMat
        self.LejaPointIndicesBoolVector = np.delete(self.LejaPointIndicesBoolVector, indices,0)
        self.LejaPointIndicesMatrix = np.delete(self.LejaPointIndicesMatrix, indices, 0)


    def setUpTransitionMatrix(self, sde, parameters):
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
            self.integrator = IntegratorTrapezoidal(sde.dimension, parameters, self.timeDiscretizationMethod)


    def StepForwardInTime(self, sde, parameters):
        self.pdf.pdfVals = self.integrator.computeTimeStep(sde, parameters, self)
        self.pdfTrajectory.append(np.copy(self.pdf.pdfVals))
        self.meshTrajectory.append(np.copy(self.pdf.meshCoordinates))

    def computeAllTimes(self, sde, parameters):
        self.pdfTrajectory.append(np.copy(self.pdf.pdfVals))
        self.meshTrajectory.append(np.copy(self.pdf.meshCoordinates))
        self.times.append(parameters.h)
        numSteps = int(self.endTime/parameters.h)
        timing = []
        timeStart = time.time()
        for i in trange(1, numSteps):
            if i>2 and parameters.useAdaptiveMesh ==True:
                self.checkIncreaseSizeStorageMatrices(parameters)
                self.meshUpdater.addPointsToMeshProcedure(self.pdf, parameters, self, sde)
                # print(len(self.pdfTrajectory[-1]), "****************")
                if i>=9 and i%25==1:
                    self.meshUpdater.removePointsFromMeshProcedure(self.pdf, self, parameters, sde)
            # self.meshUpdater.removeOutlierPoints(self.pdf, self, parameters, sde)
            self.StepForwardInTime(sde, parameters)
            self.times.append((i+1)*parameters.h)
            timing.append(time.time()- timeStart)
        return timing







