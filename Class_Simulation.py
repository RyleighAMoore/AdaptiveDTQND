# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:07:18 2021

@author: Rylei
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Class_TimeDiscretizationMethod import EulerMaruyamaTimeDiscretizationMethod, AndersonMattinglyTimeDiscretizationMethod
from Class_PDF import PDF
from Class_MeshUpdater import MeshUpdater
from Class_Integrator import IntegratorLejaQuadrature, IntegratorTrapezoidal


class Simulation():
    def __init__(self, sde, parameters, endTime):
        self.timeDiscretizationMethod = None
        self.pdf = PDF(sde, parameters)
        self.endTime = endTime
        self.pdfTrajectory = []
        self.meshTrajectory = []
        self.times = []
        self.setIntegrator(sde, parameters, self.pdf)
        self.setTimeDiscretizationDriver(parameters, self.pdf, sde)
        self.setUpTransitionMatrix(self.pdf, sde, parameters)
        self.meshUpdater = MeshUpdater(parameters, self.pdf, sde.dimension)

    def removePoints(self, index):
        self.TransitionMatrix = np.delete(self.TransitionMatrix, index,0)
        self.TransitionMatrix = np.delete(self.TransitionMatrix, index,1)
        # self.LejaPointIndicesMatrix = np.delete(self.LejaPointIndicesMatrix, index,0)
        # self.LejaPointIndicesBoolVector = np.delete(self.LejaPointIndicesBoolVector, index)


    def checkIncreaseSizeStorageMatrices(self, parameters):
        pdf = self.pdf
        sizer = 2*pdf.meshLength
        if pdf.meshLength*2 >= np.size(self.TransitionMatrix,1):
            GMat2 = np.empty([2*sizer, 2*sizer])*np.NaN
            GMat2[:pdf.meshLength, :pdf.meshLength]= self.TransitionMatrix[:pdf.meshLength, :pdf.meshLength]
            self.TransitionMatrix = GMat2

            LPMat2 = np.empty([2*sizer, parameters.numLejas])*np.NaN
            LPMat2[:pdf.meshLength, :pdf.meshLength]= self.LejaPointIndicesMatrix[:pdf.meshLength, :parameters.numLejas]
            self.LejaPointIndicesMatrix = LPMat2

            LPMatBool2 = np.zeros((2*sizer,1), dtype=bool)
            LPMatBool2[:pdf.meshLength]= self.LejaPointIndicesBoolVector[:pdf.meshLength]
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


    def setUpTransitionMatrix(self, pdf, sde, parameters):
        self.TransitionMatrix = self.timeDiscretizationMethod.computeTransitionMatrix(pdf, sde, parameters)
        self.LejaPointIndicesMatrix = np.zeros((self.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty, parameters.numLejas))
        self.LejaPointIndicesBoolVector = np.zeros((self.timeDiscretizationMethod.sizeTransitionMatrixIncludingEmpty,1))


    def setTimeDiscretizationDriver(self, parameters, pdf, sde):
        if parameters.timeDiscretizationType == "EM":
            self.timeDiscretizationMethod = EulerMaruyamaTimeDiscretizationMethod(pdf, parameters, self.integrator.altMethodLejaPoints)
        if parameters.timeDiscretizationType == "AM":
            self.timeDiscretizationMethod = AndersonMattinglyTimeDiscretizationMethod(pdf, parameters, self.integrator.altMethodLejaPoints)

    def setIntegrator(self, sde, parameters, pdf):
        if parameters.integratorType == "LQ":
            self.integrator = IntegratorLejaQuadrature(sde.dimension, parameters)
        if parameters.integratorType == "TR":
            self.integrator = IntegratorTrapezoidal(self, sde, parameters, self.pdf)


    def computeTimestep(self, sde, pdf, parameters):
        pdf.pdfVals = self.integrator.computeTimeStep(sde, parameters, self)
        self.pdfTrajectory.append(np.copy(pdf.pdfVals))
        self.meshTrajectory.append(np.copy(pdf.meshCoordinates))

    def computeAllTimes(self, sde, pdf, parameters):
        self.pdfTrajectory.append(np.copy(pdf.pdfVals))
        self.meshTrajectory.append(np.copy(pdf.meshCoordinates))
        self.times.append(parameters.h)
        numSteps = int(self.endTime/parameters.h)
        for i in range(1, numSteps):
            if i>2 and parameters.useAdaptiveMesh ==True:
                self.checkIncreaseSizeStorageMatrices(parameters)
                self.meshUpdater.addPointsToMeshProcedure(pdf, parameters, self, sde)
                # print(len(self.pdfTrajectory[-1]), "****************")
                if i>=9 and i%25==1:
                    self.meshUpdater.removePointsFromMeshProcedure(pdf, self, parameters, sde)
            self.computeTimestep(sde, pdf, parameters)
            self.times.append((i+1)*parameters.h)







