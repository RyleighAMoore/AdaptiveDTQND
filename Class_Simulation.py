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
        self.setTimeDiscretizationDriver(parameters, self.pdf)
        self.meshUpdater = MeshUpdater(parameters, self.pdf, sde.dimension)
        self.setIntegrator(sde, parameters, self.pdf)


    def setTimeDiscretizationDriver(self, parameters, pdf):
        if parameters.timeDiscretizationType == "EM":
            self.timeDiscretizationMethod = EulerMaruyamaTimeDiscretizationMethod(pdf, adaptive = parameters.useAdaptiveMesh)
        if parameters.timeDiscretizationType == "AM":
            self.timeDiscretizationMethod = AndersonMattinglyTimeDiscretizationMethod(pdf, adaptive = parameters.useAdaptiveMesh)

    def setIntegrator(self, sde, parameters, pdf):
        if parameters.integratorType == "LQ":
            self.integrator = IntegratorLejaQuadrature(self, sde, parameters, self.pdf)
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
                self.integrator.checkIncreaseSizeStorageMatrices(pdf,parameters)
                self.meshUpdater.addPointsToMeshProcedure(pdf, parameters, self, sde)
                # print(len(self.pdfTrajectory[-1]), "****************")
                if i>=9 and i%25==1:
                    self.meshUpdater.removePointsFromMeshProcedure(pdf, self, parameters, sde)
            self.computeTimestep(sde, pdf, parameters)
            self.times.append((i+1)*parameters.h)







