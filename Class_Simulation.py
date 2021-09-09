# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:07:18 2021

@author: Rylei
"""
from Class_TimeDiscretizationMethod import EulerMaruyamaTimeDiscretizationMethod, AndersonMattinglyTimeDiscretizationMethod

class Simulation():
    def __init__(self, sde, parameters, pdf):
        self.setTimeDiscretizationDriver(parameters)
        self.timeDiscretizationMethod = AndersonMattinglyTimeDiscretizationMethod()
        self.TransitionMatrix = self.timeDiscretizationDriver.computeTransitionMatrix(pdf, sde, parameters.h)
        self.computeTimestep()

    def setTimeDiscretizationDriver(self, parameters):
        if parameters.timeDiscretizationType == "EM":
            self.timeDiscretizationDriver = EulerMaruyamaTimeDiscretizationMethod()
        if parameters.timeDiscretizationType == "AM":
            self.timeDiscretizationDriver = AndersonMattinglyTimeDiscretizationMethod()

    def computeTimestep(self):
        print(self.TransitionMatrix)
        # doMeshUpdates()
        # parameters.integrationMethod.computeIntegral()
        # return valuesAtTimestep




