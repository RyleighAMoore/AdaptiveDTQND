import numpy as np
import matplotlib.pyplot as plt

from Class_Gaussian import GaussScale
from Functions import nDGridMeshCenteredAtOrigin

class PDF:
    def __init__(self, sde, parameters, UseNoise=False):
        '''
        Parameters:
        sde: stochastic differential equation to solve (class object)
        parameters: parameters for the simulation (class object)
        endTime: ending time of simulation
        '''
        self.pdfVals = None
        self.meshCoordinates = None
        self.meshLength = None
        self.UseNoise = UseNoise
        self.setInitialMeshCoordinates(sde, parameters)
        self.computeFirstTimeStepFromDiracInitialCondition(sde, parameters)


    def addPointsToMesh(self, newPoints):
        self.meshCoordinates = np.append(self.meshCoordinates, newPoints, axis=0)
        self.meshLength = len(self.meshCoordinates)

    def addPointsToPdf(self, newPoints):
        self.pdfVals = np.append(self.pdfVals, newPoints)

    def removePointsFromMesh(self, indexToRemove):
        self.meshCoordinates = np.delete(self.meshCoordinates, indexToRemove,0)
        self.meshLength = len(self.meshCoordinates)

    def removePointsFromPdf(self, index):
        self.pdfVals = np.delete(self.pdfVals, index)

    def setIntegrandBeforeDividingOut(self, integrandBeforeDividingOut):
        self.integrandBeforeDividingOut = integrandBeforeDividingOut

    def setIntegrandAfterDividingOut(self, integrandAfterDividingOut):
        self.integrandAfterDividingOut = integrandAfterDividingOut

    def setInitialMeshCoordinates(self, sde, parameters):
        if parameters.integratorType == "LQ":
            self.meshCoordinates = nDGridMeshCenteredAtOrigin(sde.dimension, parameters.radius, parameters.kstepMin)
        if parameters.integratorType == "TR" and parameters.OverideMesh is not None:
            self.meshCoordinates = parameters.OverideMesh
        if parameters.integratorType == "TR" and parameters.OverideMesh is None:
            self.meshCoordinates = nDGridMeshCenteredAtOrigin(sde.dimension, parameters.radius, parameters.kstepMin, parameters.initialMeshCentering)

        self.meshLength = len(self.meshCoordinates)

    def computeFirstTimeStepFromDiracInitialCondition(self, sde, parameters):
        '''We use a dirac initial condition centered at the origin, so the first time step
        can be computed exactly.
        '''
        scale = GaussScale(sde.dimension)
        scale.setMu(parameters.h*sde.driftFunction(np.zeros(sde.dimension)).T)
        scale.setCov((parameters.h*sde.diffusionFunction(np.zeros(sde.dimension))*sde.diffusionFunction(np.zeros(sde.dimension)).T).T)
        pdf = scale.ComputeGaussian(self.meshCoordinates, sde.dimension)
        self.pdfVals = pdf

    def plot(self):
        plt.figure()
        plt.scatter(self.meshCoordinates, self.pdfVals)
        plt.show()

