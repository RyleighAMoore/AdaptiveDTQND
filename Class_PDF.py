import numpy as np
from Class_Gaussian import GaussScale
import matplotlib.pyplot as plt
from Functions import nDGridMeshCenteredAtOrigin, nDGridMeshSquareCenteredAroundGivenPoint
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
        self.setInitialConditionMeshCoordinates(sde, parameters)
        self.setInitialCondition(sde, parameters)


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

    def setInitialConditionMeshCoordinates(self, sde, parameters):
        if parameters.integratorType == "LQ":
            self.meshCoordinates = nDGridMeshCenteredAtOrigin(sde.dimension, parameters.radius, parameters.kstepMin)
        if parameters.integratorType == "TR":
            if parameters.OverideMesh is not None:
                self.meshCoordinates = parameters.OverideMesh
            else:
                self.meshCoordinates = nDGridMeshSquareCenteredAroundGivenPoint(sde.dimension, parameters.radius, parameters.kstepMin, parameters.initialMeshCentering)

        # self.meshCoordinates = self.meshCoordinates + parameters.initialMeshCentering*np.ones(np.shape(self.meshCoordinates))
        self.meshLength = len(self.meshCoordinates)


    def setInitialCondition(self, sde, parameters):
        scale = GaussScale(sde.dimension)
        scale.setMu(parameters.h*sde.driftFunction(np.zeros(sde.dimension)).T)
        scale.setCov((parameters.h*sde.diffusionFunction(np.zeros(sde.dimension))*sde.diffusionFunction(np.zeros(sde.dimension)).T).T)
        pdf = scale.ComputeGaussian(self.meshCoordinates, sde.dimension)
        self.pdfVals = pdf

    def plot(self):
        plt.figure()
        plt.scatter(self.meshCoordinates, self.pdfVals)
        plt.show()

