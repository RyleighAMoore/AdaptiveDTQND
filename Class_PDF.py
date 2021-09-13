import numpy as np
from pyopoly1.Class_Gaussian import GaussScale
import matplotlib.pyplot as plt

class PDF:
    def __init__(self, sde, parameters, UseNoise=False):
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
        self.meshCoordinates = nDGridMeshCenteredAtOrigin(sde.dimension, parameters.radius, parameters.kstepMin)
        self.meshLength = len(self.meshCoordinates)


    def setInitialCondition(self, sde, parameters):
        scale = GaussScale(sde.dimension)
        scale.setMu(parameters.h*sde.driftFunction(np.zeros(sde.dimension)).T)
        scale.setCov((parameters.h*sde.diffusionFunction(np.zeros(sde.dimension))*sde.diffusionFunction(np.zeros(sde.dimension)).T).T)
        pdf = scale.ComputeGaussian(self, sde)
        self.pdfVals = pdf

    def plot(self):
        plt.figure()
        plt.scatter(self.meshCoordinates, self.pdfVals)
        plt.show()



def nDGridMeshCenteredAtOrigin(dimension, radius, stepSize, useNoiseBool = False):
        subdivision = radius/stepSize
        step = radius/subdivision
        grid= np.mgrid[tuple(slice(step - radius, radius, step) for _ in range(dimension))]
        mesh = []
        for i in range(grid.shape[0]):
            new = grid[i].ravel()
            if useNoiseBool:
                shake = 0.1*stepSize
                noise = np.random.uniform(-stepSize, stepSize ,size = (len(new)))
                noise = -stepSize*shake +(stepSize*shake - - stepSize*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
                new = new+noise
            mesh.append(new)
        grid = np.asarray(mesh).T
        distance = 0
        for i in range(dimension):
            distance += grid[:,i]**2
        distance = distance**(1/2)
        distance = distance < radius
        grid  =  grid[distance,:]
        return grid

