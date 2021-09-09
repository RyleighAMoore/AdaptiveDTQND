import numpy as np
from pyopoly1.Class_Gaussian import GaussScale
class PDF:
    def __init__(self, sde, parameters, UseNoise=False):
        self.pdfVals = None
        self.meshCoordinates = None
        self.meshLength = None
        self.UseNoise = UseNoise
        self.setInitialConditionMeshCoordinates(sde, parameters)



    def setInitialConditionMeshCoordinates(self, sde, parameters):
        self.meshCoordinates = nDGridMeshCenteredAtOrigin(sde.dimension, parameters.radius, parameters.kstepMin)
        self.meshLength = len(self.meshCoordinates)


    def initialCondition(self, sde, mesh, parameters):
        scale = GaussScale(sde.dimension)
        scale.setMu(parameters.h*sde.driftfun(np.zeros(sde.dimension)).T)
        scale.setCov((parameters.h*sde.difffun(np.zeros(sde.dimension))*sde.difffun(np.zeros(sde.dimension)).T).T)
        pdf = scale.ComputeGaussian(mesh)
        self.pdfVals.append(pdf)



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

