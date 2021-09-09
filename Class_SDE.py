import numpy as np
import NDFunctionBank
class SDE:
    def __init__(self, dimension, driftFunction, diffusionFunction, spatialDiff):
        self.dimension = dimension
        self.driftFunction = driftFunction
        self.diffusionFunction = diffusionFunction
        self.spatialDiff = spatialDiff

    def createExampleSimpleSDE(dimension, driftConstant, diffusionConstant):
        # define the functions
        return SDE(dimension, driftFunction, diffusionFunction)

