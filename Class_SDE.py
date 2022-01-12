import numpy as np
import NDFunctionBank
from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_Simulation import Simulation
import matplotlib.pyplot as plt
from tqdm import trange


class SDE:

    def __init__(self, dimension, driftFunction, diffusionFunction, spatialDiff):
        '''
        Manages the stochastic differential equation we are solving.
        dXt = f(Xt)dt + g(Xt)dWt
        Wt: Brownian motion
        X0: Initial condition, we assume a Dirac mass centered at the origin
        Parameters:
        dimension: dimension of the SDE, typically 1,2, 3, or 4 but code can do higher
        driftFunction: the vector-valued function defining the drift f(Xt)
        diffFunction: the square matrix-valued function denining the diffusion g(Xt)
        spatialDiff: boolean True if the diffusion is dependent on space, False if independent
        '''
        self.dimension = dimension
        self.driftFunction = driftFunction
        self.diffusionFunction = diffusionFunction
        self.spatialDiff = spatialDiff

    def ApproxExactSoln(self, endTime, radius, xStep):
        '''
        Uses the Trapezoidal rule to approximate the solution of the SDE for error analysis.

        Parameters:
        endTime: The time we want to solve the SDE at
        radius: Determines the points of the approximated solution
        xStep: The spacing of the approximiated solution
        '''
        beta = 3 # Not really used
        kstepMin = xStep
        kstepMax = xStep # Not really used
        h = 0.005 # Do not make less

        parameters = Parameters(self, beta, radius, kstepMin, kstepMax, h, False, timeDiscretizationType = "EM")
        pdf = PDF(self, parameters)
        simulation= Simulation(self, parameters, endTime)
        simulation.setUpTransitionMatrix(self, parameters)

        G = simulation.integrator.TransitionMatrix
        numSteps = int(endTime/parameters.h)
        # plt.figure()
        for i in trange(numSteps):
            pdf.pdfVals = kstepMin**self.dimension*G[:len(pdf.pdfVals), :len(pdf.pdfVals)]@pdf.pdfVals
            # plt.scatter(pdf.meshCoordinates, pdf.pdfVals)
        return pdf.meshCoordinates, pdf.pdfVals

    def exactSolution(self, mesh, endTime):
        '''
        Computes the exact solution for an SDE with constant drift and diffusion.

        Parameters:
        mesh: Mesh values at which to compute the exact solution
        endTime: the time which to approximate the solution up to
        '''
        print("WARNING: The exact solution is only accurate if the drift and diffusion are constant.")
        drift = self.driftFunction(np.asarray([mesh[0]]))
        drift = drift[0][0]
        diff = self.diffusionFunction(mesh[0])[0,0]
        D = diff**2*0.5
        r = (mesh[:,0]-drift*endTime)**2
        for ii in range(1,self.dimension):
            r += (mesh[:,ii])**2
        vals = np.exp(-r/(4*D*endTime))*(1/(4*np.pi*D*endTime))**(self.dimension/2)
        return vals

