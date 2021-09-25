import numpy as np
import NDFunctionBank
from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_Simulation import Simulation
import matplotlib.pyplot as plt

class SDE:
    def __init__(self, dimension, driftFunction, diffusionFunction, spatialDiff):
        self.dimension = dimension
        self.driftFunction = driftFunction
        self.diffusionFunction = diffusionFunction
        self.spatialDiff = spatialDiff

    def ApproxExactSoln(self, endTime, radius,xStep):
        beta = 3 # Not really used
        kstepMin = xStep
        kstepMax = xStep # Not really used
        h = 0.005 # Do not make less

        parameters = Parameters(self, beta, radius, kstepMin, kstepMax, h, False, timeDiscretizationType = "EM")
        pdf = PDF(self, parameters)
        simulation= Simulation(self, parameters, endTime)
        G = simulation.integrator.TransitionMatrix
        numSteps = int(endTime/parameters.h)
        # plt.figure()
        for i in range(numSteps):
            pdf.pdfVals = kstepMin**self.dimension*G[:len(pdf.pdfVals), :len(pdf.pdfVals)]@pdf.pdfVals
            # plt.scatter(pdf.meshCoordinates, pdf.pdfVals)
        return pdf.meshCoordinates, pdf.pdfVals


        # surfaces2 = []
        # for i in solnIndices:
        #     surfaces2.append(surfaces[int(i)])
        # solns = []
        # for i in range(len(surfaces2)):
        #     gridSolnOnLejas = griddata(mesh, surfaces2[i], Meshes[i], method='cubic', fill_value=-1)
        #     solns.append(np.squeeze(gridSolnOnLejas))

        # LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, solns, h, plot=False)


