
import numpy as np
from scipy.stats import uniform, beta, norm, multivariate_normal
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions import GVals, Gaussian, G, weightExp, GAndersonMat
from scipy.interpolate import griddata, interp2d
from pyopoly1.LejaPoints import getLejaSetFromPoints, getLejaPoints
from pyopoly1.QuadratureRules import QuadratureByInterpolationND, QuadratureByInterpolation_Simple, QuadratureByInterpolationND_DivideOutGaussian
from pyopoly1.families import HermitePolynomials
from pyopoly1.Class_Gaussian import GaussScale
from pyopoly1 import indexing
import math
from pyopoly1 import variableTransformations as VT


def getMeshValsThatAreClose(Mesh, pdf, sigmaX, sigmaY, muX, muY, numStd = 4):
    MeshToKeep = []
    PdfToKeep = []
    for i in range(len(Mesh)):
        Px = Mesh[i,0]; Py = Mesh[i,1]
        if np.sqrt((Px-muX)**2 + (Py-muY)**2) < numStd*max(sigmaX,sigmaY):
            MeshToKeep.append([Px,Py])
            PdfToKeep.append(pdf[i])
    return np.asarray(MeshToKeep), np.asarray(PdfToKeep)


# '''Generate Leja Sample for use in alternative method if needed'''
# poly = HermitePolynomials(rho=0)
# d=2
# k = 40
# ab = poly.recurrence(k+1)
# lambdas = indexing.total_degree_indices(d, k)
# poly.lambdas = lambdas
# lejaPointsFinal, new = getLejaPoints(10, np.asarray([[0,0]]).T, poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

def Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly, h, NumLejas, step, GMat, LPMat, LPMatBool, numQuadFit, removeZerosValuesIfLessThanTolerance, conditionNumForAltMethod, drift, diff,numPointsForLejaCandidates, SpatialDiff,lejaPointsFinal, TimeStepType, minDistanceBetweenPoints, PrintStuff = True):
    dimension = np.size(mesh,1)
    numLejas = LPMat.shape[1]
    newPDF = []
    # condNums = []
    countUseMorePoints = 0 # Used to count if we have to revert to alternative procedure
    meshSize = len(mesh)
    LPUse = 0
    '''Try to Divide out Guassian using quadratic fit'''
    for ii in range(len(mesh)):
        # print('########################',ii/len(mesh)*100, '%')


        dr = h*drift(mesh[ii,:])[0]
        # muX = mesh[ii,0] + dr[0][0]
        # muY = mesh[ii,1] + dr[0][1]
        mu = mesh[ii,:]+dr

        scaling = GaussScale(dimension)
        scaling.setMu(np.asarray([mu]).T)


        GPDF = np.expand_dims(GMat[ii,:meshSize], 1)*pdf
        # plt.plot(mesh,GPDF, 'o')
        # plt.plot(mesh, (np.exp(-(-mesh)**2/0.02)/np.sqrt(0.02*np.pi))**2, '.')
        # GPDF2 = np.expand_dims(GVals2(muX, muY, mesh, h),1)*pdf
        # assert np.max(abs(GPDF2-GPDF)) < 10**(-7)

        value, condNum, scaleUsed, LPMat, LPMatBool, reuseLP = QuadratureByInterpolationND_DivideOutGaussian(scaling, h, poly, mesh, GPDF, LPMat, LPMatBool,ii,NumLejas, numQuadFit, diff, numPointsForLejaCandidates)
        # print(value)
        # print(condNum)
        if PrintStuff:
            LPUse = LPUse+reuseLP
        '''Alternative Method'''
        if math.isnan(condNum) or value < 0 or condNum > conditionNumForAltMethod:
            scaling.setCov((h*diff(np.asarray(mu))*diff(np.asarray(mu)).T).T)

            mesh12 = VT.map_from_canonical_space(lejaPointsFinal, scaling)
            meshLP, distances, indx = UM.findNearestKPoints(scaling.mu, mesh,numQuadFit, getIndices = True)
            pdfNew = pdf[indx]

            pdf12 = np.asarray(griddata(np.squeeze(meshLP), pdfNew, np.squeeze(mesh12), method='linear', fill_value=np.min(pdf)))
            pdf12[pdf12 < 0] = np.min(pdf)

            # plt.figure()
            # plt.plot(mesh,pdf, '*')
            # plt.plot(meshLP, pdfNew, 'o')
            # plt.plot(mesh12, pdf12, '.')

            if TimeStepType == "EM":
                v = np.expand_dims(G(0,mesh12, h, drift, diff, SpatialDiff),1)
            elif TimeStepType == "AM":
                v = np.expand_dims(GAndersonMat(mesh12, 0, h, drift, diff, dimension, minDistanceBetweenPoints, SpatialDiff),1)

            if dimension > 1:
                L = np.linalg.cholesky((scaling.cov))
                JacFactor = np.prod(np.diag(L))
            if dimension ==1:
                L = np.sqrt(scaling.cov)
                JacFactor = np.squeeze(L)

            g = weightExp(scaling,mesh12)*1/(np.pi*JacFactor)

            testing = np.squeeze((pdf12*v)/np.expand_dims(g,1))


            value, condNum = QuadratureByInterpolation_Simple(poly, scaling, mesh12, testing)
            # print(value)
            if PrintStuff:
                countUseMorePoints = countUseMorePoints+1
        # if value>10000:
        #     print(value)

        newPDF.append(value)
        # condNums.append(condNum)
    if PrintStuff:
        print('\n',(countUseMorePoints/len(mesh))*100, "% Used Alternative Method**************")
        print('\n',(LPUse/len(mesh))*100, "% Reused Leja Points")

    newPDFs = np.asarray(newPDF)
    # condNums = np.asarray([condNums]).T
    return newPDFs, mesh, LPMat, LPMatBool, LPUse, countUseMorePoints


