# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:21:05 2020

@author: Ryleigh
"""
import matplotlib.pyplot as plt
import numpy as np
from pyopoly1 import opolynd
from pyopoly1.LejaUtilities import get_lu_leja_samples, sqrtNormal_weights
from pyopoly1.opolynd import opolynd_eval
from mpl_toolkits.mplot3d import Axes3D
import math
from pyopoly1 import variableTransformations as VT
np.random.seed(10)


'''
num_leja_samples: Total number of samples to be returned (including initial samples).
initial_samples: The samples that we must include in the leja sequence.
poly: Polynomial chaos expansion, fully implemented with options, indices, etc.
num_candidate_samples: Number of samples that are chosen from to complete the leja sequence after initial samples are used.
candidateSampleMesh: If num_candidate_samples is zero, this variable defines the candidate samples to use
returnIndices: Returns the indices of the leja sequence if True.
'''
def getLejaPoints(num_leja_samples, initial_samples, poly, num_candidate_samples = 10000, candidateSampleMesh = [], returnIndices = False):
    num_vars = np.size(initial_samples,0)
    # generate_candidate_samples = lambda n: np.sqrt(2*np.sqrt(2*num_leja_samples))*np.random.normal(0, 1, (num_vars, n))
    generate_candidate_samples = lambda n: 7*np.random.normal(0, 1, (num_vars, n))
    # generate_candidate_samples = lambda n: np.sqrt(2)*num_leja_samples*np.random.normal(0, 1, (num_vars, n))


    if num_candidate_samples == 0:
        candidate_samples = candidateSampleMesh
    else:
        candidate_samples = generate_candidate_samples(num_candidate_samples)

    num_initial_samples = len(initial_samples.T)

    # precond_func = lambda matrix, samples: christoffel_weights(matrix)
    precond_func = lambda matrix, samples: sqrtNormal_weights(samples)

    samples, data_structures, successBool = get_lu_leja_samples(poly,
        opolynd_eval,candidate_samples,num_leja_samples,
        preconditioning_function=precond_func,
        initial_samples=initial_samples)


    if returnIndices:
        if successBool == False:
            # print("LEJA FAIL - LEJA FAIL - LEJA FAIL")
            return [float('nan')], [float('nan')]
        assert successBool == True, "Need to implement returning indices when successBool is False."

    if successBool ==True:
        if returnIndices:
            indicesLeja = data_structures[2]
            return np.asarray(samples).T, indicesLeja
        return np.asarray(samples).T, np.asarray(samples[:,num_initial_samples:]).T

import Functions as fun
def getLejaSetFromPoints(scale, Mesh, numLejaPointsToReturn, poly, Pdf, diff, numPointsForLejaCandidates):
    # candidatesFull = VT.map_to_canonical_space(Mesh,scale)
    candidatesFull = Mesh # don't need to transform since the scale is normal when this function is used.
    indices = [np.nan]
    candidates, distances, indik = fun.findNearestKPoints(scale.mu, candidatesFull,numPointsForLejaCandidates, getIndices = True)
    # Px = candidates[0,0]
    # Py = candidates[0,1]
    point = candidates[0]
    pointPDF = Pdf[0]
    candidates = candidates[1:]
    # lejaPointsFinal, indices = getLejaPoints(numLejaPointsToReturn, np.asarray([[Px,Py]]).T, poly, num_candidate_samples = 0, candidateSampleMesh = candidates.T, returnIndices=True)

    lejaPointsFinal, indices = getLejaPoints(numLejaPointsToReturn, np.asarray([point]).T, poly, num_candidate_samples = 0, candidateSampleMesh = candidates.T, returnIndices=True)

    if math.isnan(indices[0]):
        print("LEJA FAIL - Try increasing numPointsForLejaCandidates and/or the numQuadFit paramaters.")
        return 0, 0, indices, False

    # lejaPointsFinal = VT.map_from_canonical_space(lejaPointsFinal, scale)

    plot= False
    if plot:
        plt.figure()
        plt.plot(Mesh,Pdf, '*k', label='mesh', markersize=14)
        plt.plot(point,pointPDF,  '*r',label='Main Point',markersize=14)
        plt.plot(lejaPointsFinal, Pdf[indices], '.c', label='Leja Points',markersize=10)
        plt.legend()
        plt.show()

    indicesNew = indik[indices]
    return Mesh[indicesNew], Pdf[indicesNew], indicesNew, True


