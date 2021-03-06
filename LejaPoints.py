import numpy as np
import math

import Functions as fun
from LejaUtilities import get_lu_leja_samples, sqrtNormal_weights
from opolynd import opolynd_eval
np.random.seed(10)

def getLejaPoints(num_leja_samples, initial_samples, poly, num_candidate_samples = 10000, candidateSampleMesh = [], returnIndices = False):
    '''
    Parameters:
    num_leja_samples: Total number of samples to be returned (including initial samples).
    initial_samples: The samples that we must include in the leja sequence.
    poly: Polynomial chaos expansion, fully implemented with options, indices, etc.
    num_candidate_samples: Number of samples that are chosen from to complete the leja sequence after initial samples are used.
    candidateSampleMesh: If num_candidate_samples is zero, this variable defines the candidate samples to use
    returnIndices: Returns the indices of the leja sequence if True.
    '''

    num_vars = np.size(initial_samples,0)

    generate_candidate_samples = lambda n: 7*np.random.normal(0, 1, (num_vars, n))

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
            return [float('nan')], [float('nan')]
        assert successBool == True, "Need to implement returning indices when successBool is False."

    if successBool ==True:
        if returnIndices:
            indicesLeja = data_structures[2]
            return np.asarray(samples).T, indicesLeja
        return np.asarray(samples).T, np.asarray(samples[:,num_initial_samples:]).T


def getLejaSetFromPoints(scale, Mesh, numLejaPointsToReturn, poly, numPointsForLejaCandidates):
    candidatesFull = Mesh # don't need to transform since the scale is normal when this function is used.
    indices = [np.nan]
    candidates, distances, indik = fun.findNearestKPoints(scale.mu, candidatesFull,numPointsForLejaCandidates, getIndices = True)
    point = candidates[0]
    candidates = candidates[1:]

    lejaPointsFinal, indices = getLejaPoints(numLejaPointsToReturn, np.asarray([point]).T, poly, num_candidate_samples = 0, candidateSampleMesh = candidates.T, returnIndices=True)

    if math.isnan(indices[0]):
        return 0, indices, False

    indicesNew = indik[indices]
    return Mesh[indicesNew], indicesNew, True


