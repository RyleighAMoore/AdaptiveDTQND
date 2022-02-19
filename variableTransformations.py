import numpy as np

'''Functions used to transform mesh points.'''

def map_to_canonical_space(user_samples, scale_parameters):
    dimension = np.size(user_samples,1)
    if dimension == 1:
        Linv = 1/np.sqrt(scale_parameters.cov)
        mean = scale_parameters.mu
        delta = np.ones(np.shape(user_samples))*mean
        shiftedMesh = user_samples - delta
        canonical_samples = (Linv*shiftedMesh.T).T
        return canonical_samples

    L = np.linalg.cholesky((scale_parameters.cov))
    Linv = np.linalg.inv(L)
    mean = scale_parameters.mu
    delta = np.zeros(np.shape(user_samples))
    for i in range(np.size(mean)):
        delta[:,i] = mean[i]*np.ones((len(user_samples))).T
    shiftedMesh = user_samples - delta
    canonical_samples = (Linv @ shiftedMesh.T).T
    return canonical_samples

def map_from_canonical_space(user_samples, scale_parameters):
    dimension = np.size(user_samples,1)
    if dimension ==1:
        L = np.sqrt(scale_parameters.cov)
        mean = scale_parameters.mu
        delta = np.ones(np.shape(user_samples))*mean
        vals = L*np.asarray(user_samples).T + delta.T
        return vals.T

    mean = scale_parameters.mu
    delta = np.zeros(np.shape(user_samples))
    for i in range(np.size(mean)):
        delta[:,i] = mean[i]*np.ones((len(user_samples))).T
    cov = scale_parameters.cov
    vals = np.linalg.cholesky(cov)@np.asarray(user_samples).T + delta.T
    return vals.T




