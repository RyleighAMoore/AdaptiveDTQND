import numpy as np
from scipy.special import erf

def zeroDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 0
    return dr

def oneDiffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.diag(np.ones(np.size(mesh,1)))


def erfDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = 2*erf(10*mesh)
    return dr

def oneDiffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.diag(np.ones(np.size(mesh,1)))