import numpy as np
from scipy.special import erf

def zeroDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 0
    return dr

def oneDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 1
    return dr

def ptFiveDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 0.5
    return dr

def twoDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = 2
    return dr

def erfDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = 2*erf(10*mesh)
    return dr

def spiralDrift_2D(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    y = mesh[:,1]
    r = np.sqrt(x ** 2 + y ** 2)
    return np.asarray([5*(4*erf(5*x)+2*y)/(r+10), 5*(-2*x+y)/(r+10)]).T


def complextDrift_2D(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    x = mesh[:,0]
    return np.asarray([2*erf(10*x), 0*x]).T

def oneDiffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.diag(np.ones(np.size(mesh,1)))

def pt75Diffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return 0.75*np.diag(np.ones(np.size(mesh,1)))

def ptSixDiffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return 0.6*np.diag(np.ones(np.size(mesh,1)))

def ptfiveDiffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return 0.5*np.diag(np.ones(np.size(mesh,1)))

def oneDiffusion(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.diag(np.ones(np.size(mesh,1)))

def complexDiff(mesh):
    if mesh.ndim == 1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.diag([0.01*mesh[:,0][0]**2+0.5,0.01*mesh[:,1][0]**2+0.5]) + np.ones((2,2))*0.2

def bimodal1D(mesh):
    return mesh*(4-mesh**2)