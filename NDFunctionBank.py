import numpy as np

class SimpleDriftSDE:
  def __init__(self, drift, diff, dimension):
    self.drift = drift
    self.diff = diff
    self.dim = dimension
    
  def Drift(self, mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = self.drift
    return dr

  def Diff(self, mesh):
      return self.diff*np.diag(np.ones(self.dim))
  

  def Solution(self, mesh, t):
    D = self.diff**2*0.5
    r = (mesh[:,0]-self.drift*t)**2
    for ii in range(1,self.dim):
        r += (mesh[:,ii])**2
    vals = np.exp(-r/(4*D*t))*(1/(4*np.pi*D*t))**(self.dim/2)
    return vals


# def ThreeDdiffusionEquation(mesh, D, t, A):
#     N=3
#     D = D**2*0.5
#     r = (mesh[:,0]-A*t)**2 + (mesh[:,1])**2 + (mesh[:,2])**2
#     den = 4*D*t
    
#     vals = np.exp(-r/(den))*(1/(4*np.pi*D*t)**(N/2))
    
#     return vals
