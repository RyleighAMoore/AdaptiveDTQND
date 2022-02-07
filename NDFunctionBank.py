import numpy as np

class SimpleSDE:
  def __init__(self, drift, diff, dimension):
    self.drift = drift
    self.diff = diff
    self.dim = dimension

  def drift(self, mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[:,0] = self.drift
    return dr

  def diff(self, mesh):
      return self.diff*np.diag(np.ones(self.dim))

  def solution(self, mesh, t):
    D = self.diff**2*0.5
    r = (mesh[:,0]-self.drift*t)**2
    for ii in range(1,self.dim):
        r += (mesh[:,ii])**2
    vals = np.exp(-r/(4*D*t))*(1/(4*np.pi*D*t))**(self.dim/2)
    return vals
