import numpy as np

class SimpleDriftSDE:
  def __init__(self, drift, diff, dimension):
    self.drift = drift
    self.diff = diff
    self.dim = dimension
    

  def MovingHillDrift(self, mesh):
    if self.dim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    dr = np.zeros(np.shape(mesh))
    dr[0,:] = self.dirft*np.ones(np.shape(mesh[0,:]))
    return dr

    
  def DiagDiff(self, mesh):
      return self.diff*np.diag(self.dim)

