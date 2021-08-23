import numpy as np
from pyopoly1.Scaling import GaussScale
from functions import Gaussian

class AndersonMatManager:
  def __init__(self, mesh, theta, a1, a2):
    self.mesh = mesh
    # self.vals = vals
    self.theta = theta
    self.a1 = a1
    self.a2 = a2
    self.meshradius = np.max(mesh)
    
  def initializeVals(self, yim1, yi, driftfun, difffun,h, dimension):
     xsum = []
     mesh = self.mesh
     theta = self.theta
     a1 = self.a1
     a2 = self.a2
     for i in mesh:
        mu2 = i + (a1*driftfun(i) - a2*driftfun(yim1))*(1-theta)*h
        assert a1*difffun(i)**2 - a2*difffun(yim1)**2 > 0
        # sig2 = np.sqrt(rho2(a1*difffun(i)**2 - a2*difffun(yim1)**2))*np.sqrt((1-theta)*h)
        sig2 = np.sqrt(a1*difffun(i)**2 - a2*difffun(yim1)**2)*np.sqrt((1-theta)*h)
        scale2 = GaussScale(dimension)
        scale2. setMu(np.asarray(mu2.T))
        scale2.setCov(np.asarray(sig2**2))
        N2 = Gaussian(scale2, yi)
        xsum.append(N2)
     self.vals = xsum
     
   def addToMesh(self, ):
        mesh = self.mesh
        theta = self.theta
        a1 = self.a1
        a2 = self.a2
        for i in mesh:
           mu2 = i + (a1*driftfun(i) - a2*driftfun(yim1))*(1-theta)*h
           assert a1*difffun(i)**2 - a2*difffun(yim1)**2 > 0
           # sig2 = np.sqrt(rho2(a1*difffun(i)**2 - a2*difffun(yim1)**2))*np.sqrt((1-theta)*h)
           sig2 = np.sqrt(a1*difffun(i)**2 - a2*difffun(yim1)**2)*np.sqrt((1-theta)*h)
           scale2 = GaussScale(dimension)
           scale2. setMu(np.asarray(mu2.T))
           scale2.setCov(np.asarray(sig2**2))
           N2 = Gaussian(scale2, yi)
           xsum.append(N2)
        
