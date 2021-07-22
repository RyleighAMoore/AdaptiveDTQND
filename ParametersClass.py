import numpy as np

class Parameters:
   def __init__(self, fun, h, conditionNumForAltMethod, beta):
      self.conditionNumForAltMethod = conditionNumForAltMethod
      # self.NumLejas = NumLejas
      # self.numPointsForLejaCandidates = numPointsForLejaCandidates
      # self.numQuadFit = numQuadFit
      diffMax = np.max(fun.Diff(np.zeros(fun.dim)))
      self.h = h
      self.kstepMin = 0.09
      self.kstepMax = 0.95
      self.beta = beta
      self.radius = np.sqrt(diffMax*h)*6 #+0.5*np.exp(-fun.dim+1)+1)
      self.NumLejas = int(10*fun.dim)
      self.numPointsForLejaCandidates = int((1/self.kstepMin)**fun.dim/3)
      self.numQuadFit = int((1/self.kstepMin)**fun.dim/3)
      # self.numPointsForLejaCandidates = 350
      # self.numQuadFit = 350
      # self.maxDiff = None
      # self.minDiff = None
      
      
   # def set_kstepMin(self, dimension, diff, h):
   #     kstepMin = 0.08
   #     self.kstepMin = kstepMin
       

   # def set_kstepMax(self, dimension, diff, h):
   #     kstepMin = 0.09
   #     self.kstepMin = kstepMin
       
   # def set_radius(self, mesh, pdfTraj):
   #     self.radius = radius
       
   # def set_NumLejas(self, dimension, diff):
   #     self.NumLejas = None
       
   # def numPointsForLejaCandidates(self, dimension, diff):
   #     self.numPointsForLejaCandidates = None

   # def numQuadFit(self, dimension, diff):
   #     self.numQuadFit = None
       
       
   

      
       
