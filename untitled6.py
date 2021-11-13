

import Class_PDF
import numpy as np
from Functions import G, alpha1, alpha2
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from Class_PDF import nDGridMeshCenteredAtOrigin
from Class_Gaussian import GaussScale
from tqdm import tqdm
import time


iters = 1000000

dimension = 1
scale1 = GaussScale(dimension)
scale1.setMu(np.asarray(np.asarray([np.zeros(dimension)])))
scale1.setCov(np.asarray((np.asarray(np.asarray([np.ones((dimension,dimension))])))))

meshAM1 = nDGridMeshCenteredAtOrigin(dimension, 70, 0.05, useNoiseBool = False)

start = time.time()
for i in range(iters):
    N2 = scale1.ComputeGaussian(meshAM1[0], dimension)

end = time.time()
print(end-start)


dimension = 2
scale1 = GaussScale(dimension)
scale1.setMu(np.asarray(np.asarray([np.zeros(dimension)])).T)
scale1.setCov(np.asarray([[1,0],[0,1]]))

meshAM = nDGridMeshCenteredAtOrigin(dimension, 1.5, 0.05, useNoiseBool = False)

start = time.time()
for i in range(iters):
    N2 = scale1.ComputeGaussian(meshAM[0], dimension)

end = time.time()
print(end-start)

