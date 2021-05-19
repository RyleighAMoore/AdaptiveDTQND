# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:50:35 2021

@author: Rylei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:46:21 2021

@author: Rylei
"""
import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import Functions as fun
from DriftDiffFunctionBank import MovingHillDrift, DiagDiffOne
from DTQTensorized import MatrixMultiplyDTQ
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact
from datetime import datetime

import pickle

L2wErrorArray  = pickle.load( open("PickledData//L2wErrorArray115a.pickle", "rb" ) )
L2wErrorArrayT  = pickle.load( open( "PickledData//L2wErrorArrayT115a.pickle", "rb" ) )

TimingArray  = pickle.load( open( "PickledData//TimingArray115a.pickle", "rb" ) )
TimingArrayT  = pickle.load( open( "PickledData//TimingArrayT115a.pickle", "rb" ) ) 

# L2wErrorArray  = np.load( open("L2wErrorArrayj.npy", "rb" ) )
# L2wErrorArrayT  = np.load( open( "L2wErrorArrayTj.npy", "rb" ) )

# TimingArray  = np.load( open( "TimingArrayj.npy", "rb" ) )
# TimingArrayT  = np.load( open( "TimingArrayTj.npy", "rb" ) ) 
    


mm = min(min(TimingArrayT), min(TimingArray))
mm = TimingArray[0]

from matplotlib import rcParams

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '18'
fontprops = {'fontweight': 'bold'}

m = np.max(np.asarray(TimingArrayT)/mm) 
# nearest_multiple = int(5 * round(m/5))
plt.figure()
# plt.yticks(np.arange(0, nearest_multiple+10, 5))
plt.semilogx(L2wErrorArrayT[:,-1],np.asarray(TimingArrayT)/mm, 'o-',label="Tensorized")
plt.semilogx(L2wErrorArray[:,-1],np.asarray(TimingArray)/mm, 'o:r', label="Adaptive")
plt.semilogx(L2wErrorArray[0,-1],np.asarray(TimingArray[0])/mm, '*k', label="Unit Time", markersize=10)
plt.ylabel("Relative Time")
plt.xlabel(r"$L_{2w}$ Error")
plt.legend()
# plt.xticks([10**(-7), 10**(-6), 10**(-5),  10**(-4), 10**(-3),  10**(-2),  10**(-1), 10**(0)])
plt.xticks([10**(-8), 10**(-6),  10**(-4),  10**(-2), 10**(0)])
plt.yticks([0,10,20,30,40,50,60, 70,80])


plt.show()
    