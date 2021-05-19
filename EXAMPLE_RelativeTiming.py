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

mydrift = MovingHillDrift
mydiff = DiagDiffOne

'''Initialization Parameters'''
NumSteps = 115

x = [1.55,3,5,7]
h=0.01

L2wErrorArray = np.zeros((len(x),NumSteps))
LengthArray = []

TimingArray = []

a = 1
kstepMin = 0.15 # lambda
kstepMax = 0.17 # Lambda
radius = 1.5 # R
count = 0
mTimes = []
numTimes = 5
for i in x:
    for j in range(numTimes):
        start = datetime.now()
        Meshes, PdfTraj, LPReuseArr, AltMethod= D.DTQ(NumSteps, kstepMin, kstepMax, h, i, radius, mydrift, mydiff, PrintStuff=False)
        end = datetime.now()
        time = end-start
        print(time)
        mTimes.append(time.total_seconds())
    TimingArray.append(sum(mTimes)/numTimes)
    
    surfaces = []
    for ii in range(len(PdfTraj)):
        ana = TwoDdiffusionEquation(Meshes[ii],mydiff(np.asarray([0,0]))[0,0], h*(ii+1), mydrift(np.asarray([0,0]))[0,0])
        surfaces.append(ana)
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, surfaces, plot=False)

    L2wErrorArray[count,:] = np.asarray(L2wErrors)
    count = count+1
    

'''Discretization Parameters'''

# x = [0.1, 0.15, 0.18]
x = [0.2, 0.15,0.12, 0.1]
# x=[0.1,.15]

h=0.01
L2wErrorArrayT = np.zeros((len(x),NumSteps))
timesArrayT = []
stepArrayT = []
LengthArrayT = []

TimingArrayT = []

meshF = np.asarray(Meshes[0])
meshL = np.asarray(Meshes[-1])

'''Find essential support for tensorized version'''
xmin = min(np.min(meshF[:,0]), np.min(meshL[:,0]))
xmax = max(np.max(meshF[:,0]), np.max(meshL[:,0]))
ymin = min(np.min(meshF[:,1]), np.min(meshL[:,1]))
ymax = max(np.max(meshF[:,1]), np.max(meshL[:,1]))
count = 0
mTimesT = []
for i in x:
    for j in range(numTimes):
        start = datetime.now()
        mesh, surfaces = MatrixMultiplyDTQ(NumSteps, i, h, mydrift, mydiff, xmin, xmax, ymin, ymax)
        end = datetime.now()
        time = end-start
        print(time)

        mTimesT.append(time.total_seconds())
    TimingArrayT.append(sum(mTimesT)/numTimes)
    
    LengthArrayT.append(len(mesh))
    Meshes = []
    for i in range(len(surfaces)):
        Meshes.append(mesh)
        
    solution = []
    for ii in range(len(surfaces)):
        ana = TwoDdiffusionEquation(Meshes[ii],mydiff(np.asarray([0,0]))[0,0], h*(ii+1),mydrift(np.asarray([0,0]))[0,0])
        solution.append(ana)
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, surfaces, solution, plot=False)
    
    L2wErrorArrayT[count,:] = np.asarray(L2wErrors)
    count = count+1
    
import pickle
with open('L2wErrorArrayT115a.pickle', 'wb') as f:
    pickle.dump(L2wErrorArrayT, f)
f.close()
with open('L2wErrorArray115a.pickle', 'wb') as f:
    pickle.dump(L2wErrorArray, f)
f.close()
with open('TimingArrayT115a.pickle', 'wb') as f:
    pickle.dump(np.asarray(TimingArrayT), f)
f.close()
with open('TimingArray115a.pickle', 'wb') as f:
    pickle.dump(np.asarray(TimingArray), f)
f.close()
    
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
plt.semilogx(L2wErrorArray[0,-1],np.asarray(TimingArray[0])/mm, 'ok', label="Unit Time")
plt.ylabel("Relative Time")
plt.xlabel(r"$L_{2w}$ Error")
plt.legend()
plt.show()
    
    
   
