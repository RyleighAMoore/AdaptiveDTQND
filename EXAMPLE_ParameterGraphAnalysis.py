import DTQAdaptive as D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import Functions as fun
from DriftDiffFunctionBank import MovingHillDrift, DiagDiffOne
from exactSolutions import TwoDdiffusionEquation
from Errors import ErrorValsExact
import ParametersClass as Param


mydrift = MovingHillDrift
mydiff = DiagDiffOne
dimension = 2



'''Initialization Parameters'''
NumSteps = 115
'''Discretization Parameters'''


x = [1,2,3,4,5,6,7,8,9,10]
x=[2]

h=0.01
times = np.asarray(np.arange(h,(NumSteps+1)*h,h))

L2ErrorArray = np.zeros((len(x),len(times)))
LinfErrorArray = np.zeros((len(x),len(times)))
L1ErrorArray = np.zeros((len(x),len(times)))
L2wErrorArray = np.zeros((len(x),len(times)))
timesArray = []
stepArray = []
count = 0
table = ""

a = 1
kstepMin = 0.15 # lambda
kstepMax = 0.17 # Lambda
# beta = 3
radius = 1.5 # R
SpatialDiff = False
conditionNumForAltMethod = 8
NumLejas =10
numPointsForLejaCandidates = 40
numQuadFit = 20

par = Param.Parameters(conditionNumForAltMethod, NumLejas, numPointsForLejaCandidates, numQuadFit)


for i in x:
    Meshes, PdfTraj, LPReuseArr, AltMethod= D.DTQ(NumSteps, kstepMin, kstepMax, h, i, radius, mydrift, mydiff,dimension, SpatialDiff, par, PrintStuff=False)
    surfaces = []
    for ii in range(len(PdfTraj)):
        ana = TwoDdiffusionEquation(Meshes[ii],mydiff(np.asarray([0,0]))[0,0], h*(ii+1), mydrift(np.asarray([0,0]))[0,0])
        surfaces.append(ana)
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, surfaces, plot=False)
    
    
    table = table + str(i) + "&" +str("{:2e}".format(L2wErrors[-1]))+ "&" +str("{:2e}".format(L2Errors[-1])) + "&" +str("{:2e}".format(L1Errors[-1])) + "&" +str("{:2e}".format(LinfErrors[-1]))  + "&" + str(len(Meshes[-1])) + "\\\ \hline "
    L2ErrorArray[count,:] = np.asarray(L2Errors)
    LinfErrorArray[count,:] = np.asarray(LinfErrors)
    L1ErrorArray[count,:] = np.asarray(L1Errors)
    L2wErrorArray[count,:] = np.asarray(L2wErrors)
    for j in times:
        timesArray.append(j)
    stepArray.append(i)
    count = count+1
    
    
X,Y = np.meshgrid(times,x)
fig = plt.figure()
plt.semilogy(times, L2wErrorArray[-1], c='r', marker='.')
plt.semilogy(times, LinfErrorArray[-1], c='r', marker='.')
plt.show()    

from matplotlib import rcParams
# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '12'
fontprops = {'fontweight': 'bold'}

plt.figure()
count = 0
for k in x:
    print(count)
    # plt.semilogy(x, LinfErrorArray[k,:], label = 'Linf Error')
    # plt.semilogy(x, L2Errors[k,:], label = 'L2 Error')
    # plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
    plt.semilogy(times, L2wErrorArray[count,:], label = r'$\beta = %d$' %stepArray[count])
    plt.xlabel('Time')
    plt.ylabel(r'$L_{2w}$ Error')
    plt.legend()
    count = count+1