from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def MovingHillDrift(mesh):
    return np.asarray(np.zeros((np.shape(mesh))))
    
def DiagDiffOne(mesh):
    return np.expand_dims(np.asarray(1*np.asarray(np.ones((np.size(mesh))))),1)


mydrift = MovingHillDrift
mydiff = DiagDiffOne

'''Initialization Parameters'''
NumSteps = 4
'''Discretization Parameters'''
a = 1
h=0.01
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
kstepMin = 0.01 # lambda
kstepMax = 0.01 # Lambda
beta = 3
radius = 0.5 # R
dimension = 1
SpatialDiff = False

Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, PrintStuff=True)

pc = []
for i in range(len(Meshes)-1):
    l = len(Meshes[i])
    pc.append(LPReuseArr[i]/l)
    
mean = np.mean(pc)
print("Leja Reuse: ", mean*100, "%")

pc = []
for i in range(len(Meshes)-1):
    l = len(Meshes[i])
    pc.append(AltMethod[i]/l)
    
mean2 = np.mean(pc)
print("Alt Method: ", mean2*100, "%")



def update_graph(num):
    graph.set_data(Meshes[num], PdfTraj[num])
    return title, graph
fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title('2D Test')
    
graph, = ax.plot(Meshes[-1], PdfTraj[-1], linestyle="", marker=".")
ax.set_xlim(-2, 2)
ax.set_ylim(0, np.max(PdfTraj[0]))


ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()

