# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:42:29 2021

@author: Rylei
"""

from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import TwoHillDrift, ComplexDiff
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mydrift = TwoHillDrift
mydiff = ComplexDiff

'''Initialization Parameters'''
NumSteps = 200
'''Discretization Parameters'''
a = 1
h=0.01
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
kstepMin = 0.1 # lambda
kstepMax = 0.12 # Lambda
beta = 3
radius = 1 # R

Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff)

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


from plots import plotErrors, plotRowThreePlots
'''Plot 3 Subplots'''
# plotRowThreePlots(Meshes, PdfTraj, h, [69,139,199], includeMeshPoints=False)
from plots import plotErrors, plotRowThreePlots, plot2DColorPlot, plotRowThreePlotsMesh, plotRowSixPlots

plotRowSixPlots(Meshes, PdfTraj, h, [69,139,199])

def update_graph(num):
    graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
    graph.set_3d_properties(PdfTraj[num])
    title.set_text('3D Test, time={}'.format(num))
    return title, graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
    
graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
ax.set_zlim(0, 4.5)
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()

