# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:43:57 2022

@author: Rylei
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '18'
fontprops = {'fontweight': 'bold'}


objects = []
with (open("Output_Saved\\fileT20.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

betaDict_times = objects[0][0]
betaDict_errors = objects[0][1]
bufferDict_times = objects[0][2]
bufferDict_errors = objects[0][3]
betaVals = objects[0][4]
bufferVals=objects[0][5]
spacingLQVals= objects[0][6]
spacingTRVals= objects[0][7]
numPointsLQ=objects[0][8]
numPointsTR = objects[0][9]
h=objects[0][10]
radius = objects[0][11]
endTime = objects[0][12]
allTimingsArrayStorageLQ = objects[0][13]
allErrorsTimingArrayStorageLQ=objects[0][14]
allTimingsArrayStorageTR = objects[0][15]

unitTime = np.asarray(betaDict_times[min(betaVals)])[0]
unitError = np.asarray(betaDict_errors[min(betaVals)])[0]
plt.figure()
plt.plot(unitError, unitTime/unitTime, "*k", markeredgewidth=1, markersize = "20",markerfacecolor="None", label = "Unit Time")
for betaVal in betaVals:
    if betaVal in betaDict_errors:
        Errors = betaDict_errors[betaVal]
        timing = betaDict_times[betaVal]
        labelString = 'LQ, \u03B2 = %.2f' %betaVal
        plt.semilogx(np.asarray(Errors), np.asarray(timing)/unitTime, "o", label= labelString)

for buff in bufferVals:
    if buff in bufferDict_errors:
        Errors = bufferDict_errors[buff]
        timing = bufferDict_times[buff]
        if buff == 0:
            labelString = 'TR Oracle, buffer = %d%%' %(buff*100)
        else:
            labelString = 'TR, buffer = %d%%' %(buff*100)
        plt.semilogx(np.asarray(Errors), np.asarray(timing)/unitTime, "-s", label= labelString)

plt.ylim([0, 5])
plt.legend()
plt.xlabel(r'$L_{2w}$ Error')
plt.ylabel("Relative Running Time")
plt.title(r"Error vs. Timing, Moving Hill, $T=20$")



# plt.figure()
# count =0
# for betaVal in betaVals:
#     if betaVal in betaDict_errors:
#         Errors = np.asarray(betaDict_errors[betaVal])
#         numPoints = np.asarray(numPointsLQ[count])
#         labelString = 'LQ, \u03B2 = %.2f' %betaVal
#         plt.semilogy(numPoints, Errors, "o", label= labelString)
#         count +=1

# count = 0
# for buff in bufferVals:
#     if buff in bufferDict_errors:
#         Errors = bufferDict_errors[buff]
#         numPoints = numPointsTR[count:count + len(Errors)]
#         if buff == 0:
#             labelString = 'TR Oracle, buffer = %d%%' %(buff*100)
#         else:
#             labelString = 'TR, buffer = %d%%' %(buff*100)
#         plt.semilogy(np.asarray(numPoints), np.asarray(Errors), "-s", label= labelString)
#         count +=len(Errors)

# plt.legend()
# plt.ylabel(r'$L_{2w}$ Error')
# plt.xlabel("Number of Points")


plt.figure()
count =0
for betaVal in betaVals:
    if betaVal in betaDict_errors:
        Errors = np.asarray(betaDict_errors[betaVal])
        numPoints = np.asarray(numPointsLQ[count])
        labelString = 'LQ, \u03B2 = %.2f' %betaVal
        plt.semilogx(Errors, numPoints, "o", label= labelString)
        count +=1

count = 0
for buff in bufferVals:
    if buff in bufferDict_errors:
        Errors = bufferDict_errors[buff]
        numPoints = numPointsTR[count:count + len(Errors)]
        if buff == 0:
            labelString = 'TR Oracle, buffer = %d%%' %(buff*100)
        else:
            labelString = 'TR, buffer = %d%%' %(buff*100)
        plt.semilogx(np.asarray(Errors), np.asarray(numPoints), "-s", label= labelString)
        count +=len(Errors)

plt.legend()
plt.xlabel(r'$L_{2w}$ Error')
plt.ylabel(r'Number of Points')
plt.ylim([0, 40000])
plt.title(r"Error vs. # of Points, Moving Hill, $T=20$")

