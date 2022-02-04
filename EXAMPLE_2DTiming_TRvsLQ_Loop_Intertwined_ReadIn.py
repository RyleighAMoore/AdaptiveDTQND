# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:43:57 2022

@author: Rylei
"""

import statistics
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
with (open("Output/fileT40_20220203-185630_5.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

betaDict_L2werrors =  objects[0][0]
betaDict_L1errors =  objects[0][1]
betaDict_Linferrors =  objects[0][2]
betaDict_L2errors =  objects[0][3]
betaDict_times = objects[0][4]
betaDict_NumPoints = objects[0][5]
bufferDict_L2werrors =  objects[0][6]
bufferDict_L1errors =  objects[0][7]
bufferDict_Linferrors = objects[0][8]
bufferDict_L2errors =  objects[0][9]
bufferDict_times  =  objects[0][10]
bufferDict_NumPoints=  objects[0][11]
dimension =  objects[0][12]
radius =  objects[0][13]
h= objects[0][14]
betaVals =  objects[0][15]
bufferVals =  objects[0][16]
endTime =  objects[0][17]
spacingLQVals = objects[0][18]
spacingTRVals = objects[0][19]

LQMaxTime = []
LQMinTime = []
LQMedTime = []

LQMaxError =[]
LQMinError = []
LQMedError = []
for beta in betaVals:
    errors = []
    times = []
    for spacing in spacingLQVals:
        times = (betaDict_times[(beta, spacing)])
        LQMaxTime.append(np.copy(np.max(times)))
        maxIndex = np.where(times == np.max(times))[0][0]
        errorMax = betaDict_L2werrors[(beta,spacing)][maxIndex]
        LQMaxError.append(np.copy(errorMax))

        LQMinTime.append(np.copy(np.min(times)))
        minIndex = np.where(times == np.min(times))[0][0]
        errorMin = betaDict_L2werrors[(beta,spacing)][minIndex]
        LQMinError.append(np.copy(errorMin))

        LQMedTime.append(np.median(times))
        medIndex = np.where(times == np.median(times))[0][0]
        errorMed = betaDict_L2werrors[(beta,spacing)][medIndex]
        LQMedError.append(np.copy(errorMed))


TRMaxTime = []
TRMinTime = []
TRMedTime = []

TRMaxError =[]
TRMinError = []
TRMedError = []
for buffer in bufferVals:
    errors = []
    times = []
    for spacing in spacingTRVals:
        times = (bufferDict_times[(buffer, spacing)])
        TRMaxTime.append(np.copy(np.max(times)))
        maxIndex = np.where(times == np.max(times))[0][0]
        errorMax = bufferDict_L2werrors[(buffer,spacing)][maxIndex]
        TRMaxError.append(np.copy(errorMax))

        TRMinTime.append(np.copy(np.min(times)))
        minIndex = np.where(times == np.min(times))[0][0]
        errorMin = bufferDict_L2werrors[(buffer,spacing)][minIndex]
        TRMinError.append(np.copy(errorMin))

        TRMedTime.append(np.median(times))
        medIndex = np.where(times == np.median(times))[0][0]
        errorMed = bufferDict_L2werrors[(buffer,spacing)][medIndex]
        TRMedError.append(np.copy(errorMed))


unitTime = np.asarray(LQMedTime[0])
unitError = np.asarray(LQMedError[0])
plt.figure()
plt.plot(unitError, unitTime/unitTime, "*k", markeredgewidth=1, markersize = "20",markerfacecolor="None", label = "Unit Time")

labelString = 'LQ, \u03B2 = 3'
plt.semilogx(np.asarray(LQMedError[0]), np.asarray(LQMedTime[0])/unitTime, "o", label= labelString)
plt.semilogx(np.asarray(LQMinError[0]), np.asarray(LQMinTime[0])/unitTime,  ".k")
plt.semilogx(np.asarray(LQMaxError[0]), np.asarray(LQMaxTime[0])/unitTime,  ".k")


labelString = 'LQ, \u03B2 = 4'
plt.semilogx(np.asarray(LQMedError[1]), np.asarray(LQMedTime[1])/unitTime, "o", label= labelString)
plt.semilogx(np.asarray(LQMinError[1]), np.asarray(LQMinTime[1])/unitTime,  ".k")
plt.semilogx(np.asarray(LQMaxError[1]), np.asarray(LQMaxTime[1])/unitTime,  ".k")


labelString = 'TR Oracle, buffer = 0%'
plt.semilogx(np.asarray(TRMedError[:3]), np.asarray(TRMedTime[:3])/unitTime, "-s", label= labelString)
plt.semilogx(np.asarray(TRMinError[:3]), np.asarray(TRMinTime[:3])/unitTime,  ".k")
plt.semilogx(np.asarray(TRMaxError[:3]), np.asarray(TRMaxTime[:3])/unitTime,  ".k")

labelString = 'TR, buffer = 50%'
plt.semilogx(np.asarray(TRMedError[3:]), np.asarray(TRMedTime[3:])/unitTime, "-s", label= labelString)
plt.semilogx(np.asarray(TRMinError[3:]), np.asarray(TRMinTime[3:])/unitTime, ".k")
plt.semilogx(np.asarray(TRMaxError[3:]), np.asarray(TRMaxTime[3:])/unitTime, ".k")



# plt.ylim([0, 5])
plt.legend()
plt.xlabel(r'$L_{2w}$ Error')
plt.ylabel("Relative Running Time")
plt.title(r"Error vs. Timing, Moving Hill, $T=40$")



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


# plt.figure()
# count =0
# for betaVal in betaVals:
#     if betaVal in betaDict_errors:
#         Errors = np.asarray(betaDict_errors[betaVal])
#         numPoints = np.asarray(numPointsLQ[count])
#         labelString = 'LQ, \u03B2 = %.2f' %betaVal
#         plt.semilogx(Errors, numPoints, "o", label= labelString)
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
#         plt.loglog(np.asarray(Errors), np.asarray(numPoints), "-s", label= labelString)
#         count +=len(Errors)

# plt.legend()
# plt.xlabel(r'$L_{2w}$ Error')
# plt.ylabel(r'Number of Points')
# plt.ylim([10**3, 10**5])
# plt.title(r"Error vs. # of Points, Moving Hill, $T=40$")

