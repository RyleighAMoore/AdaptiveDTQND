"""
Created on Fri Nov 20 18:06:14 2020

@author: Rylei
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Functions as fun
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from scipy.spatial import Delaunay
import pickle
from Errors import ErrorVals
from datetime import datetime
from matplotlib import rcParams

# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = '18'
fontprops = {'fontweight': 'bold'}


def plotErrors(x, LinfErrors, L2Errors, L1Errors, L2wErrors):
    plt.figure()
    plt.semilogy(x, np.asarray(LinfErrors), 'r', label = 'Linf Error, interp')
    plt.semilogy(x, np.asarray(L2Errors), 'b', label = 'L2 Error')
    plt.semilogy(x, np.asarray(L1Errors),'g', label = 'L1 Error')
    plt.semilogy(x, np.asarray(L2wErrors), 'c', label = 'L2w Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()


def plot2DColorPlot(index, Meshes, PdfTraj):
        minVal = 0.002
        maxVal = 0.3
        plt.figure()
        M = []
        S = []
        index = -1
        for x in Meshes[index]:
            M.append(x)
        for x in PdfTraj[index]:
            S.append(x)
        M = np.asarray(M)
        S = np.asarray(S)
        levels = np.round(np.linspace(minVal, maxVal, 10),3)
        plt.plot(M[:,0], M[:,1], '.', c='gray', markersize='1', alpha=0.5)
        cntr2 = plt.tricontour(M[:,0], M[:,1], S, levels=levels[0:1], cmap="viridis")
        plt.gca().set_aspect('equal', adjustable='box')
        # cbar = plt.colorbar(cntr2)
        # cbar.set_label("PDF value")
        plt.xlabel(r'$\mathbf{x}^{(1)}$')
        plt.ylabel(r'$\mathbf{x}^{(2)}$')
        plt.xlim([-8, 8])
        plt.ylim([-8, 8])
        plt.show()



def plotFourSubplots(Meshes, PdfTraj, h, indices):
    minVal = 0.002
    maxVal = 0.3
    fig, axs = plt.subplots(2, 2)
    times = 0
    count1 = [0,0,1,1]
    count2 = [0,1,0,1]
    for ij in indices:
        M= []
        S = []
        index = ij
        for x in Meshes[index]:
            M.append(x)
        # for x in PdfTraj[0]:
        #     S.append(x)
        for x in PdfTraj[index]:
            S.append(x)
        M = np.asarray(M)
        S = np.asarray(S)
        levels = np.round(np.linspace(minVal, maxVal, 10),3)
        cntr2 = axs[count1[times],count2[times]].tricontourf(M[:,0], M[:,1], S, levels=levels, cmap="bone_r", vmin=-5.8, vmax =1)
        val = str(np.round((ij+1)*h,4))
        axs[count1[times],count2[times]].set_title('t = %s' %val)
        axs[count1[times],count2[times]].set_xlim([-8, 8])
        axs[count1[times],count2[times]].set_ylim([-8, 8])
        times = times+1


def plotRowThreePlots(Meshes, PdfTraj, h, indices, includeMeshPoints = False):
    minVal = 0.002
    maxVal = 0.3
    # plt.figure()
    fig, axs = plt.subplots(1, 3)
    times = 0
    for ij in indices:
        M= []
        S = []
        index = ij
        for x in Meshes[index]:
            M.append(x)
        for x in PdfTraj[index]:
            S.append(x)
        M = np.asarray(M)
        S = np.asarray(S)
        if includeMeshPoints:
            axs[times].plot(Meshes[ij][:,0], Meshes[ij][:,1], 'k.', markersize='0.5', alpha=0.3)
        levels = np.round(np.linspace(minVal, maxVal, 10),3)
        cntr2 = axs[times].tricontourf(M[:,0], M[:,1], S, levels=levels, cmap="viridis")
        axs[times].set(adjustable='box', aspect='equal')
        val = str(np.round((ij+1)*h,4))
        axs[times].set_title('t = %s' %val)
        axs[times].set_xlim([-8, 8])
        axs[times].set_ylim([-8, 8])
        for tick in axs[times].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axs[times].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        if times > 0:
            axs[times].set_yticklabels([])
            axs[times].set_yticks([])

        times = times+1
    cbar = plt.colorbar(cntr2, ax=axs[:3], location='bottom')
    cbar.ax.tick_params(labelsize=10)
    fig.text(0.52, 0, r'$\mathbf{x}^{(1)}$', ha='center')
    fig.text(0.04, 0.57, r'$\mathbf{x}^{(2)}$', va='center', rotation='vertical')


def plotRowSixPlots(plottingMax, Meshes, PdfTraj, h, indices, limits):
    # minVal = 0.002
    maxVal = plottingMax
    # plt.figure()
    fig, axs = plt.subplots(2, 3)
    times = 0
    for ij in indices:
        M= []
        S = []
        index = ij
        for x in Meshes[index]:
            M.append(x)
        for x in PdfTraj[index]:
            S.append(x)
        M = np.asarray(M)
        S = np.asarray(S)
        levels = np.linspace(-2.5, np.round(np.log(maxVal)), 19)
        cntr2 = axs[0,times].tricontourf(M[:,0], M[:,1], np.log10(S), levels=levels, cmap="viridis")
        axs[0,times].set(adjustable='box', aspect='equal')
        axs[1,times].set(adjustable='box', aspect='equal')

        axs[1, times].scatter(Meshes[ij][:,0], Meshes[ij][:,1],marker=".", color="k", s=0.005)
        val = str(np.round((ij+1)*h,4))
        axs[0,times].set_title('t = %s' %val)
        axs[0,times].set_xlim([limits[0], limits[1]])
        axs[0,times].set_ylim([limits[2], limits[3]])
        # axs[1,times].set_title('t = %s' %val)
        axs[1,times].set_xlim([limits[0], limits[1]])
        axs[1,times].set_ylim([limits[2], limits[3]])
        for tick in axs[1,times].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axs[1,times].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axs[0,times].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axs[0,times].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        if times > 0:
            axs[0,times].set_yticklabels([])
            # axs[0,times].set_yticks([])
            axs[1,times].set_yticklabels([])
            # axs[1,times].set_yticks([])

        if times > -1:
            axs[0,times].set_xticklabels([])
            # axs[0,times].set_xticks([])


        times = times+1

    # fig.text(0.52, 0.05, r'$\mathbf{x}^{(1)}$', ha='center')
    fig.text(0.52, 0.23, r'$x^{(1)}$', ha='center')
    fig.text(0.04, 0.6, r'$x^{(2)}$', va='center', rotation='vertical')
    cbar = plt.colorbar(cntr2, ax=axs[:3], location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r'$\log_{10}(\hat{p}(\mathbf{x}, t_n))$')



def plotRowThreePlotsMesh(Meshes, PdfTraj, h, indices, includeMeshPoints = False):
    minVal = 0.002
    maxVal = 0.3
    # plt.figure()
    fig, axs = plt.subplots(1, 3)
    times = 0
    for ij in indices:
        M= []
        S = []
        index = ij
        for x in Meshes[index]:
            M.append(x)
        for x in PdfTraj[index]:
            S.append(x)
        M = np.asarray(M)
        S = np.asarray(S)
        if includeMeshPoints:
            axs[times].plot(Meshes[ij][:,0], Meshes[ij][:,1], 'k.', markersize='0.5', alpha=0.3)
        # levels = np.round(np.linspace(minVal, maxVal, 10),3)
        # cntr2 = axs[times].tricontourf(M[:,0], M[:,1], S, levels=levels, cmap="viridis")
        axs[times].set(adjustable='box', aspect='equal')
        val = str(np.round((ij+1)*h,4))
        axs[times].set_title('t = %s' %val)
        axs[times].set_xlim([-8, 8])
        axs[times].set_ylim([-8, 8])
        for tick in axs[times].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axs[times].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        if times > 0:
            axs[times].set_yticklabels([])
            axs[times].set_yticks([])

        times = times+1
    # cbar = plt.colorbar(cntr2, ax=axs[:3], location='bottom')
    # cbar.ax.tick_params(labelsize=10)
    fig.text(0.52, 0.22, r'$\mathbf{x}^{(1)}$', ha='center')
    fig.text(0.04, 0.52, r'$\mathbf{x}^{(2)}$', va='center', rotation='vertical')
# sizes = []
# Times = []
# T = []
# for i in range(1,len(Meshes)):
#     sizes.append(len(Meshes[i]))
#     Times.append((Timing[i]-Timing[i-1]).total_seconds())
#     T.append((Timing[i]-Timing[0]).total_seconds())


# ii = np.linspace(1,len(PdfTraj)-1,len(PdfTraj)-1)/100
# # # plt.plot(ii, np.asarray(LPReuseArr),'o', label = 'Reused Leja Points')
# # plt.plot(ii,sizes,'.',label = 'Mesh Size')
# # # plt.plot(ii,np.asarray(AltMethod),'o',label = 'Alt. Method Used')
# # plt.xlabel('Time')
# # plt.ylabel('Number of Points')
# # plt.legend()

# fig, axs = plt.subplots(3)
# axs[0].plot(ii,sizes,'.')
# axs[0].set(ylabel="Number of Points")
# axs[1].plot(ii, 100*np.asarray(LPReuseArr)/sizes,'.')
# axs[1].set(ylabel="% Reusing Leja Points")
# axs[2].plot(ii,100*np.asarray(AltMethod)/sizes,'.')
# axs[2].set(xlabel="Time", ylabel="% Using Alt. Method")
# # axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))



# # plt.legend()

# # plt.plot(ii, np.asarray(Times),'o',label = 'Reused Leja Points')

# plt.figure()
# plt.plot(ii, np.asarray(T)/60,'.')
# plt.title("Cumulative Timing vs. Time Step: Erf")
# plt.xlabel('Time')
# plt.ylabel('Cumulative time in minutes')


# plt.figure()
# plt.plot(ii, np.asarray(Times)/sizes, '.')
# plt.title("Timing vs. Degrees of Freedom")
# plt.xlabel('Step Size')
# plt.ylabel('Time per Point (seconds)')

# plt.plot(sizes, np.asarray(Times)/sizes, '.')




