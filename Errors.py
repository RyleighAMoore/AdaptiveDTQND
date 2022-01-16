import numpy as np
from scipy.interpolate import griddata, interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import opolynd as op
import families as f
from indexing import total_degree_indices
import numpy as np
import matplotlib.pyplot as plt


def ErrorValsOneTime(Meshes, PdfTraj, mesh2, surfaces, interpolate= True):

    # Interpolate the fine grid soln to the leja points
    if interpolate:
        gridSolnOnLejas = griddata(mesh2, surfaces, Meshes, method='cubic', fill_value=np.min(surfaces))
        gridSolnOnLejas = np.squeeze(gridSolnOnLejas)
    else:
        gridSolnOnLejas = surfaces

    # fig = plt.figure()
    # plt.scatter(Meshes, np.abs((PdfTraj-gridSolnOnLejas)), c='k', marker='.')
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(Meshes[:,0], Meshes[:,1],  np.abs((PdfTraj-gridSolnOnLejas)))
    # plt.show()

    #compute errors
    l2w = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj))**2*gridSolnOnLejas)/np.sum(gridSolnOnLejas))

    l2 = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj)*1)**2)/len(PdfTraj))

    l1 = np.sum(np.abs(gridSolnOnLejas - PdfTraj)*gridSolnOnLejas)/len(PdfTraj)

    linf = np.max(np.abs(gridSolnOnLejas - PdfTraj))
    return linf, l2, l1, l2w



def ErrorVals(Meshes, PdfTraj, mesh2, surfaces, PrintStuff=True):
    L2Errors = []
    LinfErrors = []
    L1Errors = []
    L2wErrors = []
    if PrintStuff:
        print('l2w errors:')

    for step in range(len(PdfTraj)):
        # Interpolate the fine grid soln to the leja points
        gridSolnOnLejas = griddata(mesh2, surfaces[1*step], Meshes[step], method='cubic', fill_value=np.min(surfaces[1*step]))

        #compute errors
        l2w = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step]))**2*gridSolnOnLejas)/np.sum(gridSolnOnLejas))
        L2wErrors.append(np.copy(l2w))
        if PrintStuff:
            print(l2w)

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(Meshes[step][:,0], Meshes[step][:,1], np.abs((PdfTraj[step]-gridSolnOnLejas)), c='k', marker='.')
        # plt.show()

        l2 = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step])*1)**2)/len(PdfTraj[step]))
        L2Errors.append(np.copy(l2))

        l1 = np.sum(np.abs(gridSolnOnLejas - PdfTraj[step])*gridSolnOnLejas)/len(PdfTraj[step])
        L1Errors.append(np.copy(l1))

        linf = np.max(np.abs(gridSolnOnLejas - PdfTraj[step]))
        LinfErrors.append(np.copy(linf))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(Meshes[1][:,0], Meshes[1][:,1], np.abs((gridSolnOnLejas - PdfTraj[1])), c='k', marker='.')
    # ax.scatter(Meshes[1][:,0], Meshes[1][:,1], PdfTraj[0], c='r', marker='.')
    # ax.scatter(mesh2[:,0], mesh2[:,1], surfaces[0], c='k', marker='.')


    x = range(len(L2Errors))
    plt.figure()
    plt.semilogy(x, np.asarray(LinfErrors), label = 'Linf Error')
    plt.semilogy(x, np.asarray(L2Errors), label = 'L2 Error')
    plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
    plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.legend()


def ErrorValsExact(Meshes, PdfTraj, exactSoln,h, plot=True):
    L2Errors = []
    LinfErrors = []
    L1Errors = []
    L2wErrors = []
    print('l2w errors:')
    for step in range(len(PdfTraj)):
        # Interpolate the fine grid soln to the leja point
        #compute errors
        gridSolnOnLejas = exactSoln[step]
        l2w = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step]))**2*gridSolnOnLejas)/np.sum(gridSolnOnLejas))
        L2wErrors.append(np.copy(l2w))
        print(l2w)

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(Meshes[step][:,0], Meshes[step][:,1], np.abs((PdfTraj[step]-gridSolnOnLejas)), c='k', marker='.')
        # plt.show()

        l2 = np.sqrt(np.sum(np.abs((gridSolnOnLejas - PdfTraj[step])*1)**2)/len(PdfTraj[step]))
        L2Errors.append(np.copy(l2))

        l1 = np.sum(np.abs(gridSolnOnLejas - PdfTraj[step])*gridSolnOnLejas)/len(PdfTraj[step])
        L1Errors.append(np.copy(l1))

        linf = np.max(np.abs(gridSolnOnLejas - PdfTraj[step]))
        LinfErrors.append(np.copy(linf))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(Meshes[1][:,0], Meshes[1][:,1], np.abs((gridSolnOnLejas - PdfTraj[1])), c='k', marker='.')
    # ax.scatter(Meshes[1][:,0], Meshes[1][:,1], PdfTraj[0], c='r', marker='.')
    # ax.scatter(mesh2[:,0], mesh2[:,1], surfaces[0], c='k', marker='.')

    if plot:
        # x = range(len(L2Errors))
        x = np.linspace(1,len(PdfTraj), len(PdfTraj))*h
        plt.figure()
        plt.semilogy(x, np.asarray(LinfErrors), label = 'Linf Error')
        plt.semilogy(x, np.asarray(L2Errors), label = 'L2 Error')
        plt.semilogy(x, np.asarray(L1Errors), label = 'L1 Error')
        plt.semilogy(x, np.asarray(L2wErrors), label = 'L2w Error')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()

    return LinfErrors, L2Errors, L1Errors, L2wErrors




