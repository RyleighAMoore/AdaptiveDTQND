import numpy as np
import matplotlib.pyplot as plt
import GMatrix as GMatrix
import pickle

'''These are the trapadzoidal rules used in the 1D code,
if you are looking for the leja quadrature rules see pyopoly1/QuadratureRules.py'''

def TrapUnequal(G, phat, kvect):
    first = np.matmul(G[:, :-1], phat[:-1] * kvect)
    second = np.matmul(G[:, 1:], phat[1:] * kvect)
    half = (first + second) * 0.5
    return half


def Unequal_Gk(G, kvect, xvec, h):
    GW = np.zeros((len(kvect) + 1, len(kvect) + 1))
    # for col in range(len(G)):  # interiors
    #     for row in range(1, len(G) - 1):
    #         GW[row, col] = ((G[row, col] * (xvec[row] - xvec[row - 1])) + (
    #                     G[row, col] * (xvec[row + 1] - xvec[row]))) * 0.5
    #
    # for col in range(len(G)):  # interiors
    #     GW[0, col] = (G[0, col]) * kvect[0] * 0.5
    #
    # for col in range(len(G)):  # interiors
    #     GW[-1, col] = (G[-1, col]) * kvect[-1] * 0.5

    KA = np.concatenate((kvect*0.5, 0), axis=None)
    KB = np.concatenate((0, kvect*0.5), axis=None)
    K = (KA + KB)
    KDiag = np.diag(K, 0)
    GW = np.matmul(G, KDiag)
    WG = np.matmul(KDiag,G)
    WGWinv = np.matmul(np.matmul(KDiag,GW),np.linalg.inv(KDiag))
    plt.figure()
    for i in range(len(G)):
        if i % 30 == 0:
            plt.plot(xvec, G[:, i], label='Gk Col')
        # plt.plot(xvec, G[:, i], label='G Col')
        #plt.plot(xvec, WG[i, :], label='WG Row')
    #plt.plot(xvec, GW[:, 10], label = 'GW Col')
    #plt.plot(xvec, G[:, 10], label = 'G Col')
    #plt.plot(xvec, WG[10, :], label='WG Row')
    #plt.plot(xvec, G[10, :], label='G Row')
    #plt.legend()
    plt.show()
    colSums = np.sum(GW, axis=0)
    rowSums = np.sum(GW, axis=1)
    #GW = np.matmul(GW)
    sums = np.sum(WGWinv, axis=0)
    vals, vects = np.linalg.eig(WGWinv)
    vals = np.abs(vals)
    largest_eigenvector_unscaled = vects[:, 0]
    largest_eigenvector_unscaled1 = vects[:, 1]

    vals = np.real(vals)
    # scaled_eigvect = GMatrix.scaleEigenvector(largest_eigenvector_unscaled,kvect)
    plt.figure()
    plt.plot(xvec, abs(largest_eigenvector_unscaled))
    plt.plot(xvec, abs(largest_eigenvector_unscaled1))

    file = open('WG.p', 'wb')
    pickle.dump(WG, file)
    file.close()
    file = open('GW.p', 'wb')
    pickle.dump(GW, file)
    file.close()
    file = open('xvec.p', 'wb')
    pickle.dump(xvec, file)
    file.close()
    file = open('G.p', 'wb')
    pickle.dump(G, file)
    file.close()
    file = open('W.p', 'wb')
    pickle.dump(KDiag, file)
    file.close()

    plt.show()
    return GW
