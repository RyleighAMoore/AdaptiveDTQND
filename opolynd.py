'''
Credit: Akil Narayan - Pyopoly
'''

import numpy as np
import opoly1d

def opolynd_eval(x, lambdas, ab, poly1d):
    # Evaluates tensorial orthonormal polynomials associated with the
    # univariate recurrence coefficients ab.
    try:
        M, d = x.shape
    except Exception:
        d = x.size
        M = 1
        x = np.reshape(x, (M, d))

    N, d2 = lambdas.shape

    assert d==d2, "Dimension 1 of x and lambdas must be equal"

    p = np.ones([M, N])

    for qd in range(d):
        p = p * poly1d.eval(x[:,qd], lambdas[:,qd])

    return p


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import opoly1d, indexing
    from families import HermitePolynomials

    #plot 1D
    d = 1
    k = 6

    H = HermitePolynomials()

    ab = H.recurrence(k+1)

    N = 50
    x = np.linspace(-1, 1, N)
    XX = np.reshape(x, (len(x),1))
    lambdas = indexing.total_degree_indices(d, k)
    p = opolynd_eval(XX, lambdas, ab, H)

    fig = plt.figure()
    for i in range(len(p.T)):
        plt.plot(x,p[:,i])
    plt.show()


    #plot 2D
    d = 2
    k = 6

    H = HermitePolynomials()
    H.probability_measure = True

    ab = H.recurrence(k+1)

    N = 50
    x = np.linspace(-6, 6, N)
    X,Y = np.meshgrid(x,x)
    XX = np.concatenate((X.reshape(X.size,1), Y.reshape(Y.size,1)), axis=1)

    lambdas = indexing.total_degree_indices(d, k)

    p = opolynd_eval(XX, lambdas, ab, H)

    j = 6
    assert j < lambdas.shape[0]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, p[:,j].reshape(N,N), cmap=cm.coolwarm, linewidth=0,antialiased=True)
    plt.show()
