# Demonstration of constructing interpolants on Gauss quadrature nodes

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from families import HermitePolynomials
import variableTransformations as VT


# Font styling
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
fontprops = {'fontweight': 'bold'}

# Test function
u = lambda xx: xx**2

N=30

H = HermitePolynomials()

xg, wg = H.gauss_quadrature(N)

sigma = .1
mu = 0
scaling = np.asarray([[mu, sigma]])

xCan=VT.map_to_canonical_space(xg,scaling)

V = H.eval(xg, range(np.max(N)))

w2 = (1/(np.sqrt(2*np.pi)))*np.exp(-xg**2/2)

w2 = w2/np.sum(w2)

plt.plot(xg, w2, '.')
plt.plot(xg, wg)

wV = (np.sqrt(w2)*V.T).T
# Quadrature method
c2 = np.dot(wV.T, np.sqrt(w2)*u(xg))




Vinv = np.linalg.inv(wV)
c = np.matmul(Vinv, u(xg))


interp = np.dot(V,c)


plt.figure()
lines = []
lines.append(plt.plot(xg, u(xg))[0])
lines.append(plt.plot(xg, interp, '.')[0])

# # plt.xlabel(r'$x$', **fontprops)
# # plt.title(r'Plot of $u$ and $N$-point interpolants', **fontprops)

