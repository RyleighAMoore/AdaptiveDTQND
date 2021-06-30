import numpy as np
import Functions as fun
from scipy.spatial import Delaunay
import LejaQuadrature as LQ
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
import MeshUpdates2D as MeshUp
from pyopoly1.Scaling import GaussScale
import ICMeshGenerator as M
from pyopoly1.LejaPoints import getLejaSetFromPoints, getLejaPoints
import matplotlib.pyplot as plt
from DTQAdaptive import DTQ
import numpy as np
# from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def driftfun(mesh):
    return 0*mesh

def difffun(mesh):
    return np.ones(np.shape(mesh))

# simulation parameters
T = 1
s = 0.75
h = 0.1
init = 0
numsteps = int(np.ceil(T/h))-1
k = h**s
yM = k*(np.pi/(k**2))
M = int(np.ceil(yM/k))
xvec = k*np.linspace(-M,M,2*M+1)


def alpha1(th):
    return(1/(2*th*(1-th)))

def alpha2(th):
  num = (1-th)**2 + th**2
  denom = 2*th*(1-th)
  return(num/denom)

def rho(x):
  v = x
  v[x<0] = 0
  return(v)

def rho2(x):
    v=x
    if v<0:
        v=0
    return v

theta = 0.5
a1 = alpha1(theta)
a2 = alpha2(theta)

A = np.zeros((2*M+1,2*M+1))

xjmat = np.repeat(np.expand_dims(xvec,1), 2*M+1, axis=1)
xstarmat = xjmat.T
for i in range(2*M+1):
    xjm1 = xvec[i]
    mu1 = xjm1 + driftfun(xjm1)*theta*h
    sig1 = abs(difffun(xjm1))*np.sqrt(theta*h)
    scale = GaussScale(1)
    scale.setMu(np.asarray([mu1]))
    scale.setCov(np.asarray([sig1**2]))
    
    pvec = fun.Gaussian(scale, xvec)
 
    mu2 = xstarmat + (a1*driftfun(xstarmat) - a2*driftfun(xjm1))*(1-theta)*h
    sig2 = np.sqrt(rho(a1*difffun(xstarmat)**2 - a2*difffun(xjm1)**2))*np.sqrt((1-theta)*h)
    pmat = np.exp(-(xjmat-mu2)**2/(2*sig2*sig2))/(sig2*np.sqrt(2*np.pi))
    
    A[:,i] = k*(pmat @ pvec)

# A2 = np.zeros(np.shape(A))
# val=0
# for i in range(len(xvec)):
#     print(i)
#     xrow = xvec[i]
#     for j in range(len(xvec)):
#         xcol = xvec[j]
#         prow = []
#         pvec = []
#         for m,xm in enumerate(xvec):
#             xsum = xm
#             mu1 = xcol + driftfun(xcol)*theta*h
#             sig1 = abs(difffun(xcol))*np.sqrt(theta*h)
#             scale = GaussScale(1)
#             scale.setMu(np.asarray([mu1]))
#             scale.setCov(np.asarray([sig1**2]))
            
#             N1 = fun.Gaussian(scale, xsum)
#             pvec.append(N1)
#             # print(N1)
            
#             mu2 = xsum + (a1*driftfun(xsum) - a2*driftfun(xcol))*(1-theta)*h
#             sig2 = np.sqrt(rho2(a1*difffun(xsum)**2 - a2*difffun(xcol)**2))*np.sqrt((1-theta)*h)
            
#             scale2 = GaussScale(1)
#             scale2.setMu(np.asarray([mu2]))
#             scale2.setCov(np.asarray([sig2**2]))
            
#             # N2 = np.exp(-(xrow-mu2)**2/(2*sig2*sig2))/(sig2*np.sqrt(2*np.pi))
#             N2 = fun.Gaussian(scale2, xrow)
#             # print(N2)
#             prow.append(N2)
            
#         A2[i,j]= k*np.asarray(prow)@np.asarray(pvec)
    
    
# pdf after one time step with Dirac \delta(x-init) initial condition
mymu = init + driftfun(init)*h
mysigma = abs(difffun(init))*np.sqrt(h)
scale = GaussScale(1)
scale.setMu(np.asarray([mymu]))
scale.setCov(np.asarray([mysigma**2]))
phat = fun.Gaussian(scale, xvec)

PdfTraj =[]
for i in range(numsteps): 
    phat = k*(A@phat)
    PdfTraj.append(phat)
    
    
# compare solutions
plt.figure()
plt.plot(xvec,PdfTraj[-1],'o')
truepdf = np.exp(-xvec**2/(1 - np.exp(-2*T)))/np.sqrt(np.pi*(1-np.exp(-2*T)))
plt.plot(xvec,truepdf,'.r')
    

def update_graph(num):
    graph.set_data(xvec, PdfTraj[num])
    return title, graph

fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title('2D Test')
    
graph, = ax.plot(xvec, PdfTraj[-1], linestyle="", marker=".")
ax.set_xlim(-4, 4)
ax.set_ylim(0, np.max(PdfTraj[0]))


ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()
    


from mpl_toolkits.mplot3d.art3d import juggle_axes

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
graph = ax.scatter3D(xstarmat,xjmat, A, cmap='bone_r', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")
plt.xlabel("xstar")
plt.ylabel("xym1")
plt.show()

        
from mpl_toolkits.mplot3d.art3d import juggle_axes
    
xvec = np.expand_dims(xvec,1)
xjmat = np.repeat(xvec, len(xvec), axis=1)
xstarmat = xjmat.T
fig = plt.figure()
plt.scatter(xstarmat,xjmat, c=np.abs(A-GMat[:201, :201]), cmap='bone_r', marker=".")
plt.ylabel("$y_i$")
plt.xlabel(r"$y_{i-1}$")
plt.title("Difference")
plt.colorbar()
plt.show()