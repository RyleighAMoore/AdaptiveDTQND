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
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def driftfun(mesh):
    # return -1/2*np.tanh(mesh)*(1/np.cosh(mesh))**2
    return 0*mesh

def difffun(mesh):
    # return 1/np.cosh(mesh)
    return np.ones(np.shape(mesh))


def AndersonMattingly(T, h, k, plot=False):
    # simulation parameters
    # T = 5
    s = 0.75
    # h=0.1
    init = 0
    numsteps = int(np.ceil(T/h))-1
    # k = h**s
    yM = k*(np.pi/(k**2))
    # yM=15
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
    
        
    # pdf after one time step with Dirac \delta(x-init) initial condition
    mymu = init + driftfun(init)*h
    mysigma = abs(difffun(init))*np.sqrt(h)
    scale = GaussScale(1)
    scale.setMu(np.asarray([mymu]))
    scale.setCov(np.asarray([mysigma**2]))
    phat = fun.Gaussian(scale, xvec)
    
    PdfTraj =[]
    PdfTraj.append(phat)
    for i in range(numsteps): 
        phat = k*(A@phat)
        PdfTraj.append(phat)
        
        
    trueSoln = []
    from exactSolutions import OneDdiffusionEquation
    for i in range(len(PdfTraj)):
        t=(i+1)*h
        truepdf = OneDdiffusionEquation(np.expand_dims(xvec,1), difffun(xvec), (i+1)*h, 0)
        # truepdf = (2*np.pi*t)**(-1/2)*np.cosh(xvec)*np.exp(-(np.sinh(xvec))**2/(2*t))
        # truepdf = solution(xvec,-1,T)
        trueSoln.append(np.squeeze(np.copy(truepdf)))
    
    from Errors import ErrorValsExact
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(xvec, PdfTraj, trueSoln, plot=False)
    
    # compare solutions
    if plot == True:
        plt.figure()
        plt.plot(xvec,PdfTraj[-1],'o')
        plt.plot(xvec,trueSoln[-1],'.r')
    
    return LinfErrors, L2Errors, L1Errors, L2wErrors


T=10
h = np.arange(0.05, 2.1, 0.5)
k = np.arange(0.1, 0.2, 0.05)
# h=[0.05]

hs = []
ks = []
errors = []
for hh in h:
    # for kk in k:
        print(hh)
        LinfErrors, L2Errors, L1Errors, L2wErrors = AndersonMattingly(T, hh, hh**0.75, plot=False)
        hs.append(hh)
        # ks.append(kk)
        errors.append(L2wErrors[-1])
        
        
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


plt.loglog(h, errors, label = "L2w Error")
plt.loglog(h, h**(15), label="h^15")
plt.show()
plt.xlabel("h")
plt.ylabel("L2w Error")
plt.legend()


# axes instance
# fig = plt.figure(figsize=(6,6))
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# # plot
# sc = ax.scatter(hh, kk, errors)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# # legend
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# # save
# plt.savefig("scatter_hue", bbox_inches='tight')
        