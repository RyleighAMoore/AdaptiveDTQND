import numpy as np
from scipy.stats import multivariate_normal

# Density, distribution ction, quantile ction and random generation for the
# normal distribution with mean equal to mu and standard deviation equal to sigma.
def dnorm(x, mu, sigma):
    return np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def dnorm_partialx(x, mu, sigma):
    return np.divide(-x+mu,sigma**2)*np.divide(1, (sigma * np.sqrt(2 * np.pi))) * np.exp(np.divide(-(x - mu) ** 2, 2 * sigma ** 2))


def GVals(indexOfMesh, mesh, h):
    val = G(indexOfMesh,mesh,h)
    return val

def Gaussian(scaling, mesh):
    rv = multivariate_normal(scaling.mu.T[0], scaling.cov)        
    soln_vals = np.asarray([rv.pdf(mesh)]).T
    return np.squeeze(soln_vals)

def weightExp(scaling,mesh):
    if np.size(mesh,1) == 1:
        mu = scaling.mu
        cov = scaling.cov
        newvals = np.exp(-(mesh-mu)**2)*(1/cov)
        return np.squeeze(newvals)
        
    mu = scaling.mu
    D = mesh.shape[1]
    cov = scaling.cov
    soln_vals = np.empty(len(mesh))
    # const = 1/(np.sqrt((np.pi)**D*abs(np.linalg.det(cov))))
    invCov = np.linalg.inv(cov)
    for j in range(len(mesh)):
        x = np.expand_dims(mesh[j,:],1)
        Gs = np.exp(-(x-mu).T@invCov@(x-mu))
        soln_vals[j] = Gs
    return soln_vals


def covPart(Px, Py, mesh, cov):
    vals = []
    for i in range(len(mesh)):
        val = np.exp(-2*cov*(mesh[i,0]-Px)*(mesh[i,1]-Py))
        vals.append(np.copy(val))
    return np.asarray(vals)


def G(indexOfMesh,mesh, h, drift, diff, SpatialDiff):
    '''Changing mu and cov over each row'''
    x = mesh[indexOfMesh,:]
    D = mesh.shape[1]
    mean = mesh+drift(mesh)*h
    
    if D == 1:
        newpointVect = x*np.ones(np.shape(mesh))
        var = h*diff(mesh)**2
        newVals = 1/(np.sqrt((2*np.pi*var)))*np.exp(-(newpointVect-mean)**2/(2*var))
        return np.squeeze(newVals)
    
    if not SpatialDiff:
            cov = diff(x)@diff(x).T * h
            const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
            invCov = np.linalg.inv(cov)
        
    soln_vals = np.empty(len(mesh))
    for j in range(len(mesh)):
        if SpatialDiff:
            m = mesh[j,:]
            cov = diff(m)@diff(m).T * h
            const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
            invCov = np.linalg.inv(cov)
        mu = mean[j,:]
        Gs = np.exp(-1/2*((x-mu).T@invCov@(x.T-mu.T)))
        soln_vals[j] = Gs
    return soln_vals*const



def AddPointToG(mesh, newPointindex, h, GMat, drift, diff, SpatialDiff):
    newRow = G(newPointindex, mesh,h, drift, diff, SpatialDiff)
    GMat[newPointindex,:len(newRow)] = newRow

    if np.size(mesh,1) == 1:
        var = diff(mesh[newPointindex,:])**2*h
        mean = mesh[newPointindex,:]+ drift(mesh[newPointindex,:])*h
        newCol = 1/(np.sqrt(2*np.pi*var))*np.exp(-(mesh-mean)**2/(2*var))
        GMat[:len(newCol),newPointindex] = np.squeeze(newCol)
        return GMat
        
    #Now, add new column
    D = mesh.shape[1]
    # mu1 = mesh[-1,:]+drift(np.expand_dims(mesh[-1,:],axis=0))*h
    # mu1 = mu1[0]
    mu = mesh[newPointindex,:]+drift(np.expand_dims(mesh[newPointindex,:],axis=0))*h
    mu = mu[0]
    # assert(np.array_equal(mu1,mu))
    dfn = diff(mesh[newPointindex,:])
    cov = dfn@dfn.T*h
    newCol = np.empty(len(mesh))
    const = 1/(np.sqrt((2*np.pi)**D*abs(np.linalg.det(cov))))
    covInv = np.linalg.inv(cov)
    for j in range(len(mesh)):
        x = mesh[j,:]
        Gs = np.exp(-1/2*((x-mu).T@covInv@(x.T-mu.T)))
        newCol[j] = (Gs)
    GMat[:len(newCol),newPointindex] = newCol*const
    return GMat

from pyopoly1.Scaling import GaussScale

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
    # if np.linalg.det(x) <0:
        
    # v=x
    # if v<0:
    #     v=0
    return x

def AndersonMattingly(yim1, yi, mesh, h, driftfun, difffun, SpatialDiff, theta, a1, a2, dimension):
    # yim1 =np.asarray([0])
    # yi = np.asarray([0])
    
    # for yi in mesh:
    mu1 = yim1 + driftfun(yim1)*theta*h
    sig1 = abs(difffun(yim1))*np.sqrt(theta*h)
    scale = GaussScale(dimension)
    scale.setMu(np.asarray(mu1.T))
    scale.setCov(np.asarray(sig1**2))

    N1 = Gaussian(scale, mesh)
    
    mu2 = yi + (a1*driftfun(yi) - a2*driftfun(yim1))*(1-theta)*h
    sig2 = np.sqrt(rho2(a1*difffun(yi)**2 - a2*difffun(yim1)**2))*np.sqrt((1-theta)*h)
    
    scale2 = GaussScale(dimension)
    scale2.setMu(np.asarray(mu2.T))
    scale2.setCov(np.asarray(sig2**2))
    
    # N2 = np.exp(-(xrow-mu2)**2/(2*sig2*sig2))/(sig2*np.sqrt(2*np.pi))
    N2 = Gaussian(scale2, mesh)
    
    # combCov = 1/(1/scale.cov + 1/scale2.cov)
    # combMu = (scale.mu/scale.cov + scale2.mu/scale2.cov)*combCov
    # S = 1/(np.sqrt(2*np.pi*(scale.cov + scale2.cov)))*np.exp(-(scale.mu-scale2.mu)**2/(2*(scale.cov+scale2.cov)))
    
    
    # scaleComb = GaussScale(dimension)
    # scaleComb.setMu(np.asarray(combMu.T))
    # scaleComb.setCov(np.asarray(combCov))
    
    # N = Gaussian(scaleComb, mesh)
    
    # Integrand = N*S
    
    
    
    # return N*S, scaleComb
    return N1*N2, 0
        



    
    # xvec = mesh
    # xcol = mesh[indexOfMesh]
    # # xrow = 
    # A2 = np.zeros((len(xvec),len(xvec)))

    # for i in range(len(xvec)):
    #     print(i)
    #     xrow = xvec[i]
    #     for j in range(len(xvec)):
    #         xcol = xvec[j]
    #         prow = []
    #         pvec = []
    #     for m,xm in enumerate(xvec):
    #         xsum = xm
    #         mu1 = xcol + driftfun(xcol)*theta*h
    #         sig1 = abs(difffun(xcol))*np.sqrt(theta*h)
    #         scale = GaussScale(dimension)
    #         scale.setMu(np.asarray(mu1.T))
    #         scale.setCov(np.asarray(sig1**2))
            
    #         N1 = Gaussian(scale, xsum)
    #         pvec.append(N1)
    #         # print(N1)
            
    #         mu2 = xsum + (a1*driftfun(xsum) - a2*driftfun(xcol))*(1-theta)*h
    #         sig2 = np.sqrt(rho2(a1*difffun(xsum)**2 - a2*difffun(xcol)**2))*np.sqrt((1-theta)*h)
            
    #         scale2 = GaussScale(dimension)
    #         scale2.setMu(np.asarray(mu2.T))
    #         scale2.setCov(np.asarray(sig2**2))
            
    #         # N2 = np.exp(-(xrow-mu2)**2/(2*sig2*sig2))/(sig2*np.sqrt(2*np.pi))
    #         N2 = fun.Gaussian(scale2, xrow)
    #         # print(N2)
    #         prow.append(N2)
            
    #     A2[i,j]= k*np.asarray(prow)@np.asarray(pvec)
