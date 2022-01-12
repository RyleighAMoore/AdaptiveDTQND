import numpy as np
import matplotlib.pyplot as plt

def nDGridMeshCenteredAtOrigin(dimension, radius, stepSize, useNoiseBool = False, trimToCircle = True):
        subdivision = radius/stepSize
        # step = radius/subdivision
        grid= np.mgrid[tuple(slice(- radius*2, radius*2, stepSize) for _ in range(dimension))]
        mesh = []
        for i in range(grid.shape[0]):
            new = grid[i].ravel()
            if useNoiseBool:
                shake = 0.1*stepSize
                noise = np.random.uniform(-stepSize, stepSize ,size = (len(new)))
                noise = -stepSize*shake +(stepSize*shake - - stepSize*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
                new = new+noise
            mesh.append(new)
        grid = np.asarray(mesh).T
        if trimToCircle:
            distance = 0
            for i in range(dimension):
                distance += (-grid[:,i])**2
            distance = distance**(1/2)
            distance = distance < radius-0.0001
            grid  =  grid[distance,:]
        return grid

def nDGridMeshSquareCenteredAroundGivenPoint(dimension, radius, stepSize, centering):
        grid = nDGridMeshCenteredAtOrigin(dimension, radius, stepSize, useNoiseBool = False, trimToCircle = False)
        for dim in range(dimension):
            grid[:, dim] += centering[dim]

        return grid


def findNearestKPoints(Coord, AllPoints, numNeighbors, getIndices = False):
    # xCoord = Coord[0]
    # yCoord= Coord[1]
    # normList1 = (xCoord*np.ones(len(AllPoints)) - AllPoints[:,0])**2 + (yCoord*np.ones(len(AllPoints)) - AllPoints[:,1])**2

    normList = np.zeros(np.size(AllPoints,0))
    size = np.size(AllPoints,0)
    for i in range(np.size(AllPoints,1)):
        normList += (Coord[i]*np.ones(size) - AllPoints[:,i])**2

    idx = np.argsort(normList)


    if getIndices:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]], idx[:numNeighbors]
    else:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]]


def findNearestPoint(Coord, AllPoints):
    points, normList, indices = findNearestKPoints(Coord, AllPoints, 2, getIndices = True)
    return points[0], np.sqrt(normList[0]), indices[0]



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



def G(indexOfMesh, mesh, h, drift, diff, SpatialDiff):
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

def G2(indexOfMesh,mesh, h, drift, diff, SpatialDiff):
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


def alpha1(th):
    return(1/(2*th*(1-th)))

def alpha2(th):
  num = (1-th)**2 + th**2
  denom = 2*th*(1-th)
  return(num/denom)
