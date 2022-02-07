import numpy as np

def nDGridMeshCenteredAtOrigin(dimension, radius, stepSize, useNoiseBool = False, trimToCircle = True):
        fullNum =  int(radius/stepSize)
        radius2 = fullNum*stepSize + stepSize

        grid= np.mgrid[tuple(slice(-radius2, radius2+stepSize, stepSize) for _ in range(dimension))]
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
            distance = distance < 1.01*radius
            grid  =  grid[distance,:]
        return grid

def nDGridMeshSquareCenteredAroundGivenPoint(dimension, radius, stepSize, centering):
        grid = nDGridMeshCenteredAtOrigin(dimension, radius, stepSize, useNoiseBool = False, trimToCircle = False)
        for dim in range(dimension):
            grid[:, dim] += centering[dim]

        return grid


def findNearestKPoints(Coord, AllPoints, numNeighbors, getIndices = False):
    normList = np.zeros(np.size(AllPoints,0))
    size = np.size(AllPoints,0)
    for i in range(np.size(AllPoints,1)):
        normList += (Coord[i]*np.ones(size) - AllPoints[:,i])**2

    idx = np.argsort(normList)
    if getIndices:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]], idx[:numNeighbors]
    else:
        return AllPoints[idx[:numNeighbors]], normList[idx[:numNeighbors]]


def findNearestPoint(Coord, AllPoints, CoordInAllPoints = False):
    points, normList, indices = findNearestKPoints(Coord, AllPoints, 2, getIndices = True)
    if not CoordInAllPoints:
        return points[0], np.sqrt(normList[0]), indices[0]
    else:
        return points[1], np.sqrt(normList[1]), indices[1]



