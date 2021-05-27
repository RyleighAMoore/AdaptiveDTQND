import numpy as np
import matplotlib.pyplot as plt

# xCoord, yCoord, is the point we are looking for the closests 
# two points to. 
# AllPoints is a Nx2  matrix of all the degrees of freedom.
#  x1   y1
#  x2   y2
#  x3   y3
#  ...  ...
def findNearestKPoints(Coord, AllPoints, numNeighbors, getIndices = False):
    xCoord = Coord[0]
    yCoord= Coord[1]
    normList1 = (xCoord*np.ones(len(AllPoints)) - AllPoints[:,0])**2 + (yCoord*np.ones(len(AllPoints)) - AllPoints[:,1])**2
    
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
    # if normList[0]==0:
    #     return points[1], np.sqrt(normList[1]), indices[1]
    # else:
    return points[0], np.sqrt(normList[0]), indices[0]
    

def plotTri(tri, points):
    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    

def generateOrderedGridCenteredAtZero(xmin, xmax, ymin, ymax, kstep, includeOrigin = True):
    stepsX = int(np.ceil(np.ceil((abs(xmin) + abs(xmax)) / (kstep))/2))
    x =[]
    x.append(0)
    for i in range(1, stepsX):
        x.append(i*kstep)
        x.append(-i*kstep)
        
    stepsY = int(np.ceil(np.ceil((abs(ymin)+ abs(ymax)) / (kstep))/2))
    y =[]
    y.append(0)
    for i in range(1, stepsY):
        y.append(i*kstep)
        y.append(-i*kstep)

    X, Y = np.meshgrid(x, y)
    x1 = []
    x2 = []
    for i in range(len(x)):
        for j in range(len(y)):
            x1.append(X[j,i])
            x2.append(Y[j,i])       
   
    mesh = np.asarray([x1,x2]).T
    if includeOrigin == False:
        mesh = np.delete(mesh,0,0)
    return mesh, stepsX, stepsY
    
