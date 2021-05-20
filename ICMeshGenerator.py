import numpy as np
import Functions as fun
import UnorderedMesh as UM
np.random.seed(10)

def getICMesh(radius, stepSize, h):
    meshSpacing = stepSize #DM.separationDistance(mesh)*2
    grid, stepsX, stepsY = UM.generateOrderedGridCenteredAtZero(-radius*2, radius*2, -radius*2, radius*2, meshSpacing , includeOrigin=True)
    noise = np.random.normal(0,1, size = (len(grid),2))
    
    noise = np.random.uniform(-meshSpacing, meshSpacing,size = (len(grid),2))
    
    shake = 0
    noise = -meshSpacing*shake +(meshSpacing*shake - - meshSpacing*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
    grid = grid+noise
    
    x,y = grid.T
    X = []
    Y = []
    for point in range(len(grid)):
        if np.sqrt(x[point]**2 + y[point]**2) < radius:
            X.append(x[point])
            Y.append(y[point])
    
    newGrid = np.vstack((X,Y))
    x,y = newGrid

    return newGrid.T

#Assumes initial condition is a dirac mass centered at the origin
def getICMeshRadius(radius, stepSize, h, dimension):
    if dimension == 1: # equispaced
        newGrid = np.asarray([0])
        newPointX = 0
        while newPointX < radius:
            newPointX = newPointX + stepSize
            newGrid = np.vstack((newGrid, newPointX))
        
        newPointX = 0
        while newPointX > -radius:
            newPointX = newPointX - stepSize
            newGrid = np.vstack((newGrid, newPointX))
        
        
    if dimension ==2: # radial initial condition
        times = np.ceil(radius/stepSize)
        newGrid = np.asarray([0,0])
        r = stepSize 
        count = 1
        while times >0:
            times = times -1
            dThetaTemp = np.arccos((stepSize**2-2*r**2)/(-2*r**2))
            points = int(np.ceil(2*np.pi/(dThetaTemp)))
            dTheta = np.pi*2/points
            for i in range(points):
                newPointX = r*np.cos(i*dTheta)
                newPointY = r*np.sin(i*dTheta)
                newGrid = np.vstack((newGrid, (newPointX, newPointY)))
            r = r+stepSize
            count = count+1
            
    if dimension >2:
        print("need to implement")
        
    return newGrid
        

if __name__ == "__main__":
    g = getICMeshRadius(0.5, 0.1, 0.01, 1)
    # gg = getICMesh(0.5, 0.1, 0.01)