import numpy as np
import Functions as fun
import UnorderedMesh as UM
np.random.seed(10)

def NDGridMesh(dimension, stepsize, radius, UseNoise = True):
    subdivision = radius/stepsize+1
    step = radius/subdivision
    grid= np.mgrid[tuple(slice(step - radius, radius, step) for _ in range(dimension))]
    mesh = []
    for i in range(grid.shape[0]):
        new = grid[i].ravel()
        # if UseNoise:
            # noise = np.random.normal(0,1, size = (len(grid),2))
            # meshSpacing = stepsize
            # noise = np.random.uniform(-meshSpacing, meshSpacing,size = (len(new)))
            
            # shake = 0.1
            # noise = -meshSpacing*shake +(meshSpacing*shake - - meshSpacing*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
            # new = new+noise
        mesh.append(new)
    grid = np.asarray(mesh).T
    noise = np.random.uniform(-0.001, 0.001, size = (np.shape(grid)))
    grid = grid+noise
    distance = 0
    for i in range(dimension):
        distance += grid[:,i]**2
    distance = distance**(1/2)
    distance = distance < radius
    # grid = np.delete(grid, distance,axis=0)
    grid  =  grid[distance,:]
    # for i in reversed(range(len(grid))):
    #     if distance[i] > radius:
    #         grid = np.delete(grid, grid[i], axis=0)
    
    return grid



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
            noise = 0.1*np.random.normal(0,1)
            newPointX = newPointX + stepSize#+noise
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
            
    if dimension ==3:
        num = radius*2/stepSize
        x = np.linspace(-radius,radius, num=int(num))
        y = np.linspace(-radius,radius, int(num))
        z = np.linspace(-radius, radius, int(num))
        mesh = np.meshgrid(x, y, z)
        newGrid = list(zip(*(dim.flat for dim in mesh)))
        noise = np.random.uniform(-0.001, 0.001, size = (len(newGrid),3))
        newGrid = newGrid +noise
        
    #     times = np.ceil(radius/stepSize)
    #     newGrid = np.asarray([[0, 0, 0]])
    #     r = stepSize 
    #     count = 1
    #     while times > 0:
    #         times = times -1
    #         dThetaTemp = np.arccos((stepSize**2-2*r**2)/(-2*r**2))
    #         points = int(np.ceil(np.pi/(dThetaTemp)))
    #         dTheta = np.pi/points
    #         # points2 = int(np.ceil(np.pi/(dThetaTemp)))
    #         dPhi = np.pi*2/points
    #         for j in range(points):
    #             for i in range(len(newGrid)):
    #                 x = r*np.cos(j*dPhi)*np.sin(i*dTheta)
    #                 y = r*np.sin(j*dPhi)*np.sin(i*dTheta)
    #                 z = r*np.cos(i*dTheta)
    #                 smallest = 100
    #                 # for j in range(len(newGrid)):
    #                 #      g = newGrid[j]
    #                 #      dist = np.sqrt((x-g[0])**2 + (y- g[1])**2+ (z- g[2])**2)
    #                 #      # print(dist)
    #                 #      if dist < smallest:
    #                 #          smallest = dist
    #                 # if smallest > 0.00000000005:
    #                 newGrid = np.vstack((newGrid, (x, y, z)))
    #         r = r+stepSize
    #         count = count+1
    #     # from UnorderedMesh import findNearestKPoints
    #     # for j in reversed(range(len(newGrid))):
    #     #      points, dist = findNearestKPoints(newGrid[j,:], newGrid, 2)
    #     #      if dist[1] < 0.0000000005:
    #     #          newGrid = np.delete(newGrid, newGrid[j], axis=0)
    # import matplotlib.pyplot as plt
    # g= newGrid
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(g[:,0], g[:,1], g[:,2], '.');
    return np.asarray(newGrid)


        

if __name__ == "__main__":
    g = getICMeshRadius(0.1, 0.05, 0.01, 3)
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(g[:,0], g[:,1], g[:,2], '.');
    
    # for i in range(len(g)):
    #     for j in range(len(g)):
    #         if i != j:
    #             dist = np.sqrt((g[i,0]- g[j,0])**2 + (g[i,1]- g[j,1])**2+ (g[i,2]- g[j,2])**2)
    #             if dist < 0.00005:
    #                 print(g[i], g[j])
    #                 print(i,j)
    #                 print(dist)
    
    
    # g = getICMeshRadius(0.5, 0.1, 0.01, 1)
    # gg = getICMesh(0.5, 0.1, 0.01)