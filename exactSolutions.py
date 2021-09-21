import numpy as np





#Solution to Dd^2p/dx^2 + Dd^2p/dy^2 - dp/dt=0
def TwoDdiffusionEquation(mesh, D, t, A):
    D = D**2*0.5
    r = (mesh[:,0]-A*t)**2 + (mesh[:,1])**2
    vals = np.exp(-r/(4*D*t))*(1/(4*np.pi*D*t))
    return vals


def ThreeDdiffusionEquation(mesh, D, t, A):
    N=3
    D = D**2*0.5
    r = (mesh[:,0]-A*t)**2 + (mesh[:,1])**2 + (mesh[:,2])**2
    den = 4*D*t

    vals = np.exp(-r/(den))*(1/(4*np.pi*D*t)**(N/2))

    return vals

def OneDdiffusionEquation(mesh, D, t, A):
    N=1
    D = D**2*0.5
    r = (mesh[:,0]-A*t)**2
    den = 4*D*t

    vals = np.exp(-r/(den))*(1/(4*np.pi*D*t)**(N/2))

    return vals

def Solution(diff, drift, mesh, t, dim):
    D = diff**2*0.5
    r = (mesh[:,0]-drift*t)**2
    for ii in range(1,dim):
        r += (mesh[:,ii])**2
    vals = np.exp(-r/(4*D*t))*(1/(4*np.pi*D*t))**(dim/2)
    return vals



# index = 15
# mesh = Meshes[index-1]
# ana = TwoDdiffusionEquation(mesh, 0.5,0.01*index, 5)
# # ana2 = TwoDdiffusionEquation(mesh, 0.5,0.1,5)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(mesh[:,0], mesh[:,1], ana, c='k', marker='.')
# ax.scatter(Meshes[index-1][:,0], Meshes[index-1][:,1], PdfTraj[index-1], c='r', marker='.')


# meshMesh = mesh
# index =1
# # mesh = Meshes[index-1]
# ana = TwoDdiffusionEquation(meshMesh, 0.5,0.01*index, 2)
# # ana2 = TwoDdiffusionEquation(mesh, 0.5,0.1,5)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(meshMesh[:,0], meshMesh[:,1], ana, c='k', marker='.')
# # ax.scatter(Meshes[index-1][:,0], Meshes[index-1][:,1], PdfTraj[index-1], c='r', marker='.')
# ax.scatter(meshMesh[:,0], meshMesh[:,1], surfaces[index-1], c='r', marker='.')

