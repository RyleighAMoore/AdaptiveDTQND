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



def DTQ(NumSteps, minDistanceBetweenPoints, maxDistanceBetweenPoints, h, degree, meshRadius, drift, diff, dimension, SpatialDiff, parameters, PrintStuff = True, RetG = False, Adaptive = True):
    UpdateMesh = Adaptive
    '''Paramaters'''
    addPointsToBoundaryIfBiggerThanTolerance = 10**(-degree)
    removeZerosValuesIfLessThanTolerance = 10**(-degree-0.5)
    conditionNumForAltMethod = 8
    NumLejas = 10
    numPointsForLejaCandidates = 40
    numQuadFit = 20
    
    '''Paramaters'''
    addPointsToBoundaryIfBiggerThanTolerance = 10**(-degree)
    removeZerosValuesIfLessThanTolerance = 10**(-degree-0.5)
    conditionNumForAltMethod = parameters.conditionNumForAltMethod
    NumLejas =parameters.NumLejas
    numPointsForLejaCandidates = parameters.numPointsForLejaCandidates
    numQuadFit = parameters.numQuadFit

    ''' Initializd orthonormal Polynomial family'''
    poly = HermitePolynomials(rho=0)
    d=dimension
    k = 40    
    lambdas = indexing.total_degree_indices(d, k)
    poly.lambdas = lambdas
    
    '''Generate Alt Leja Points'''
    lejaPointsFinal, new = getLejaPoints(10, np.zeros((d,1)), poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

    
    '''pdf after one time step with Dirac initial condition centered at the origin'''
    # if min(mesh) ==-10000:
    # mesh = M.getICMeshRadius(meshRadius, minDistanceBetweenPoints, h, dimension)
    mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # title = ax.set_title('3D Test')
    # # graph = ax.scatter3D(mesh1[:,0], mesh1[:,1],  mesh1[:,2], marker="o")

    # graph = ax.scatter3D(mesh[:,0], mesh[:,1],  mesh[:,2], marker=".")
    
    scale = GaussScale(dimension)
    scale.setMu(h*drift(np.zeros(dimension)).T)
    scale.setCov((h*diff(np.zeros(dimension))*diff(np.zeros(dimension)).T).T)
    
    # from watchpoints import watch
    pdf = fun.Gaussian(scale, mesh)
    
    # for val in pdf:
    #     if val < 10**(-degree*2):
    #         mesh = np.delete(mesh, )
    
    
    Meshes = []
    PdfTraj = []
    PdfTraj.append(np.copy(pdf))
    Meshes.append(np.copy(mesh))

    if dimension > 1:    
        '''Delaunay triangulation for finding the boundary '''
        tri = Delaunay(mesh, incremental=True)
    else: 
        tri = 0
    
    # needLPBool = numpy.zeros((2, 2), dtype=bool)
    '''Initialize Transition probabilities'''
    maxDegFreedom = len(mesh)*2*dimension
    # NumLejas = 15
    # numQuadFit = max(20,20*np.max(diff(np.asarray([0,0]))).astype(int))*2


    GMat = np.empty([maxDegFreedom, maxDegFreedom])*np.NaN
    for i in range(len(mesh)):
        v = fun.G(i,mesh, h, drift, diff, SpatialDiff)
        GMat[i,:len(v)] = v
        
    # from mpl_toolkits.mplot3d.art3d import juggle_axes
        
# xjmat = np.repeat(mesh, len(mesh), axis=1)
# xstarmat = xjmat.T
# fig = plt.figure()
# plt.scatter(xstarmat,xjmat, c=GMat[:len(mesh), :len(mesh)], cmap='bone_r', marker=".")
# plt.ylabel("$y_i$")
# plt.xlabel(r"$y_{i-1}$")
# plt.title("Euler-Maruyama method kernel")
# plt.colorbar()
# plt.show()

    LPMat = np.ones([maxDegFreedom, NumLejas])*-1
    LPMatBool = np.zeros((maxDegFreedom,1), dtype=bool) # True if we have Lejas, False if we need Lejas
        
    '''Grid updates'''
    if PrintStuff:
        LPReuseArr = []
        AltMethod = []
    
    for i in range(1,NumSteps): # Since the first step is taken before this loop.
        print(i)
        if (i >= 3) and UpdateMesh:
            # plt.plot(mesh,pdf,'.')
            '''Add points to mesh'''
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1])
            mesh, pdf, tri, addBool, GMat = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, minDistanceBetweenPoints, h, poly, GMat, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints, drift, diff, SpatialDiff)
            # plt.plot(mesh[:,0], mesh[:,1], '*r')
        if i>=9 and i%10==1 and UpdateMesh:
            '''Remove points from mesh'''
            mesh, pdf, GMat, LPMat, LPMatBool, tri = MeshUp.removePointsFromMeshProcedure(mesh, pdf, tri, True, poly, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance)
        
        if PrintStuff:
            print('Length of mesh = ', len(mesh))
        if i >-1: 
            
            '''Step forward in time'''
            pdf = np.expand_dims(pdf,axis=1)
            pdf, meshTemp, LPMat, LPMatBool, LPReuse, AltMethodCount = LQ.Test_LejaQuadratureLinearizationOnLejaPoints(mesh, pdf, poly,h,NumLejas, i, GMat, LPMat, LPMatBool, numQuadFit, removeZerosValuesIfLessThanTolerance, conditionNumForAltMethod, drift, diff, numPointsForLejaCandidates,SpatialDiff, lejaPointsFinal, PrintStuff)
            pdf = np.squeeze(pdf)
            '''Add new values to lists for graphing'''
            
            PdfTraj.append(np.copy(pdf))
            Meshes.append(np.copy(mesh))
            if PrintStuff:
                LPReuseArr.append(LPReuse)
                AltMethod.append(AltMethodCount)
             
        else:
            if PrintStuff:
                print('Length of mesh = ', len(mesh))
        
        sizer = len(mesh)
        if np.shape(GMat)[0] - sizer < sizer:
            GMat2 = np.empty([2*sizer, 2*sizer])*np.NaN
            GMat2[:sizer, :sizer]= GMat[:sizer, :sizer]
            GMat = GMat2
                
        if np.shape(LPMat)[0] - sizer < sizer:
            LPMat2 = np.ones([2*sizer, NumLejas])*-1
            LPMat2[:sizer,:]= LPMat[:sizer, :]
            LPMat = LPMat2
            LPMatBool2 = np.zeros((2*sizer,1), dtype=bool)
            LPMatBool2[:len(mesh)]= LPMatBool[:len(mesh)]
            LPMatBool = LPMatBool2
        
    if RetG:
        return Meshes, PdfTraj, LPReuseArr, AltMethod, GMat
    if PrintStuff:
        return Meshes, PdfTraj, LPReuseArr, AltMethod        
    else: 
        return Meshes, PdfTraj, [], []
