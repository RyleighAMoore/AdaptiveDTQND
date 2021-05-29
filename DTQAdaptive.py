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


def DTQ(NumSteps, minDistanceBetweenPoints, maxDistanceBetweenPoints, h, degree, meshRadius, drift, diff, dimension, SpatialDiff, PrintStuff = True):
    '''Paramaters'''
    addPointsToBoundaryIfBiggerThanTolerance = 10**(-degree)
    removeZerosValuesIfLessThanTolerance = 10**(-degree-0.5)
    conditionNumForAltMethod = 8
    NumLejas = 10
    numPointsForLejaCandidates = 40
    numQuadFit = 20

    ''' Initializd orthonormal Polynomial family'''
    poly = HermitePolynomials(rho=0)
    d=dimension
    k = 40    
    lambdas = indexing.total_degree_indices(d, k)
    poly.lambdas = lambdas
    
    '''Generate Alt Leja Points'''
    lejaPointsFinal, new = getLejaPoints(10, np.zeros((d,1)), poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

    
    '''pdf after one time step with Dirac initial condition centered at the origin'''
    mesh = M.getICMeshRadius(meshRadius, minDistanceBetweenPoints, h, dimension)

    scale = GaussScale(dimension)
    scale.setMu(h*drift(np.zeros(dimension)).T)
    scale.setCov((h*diff(np.zeros(dimension))*diff(np.zeros(dimension)).T).T)
    
    pdf = fun.Gaussian(scale, mesh)
    
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
    maxDegFreedom = len(mesh)*2
    # NumLejas = 15
    # numQuadFit = max(20,20*np.max(diff(np.asarray([0,0]))).astype(int))*2

    
    GMat = np.empty([maxDegFreedom, maxDegFreedom])*np.NaN
    for i in range(len(mesh)):
        v = fun.G(i,mesh, h, drift, diff, SpatialDiff)
        GMat[i,:len(v)] = v
        
    LPMat = np.ones([maxDegFreedom, NumLejas])*-1
    LPMatBool = np.zeros((maxDegFreedom,1), dtype=bool) # True if we have Lejas, False if we need Lejas
        
    '''Grid updates'''
    if PrintStuff:
        LPReuseArr = []
        AltMethod = []
    
    for i in range(1,NumSteps): # Since the first step is taken before this loop.
        print(i)
        if (i >= 0):
            '''Add points to mesh'''
            # plt.figure()
            # plt.scatter(mesh[:,0], mesh[:,1])
            mesh, pdf, tri, addBool, GMat = MeshUp.addPointsToMeshProcedure(mesh, pdf, tri, minDistanceBetweenPoints, h, poly, GMat, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints, drift, diff, SpatialDiff)
            # plt.plot(mesh[:,0], mesh[:,1], '*r')
        if i>=15 and i%10==9:
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
        

    if PrintStuff:
        return Meshes, PdfTraj, LPReuseArr, AltMethod
    else: 
        return Meshes, PdfTraj, [], []
