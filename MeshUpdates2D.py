# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:38:41 2019

@author: Ryleigh
"""

import numpy as np
import Functions as fun
import UnorderedMesh as UM
from scipy.spatial import Delaunay
from itertools import chain
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyopoly1 import LejaPoints as LP
from pyopoly1 import LejaPoints as LP
from scipy.interpolate import griddata
import random
import Circumsphere as CS
from itertools import combinations
from collections import defaultdict
import time


random.seed(10)


def addPointsToMeshProcedure(Mesh, Pdf, triangulation, kstep, h, poly, GMat, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints,drift, diff, SpatialDiff, TimeStepType, dimension, numPointsForLejaCandidates):
    '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
    changedBool2 = 0 
    changedBool1 = 0
    meshSize = len(Mesh)
    Mesh, Pdf, triangulation, changedBool1 = addPointsToBoundary(Mesh, Pdf, triangulation,addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints,h, drift, diff)
    ChangedBool = max(changedBool1, changedBool2)
    if ChangedBool==1:
        newMeshSize = len(Mesh)
        if TimeStepType == "EM":
            for i in range(meshSize+1, newMeshSize+1):
                if TimeStepType == "EM":
                    GMat = fun.AddPointToG(Mesh[:i,:], i-1, h, GMat, drift, diff, SpatialDiff)
        elif TimeStepType == "AM":
            indices = list(range(meshSize, newMeshSize))
            start = time.time()
            GMat = fun.AddPointsToGAndersonMat(Mesh, indices, h, GMat, diff, drift, SpatialDiff, dimension, minDistanceBetweenPoints)
            end = time.time()
            print(end-start)
            # start1 = time.time()
            # GMat3 = fun.GenerateAndersonMatMatrix(h, drift, diff, Mesh, dimension, 500, minDistanceBetweenPoints, SpatialDiff)
            # end1 = time.time()
            # print(end1-start1)


    return Mesh, Pdf, triangulation, ChangedBool, GMat

def removePointsFromMeshProcedure(Mesh, Pdf, tri, boundaryOnlyBool, poly, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance):
    '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
    Mesh, Pdf, GMat,LPMat, LPMatBool, tri = removeSmallPoints(Mesh, Pdf, tri, boundaryOnlyBool, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance)
    return Mesh, Pdf, GMat, LPMat, LPMatBool, tri


def getBoundaryPoints(Mesh, tri, alpha):
    dimension = np.size(Mesh,1)
    if dimension ==1:
        pointsOnBoundary = ([np.min(Mesh), np.max(Mesh)])
    else:
        '''Uses triangulation and alpha hull technique to find boundary points'''
        # if dimension == 2:
        #     pointsOnBoundary = alpha_shape(Mesh, tri, alpha, only_outer=True)
        # elif dimension ==3:
            # pointsOnBoundary = alpha_shape_3D(Mesh, tri, alpha)
        pointsOnBoundary = ND_Alpha_Shape(Mesh, tri, alpha*1.5, dimension)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(Mesh[:,0], Mesh[:,1], Mesh[:,2])
        # ax.scatter(Mesh[pointsOnBoundary,0], Mesh[pointsOnBoundary,1], Mesh[pointsOnBoundary,2], 'r')
        # plt.show()
    return pointsOnBoundary


def checkIntegrandForRemovingSmallPoints(PDF, Mesh, tri, removeZerosValuesIfLessThanTolerance):
    '''Check if any points are tiny and can be removed'''
    # valueList = 10000*np.ones(len(PDF)) # Set to small values for placeholder
    # pointsOnEdge = getBoundaryPoints(Mesh, tri, 0.15*1.5)
    # for i in pointsOnEdge:
    #     valueList[i]=PDF[i]
    possibleZeros = [np.asarray(PDF) < removeZerosValuesIfLessThanTolerance] # want value to be small
    return np.asarray(possibleZeros).T


def checkIntegrandForAddingPointsAroundBoundaryPoints(PDF, addPointsToBoundaryIfBiggerThanTolerance, Mesh, tri,maxDistanceBetweenPoints):
    '''Check if the points on the edge are too big and we need more points around them
    Uses alpha hull to get the boundary points if boundaryOnly is True.'''
    valueList = -1*np.ones(len(PDF)) # Set to small values for placeholder
    pointsOnEdge = getBoundaryPoints(Mesh, tri, maxDistanceBetweenPoints)
    for i in pointsOnEdge:
        valueList[i]=PDF[i]
    addingAround = [np.asarray(valueList) >= addPointsToBoundaryIfBiggerThanTolerance]
    return np.asarray(addingAround).T

#TODO: Rename this, it really removes any small points
#Also need to clean this, no while loop is needed.
def removeSmallPoints(Mesh, Pdf, tri, boundaryOnlyBool, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance):
    '''# Removing boundary points'''
    ZeroPointsBoolArray = checkIntegrandForRemovingSmallPoints(Pdf,Mesh,tri, removeZerosValuesIfLessThanTolerance)
    iivals = np.expand_dims(np.arange(len(Mesh)),1)
    index = iivals[ZeroPointsBoolArray] # Points to remove
    if len(index)>0:
        Mesh = np.delete(Mesh, index, 0)
        Pdf = np.delete(Pdf, index, 0)
        GMat = np.delete(GMat, index,0)
        GMat = np.delete(GMat, index,1)
        largerLPMat = np.zeros(np.shape(LPMat))
        
        for ii in index:
            LPUpdateList = np.where(LPMat == ii)[0]
            for i in LPUpdateList:
                LPMatBool[i] = False
            largerLP = LPMat >= ii
            largerLPMat = largerLPMat + largerLP
        
        LPMat = LPMat - largerLPMat
        
        LPMatBool = np.delete(LPMatBool, index,0)
        LPMat = np.delete(LPMat, index, 0)
        tri = houseKeepingAfterAdjustingMesh(Mesh, tri)
    
    return Mesh, Pdf, GMat, LPMat, LPMatBool, tri

def houseKeepingAfterAdjustingMesh(Mesh, tri):
    '''Updates all the Vertices information for the mesh. Must be run after removing points'''
    if np.size(Mesh,1)==1:
        return 0
    tri = Delaunay(Mesh, incremental=True)
    return tri

import ICMeshGenerator as M
def addPointsToBoundary(Mesh, Pdf, triangulation, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints, h, drift, diff):
    dimension = np.size(Mesh,1)
    ChangedBool = 0
    count = 0
    MeshOrig = np.copy(Mesh)
    PdfOrig = np.copy(Pdf)

    if dimension == 1: # 1D
        left = np.argmin(Mesh)
        right = np.argmax(Mesh)
        newPointsBool = False
        newPoints = []
        if Pdf[left] > addPointsToBoundaryIfBiggerThanTolerance:
            radius = minDistanceBetweenPoints/2 + maxDistanceBetweenPoints/2
            mm = np.min(Mesh)
            MM = np.max(Mesh)
            newPointsBool = True
        
            for i in range(1,5):
                Mesh = np.append(Mesh, np.asarray([[mm-i*radius]]), axis=0)
                newPoints.append(np.asarray(mm-i*radius))
        if Pdf[right] > addPointsToBoundaryIfBiggerThanTolerance:
            radius = minDistanceBetweenPoints/2 + maxDistanceBetweenPoints/2
            mm = np.min(Mesh)
            MM = np.max(Mesh)
            newPointsBool = True
        
            for i in range(1,5):                
                Mesh = np.append(Mesh, np.asarray([[MM+i*radius]]), axis=0)
                newPoints.append(np.asarray(MM+i*radius))
        if newPointsBool:
            interp1 = [griddata(MeshOrig,PdfOrig, np.asarray(newPoints), method='linear', fill_value=np.min(Pdf))][0]
            # interp2 = [griddata(MeshOrig,PdfOrig, np.max(Mesh)+radius, method='linear', fill_value=np.min(Pdf))][0]
            interp1[interp1<0] = np.min(PdfOrig)
            # interp2[interp2<0] = np.min(PdfOrig)
            Pdf = np.append(Pdf, interp1)
            # Pdf = np.append(Pdf, interp2)
            ChangedBool=1
    
    else: 
        while count < 1: 
            count = count + 1
            numPointsAdded = 0
            boundaryPointsToAddAround = checkIntegrandForAddingPointsAroundBoundaryPoints(Pdf, addPointsToBoundaryIfBiggerThanTolerance, Mesh, triangulation,maxDistanceBetweenPoints)
            iivals = np.expand_dims(np.arange(len(Mesh)),1)
            index = iivals[boundaryPointsToAddAround]
            candPoints = M.NDGridMesh(dimension, minDistanceBetweenPoints/2 + maxDistanceBetweenPoints/2,maxDistanceBetweenPoints, UseNoise = False)
            for indx in index:
                # newPoints = addPointsRadially(Mesh[indx,:], Mesh, 8, minDistanceBetweenPoints, maxDistanceBetweenPoints)
                curr =  np.repeat(np.expand_dims(Mesh[indx,:],1), np.size(candPoints,0), axis=1)
                newPoints = candPoints  - curr.T
                points = []
                for i in range(len(newPoints)):
                    newPoint = newPoints[i,:]
                    if len(points)>0:
                        mesh2 = np.vstack((Mesh,points))
                        nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, mesh2)
                    else:
                        nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, Mesh)
                  
                    if distToNearestPoint >= minDistanceBetweenPoints and distToNearestPoint <= maxDistanceBetweenPoints:
                        points.append(newPoint)

                newPoints = points                
                if len(newPoints)>0:
                    Mesh = np.append(Mesh, newPoints, axis=0)
                    ChangedBool = 1
                    numPointsAdded = numPointsAdded + len(newPoints)
            if numPointsAdded > 0:
                newPoints = Mesh[-numPointsAdded:,:]
                # interp = [griddata(MeshOrig, PdfOrig, newPoints, method='nearest', fill_value=np.min(Pdf))][0]
                interp = [griddata(MeshOrig, PdfOrig, newPoints, method='linear', fill_value=np.min(Pdf))][0]

                # interp = np.exp(interp)
                interp[interp<0] = np.min(Pdf)
                # interp = np.ones(len(newPoints))*removeZerosValuesIfLessThanTolerance 
                # interp = np.ones(len(newPoints))*10**(-8)
                Pdf = np.append(Pdf, interp)
                triangulation = houseKeepingAfterAdjustingMesh(Mesh, triangulation)
    return Mesh, Pdf, triangulation, ChangedBool


def addPointsRadially(point, mesh, numPointsToAdd, minDistanceBetweenPoints, maxDistanceBetweenPoints):
    radius = minDistanceBetweenPoints/2 + maxDistanceBetweenPoints/2
    points = [] 
    noise = random.uniform(0, 1)*2*np.pi
    dTheta = 2*np.pi/numPointsToAdd
    numAdded = 0
    if np.size(mesh,1) == 2:
        pointX = point[0]
        pointY= point[1]
        for i in range(numPointsToAdd):
            newPointX = radius*np.cos(i*dTheta + noise) + pointX
            newPointY = radius*np.sin(i*dTheta + noise) + pointY
            
            newPoint = np.expand_dims(np.hstack((newPointX, newPointY)),1)
            if len(points)>0:
                mesh2 = np.vstack((mesh,points))
                nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, mesh2)
            else:
                nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, mesh)
          
            if distToNearestPoint >= minDistanceBetweenPoints and distToNearestPoint <= maxDistanceBetweenPoints:
                points.append([newPointX, newPointY])
                numAdded +=1
        return np.asarray(points)
   
    if np.size(mesh,1)==3:
        pointsSphere = fibonacci_sphere(10)
        r = radius
        pointsSphere[:,0] = r*pointsSphere[:,0] +point[0]
        pointsSphere[:,1] = r*pointsSphere[:,1] +point[1]
        pointsSphere[:,2] = r*pointsSphere[:,2] +point[2]
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(pointsSphere[:,0], pointsSphere[:,1], pointsSphere[:,2])
        # ax.scatter(point[0], point[1], point[2])
        for kk in range(len(pointsSphere)):
            newPoint = pointsSphere[kk,:]
            if len(points)>0:
                nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, mesh)
            else:
                nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, mesh)
                # print(distToNearestPoint)
            if distToNearestPoint >= minDistanceBetweenPoints and distToNearestPoint <= maxDistanceBetweenPoints:
                points.append(newPoint)
                mesh = np.vstack((mesh,newPoint))
        
        return np.asarray(points)
 
    
def ND_Alpha_Shape(mesh, triangulation, alpha, dimension):
    # Del = Delaunay(mesh) # Form triangulation
    Del = triangulation
    radii = []
    for verts in Del.simplices:
        c, r = CS.circumsphere(mesh[verts])
        radii.append(r)
      
    r = np.asarray(radii)
    r = np.nan_to_num(r)

    tetras = Del.vertices[r<alpha,:]
    
    vals = np.asarray(list(range(0,dimension+1)))
    TriComb = np.asarray(list(combinations(vals, dimension)))
    
    Triangles = tetras[:,TriComb].reshape(-1,dimension)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    vals = np.asarray(list(range(0,dimension)))
    EdgeComb = np.asarray(list(combinations(vals, dimension-1)))
    
    Edges=Triangles[:,EdgeComb].reshape(-1,dimension-1)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)
    
    Vertices = np.unique(Edges)
    return Vertices

# https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
# def alpha_shape(points, triangulation, alpha, only_outer=True):
#     """
#     Compute the alpha shape (concave hull) of a set of points.
#     :param points: np.array of shape (n,2) points.
#     :param alpha: alpha value.
#     :param only_outer: boolean value to specify if we keep only the outer border
#     or also inner edges.
#     :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
#     the indices in the points array.
#     """
#     assert points.shape[0] > 3, "Need at least four points"
    
#     def add_edge(edges, i, j):
#         """
#         Add an edge between the i-th and j-th points,
#         if not in the list already
#         """
#         if (i, j) in edges or (j, i) in edges:
#             # already added
#             assert (j, i) in edges, "Can't go twice over same directed edge right?"
#             if only_outer:
#                 # if both neighboring triangles are in shape, it's not a boundary edge
#                 edges.remove((j, i))
#             return
#         edges.add((i, j))

#     tri = triangulation
#     edges = set()
#     # Loop over triangles:
#     # ia, ib, ic = indices of corner points of the triangle
#     for ia, ib, ic in tri.vertices:
#         pa = points[ia]
#         pb = points[ib]
#         pc = points[ic]
#         # Computing radius of triangle circumcircle
#         # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
#         a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
#         b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
#         c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
#         s = (a + b + c) / 2.0
#         val = s * (s - a) * (s - b) * (s - c)
#         if val <=0:
#             circum_r = float('nan')
#         else:
#             area = np.sqrt(s * (s - a) * (s - b) * (s - c))
#             circum_r = a * b * c / (4.0 * area)
#         if circum_r < alpha:
#             add_edge(edges, ia, ib)
#             add_edge(edges, ib, ic)
#             add_edge(edges, ic, ia)
            
#     # plt.figure()
#     # plt.plot(points[:, 0], points[:, 1], '.')
#     # for i, j in edges:
#     #     plt.plot(points[[i, j], 0], points[[i, j], 1], 'r')
#     # plt.show()
    
#     aa = list(chain(edges))
#     out = [item for t in aa for item in t]
#     pointsOnBoundary = np.sort(out)
#     pointsOnBoundary = pointsOnBoundary[1::2]  # Skip every other element to remove repeated elements
    
#     return pointsOnBoundary


# from collections import defaultdict

# def alpha_shape_3D(pos, tetra, alpha):
#     """
#     Compute the alpha shape (concave hull) of a set of 3D points.
#     Parameters:
#         pos - np.array of shape (n,3) points.
#         alpha - alpha value.
#     return
#         outer surface vertex indices, edge indices, and triangle indices
#     """

#     tetra = Delaunay(pos)
#     # Find radius of the circumsphere.
#     # By definition, radius of the sphere fitting inside the tetrahedral needs 
#     # to be smaller than alpha value
#     # http://mathworld.wolfram.com/Circumsphere.html
#     tetrapos = np.take(pos,tetra.vertices,axis=0)
#     normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
#     ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
#     a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
#     Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
#     Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
#     Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
#     c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
#     r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

#     # Find tetrahedrals
#     tetras = tetra.vertices[r<alpha,:]
#     # triangles
#     TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
#     Triangles = tetras[:,TriComb].reshape(-1,3)
#     Triangles = np.sort(Triangles,axis=1)
#     # Remove triangles that occurs twice, because they are within shapes
#     TrianglesDict = defaultdict(int)
#     for tri in Triangles:TrianglesDict[tuple(tri)] += 1
#     Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
#     #edges
#     EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
#     Edges=Triangles[:,EdgeComb].reshape(-1,2)
#     Edges=np.sort(Edges,axis=1)
#     Edges=np.unique(Edges,axis=0)

#     Vertices = np.unique(Edges)
#     return Vertices
#     # return Vertices,Edges,Triangles




import math
#https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))


    return np.asarray(points)
# points = fibonacci_sphere(20)
# points = np.asarray(points)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2])

