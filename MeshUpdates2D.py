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
random.seed(10)


def addPointsToMeshProcedure(Mesh, Pdf, triangulation, kstep, h, poly, GMat, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints,drift, diff, SpatialDiff):
    '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
    changedBool2 = 0 
    changedBool1 = 0
    meshSize = len(Mesh)
    Mesh, Pdf, triangulation, changedBool1 = addPointsToBoundary(Mesh, Pdf, triangulation,addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints)
    ChangedBool = max(changedBool1, changedBool2)
    if ChangedBool==1:
        newMeshSize = len(Mesh)
        for i in range(meshSize+1, newMeshSize+1):
            GMat = fun.AddPointToG(Mesh[:i,:], i-1, h, GMat, drift, diff, SpatialDiff)
    return Mesh, Pdf, triangulation, ChangedBool, GMat

def removePointsFromMeshProcedure(Mesh, Pdf, tri, boundaryOnlyBool, poly, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance):
    '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
    Mesh, Pdf, GMat,LPMat, LPMatBool, tri = removeSmallPoints(Mesh, Pdf, tri, boundaryOnlyBool, GMat, LPMat, LPMatBool, removeZerosValuesIfLessThanTolerance)
    return Mesh, Pdf, GMat, LPMat, LPMatBool, tri


def getBoundaryPoints(Mesh, tri, alpha):
    if np.size(Mesh,1) ==1:
        pointsOnBoundary = ([np.min(Mesh), np.max(Mesh)])
    else:
        '''Uses triangulation and alpha hull technique to find boundary points'''
        edges = alpha_shape(Mesh, tri, alpha, only_outer=True)
        aa = list(chain(edges))
        out = [item for t in aa for item in t]
        pointsOnBoundary = np.sort(out)
        pointsOnBoundary = pointsOnBoundary[1::2]  # Skip every other element to remove repeated elements
    
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

    
def addPointsToBoundary(Mesh, Pdf, triangulation, addPointsToBoundaryIfBiggerThanTolerance, removeZerosValuesIfLessThanTolerance, minDistanceBetweenPoints,maxDistanceBetweenPoints):
    ChangedBool = 0
    count = 0
    MeshOrig = np.copy(Mesh)
    PdfOrig = np.copy(Pdf)
    if np.size(Mesh,1) == 1: # 1D
        radius = minDistanceBetweenPoints/2 + maxDistanceBetweenPoints/2
        newPoints = []
        mm = np.min(Mesh)
        MM = np.max(Mesh)
        for i in range(1,4):
            Mesh = np.append(Mesh, np.asarray([[mm-i*radius]]), axis=0)
            newPoints.append(np.asarray(mm-i*radius))
            Mesh = np.append(Mesh, np.asarray([[MM+i*radius]]), axis=0)
            newPoints.append(np.asarray(MM+i*radius))
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
            for indx in index:
                newPoints = addPointsRadially(Mesh[indx,0], Mesh[indx,1], Mesh, 8, minDistanceBetweenPoints, maxDistanceBetweenPoints)
                if len(newPoints)>0:
                    Mesh = np.append(Mesh, newPoints, axis=0)
                    ChangedBool = 1
                    numPointsAdded = numPointsAdded + len(newPoints)
            if numPointsAdded > 0:
                newPoints = Mesh[-numPointsAdded:,:]
                interp = [griddata(MeshOrig,PdfOrig, newPoints, method='linear', fill_value=np.min(Pdf))][0]
                interp[interp<0] = np.min(Pdf)
                # interp = np.ones(len(newPoints))*removeZerosValuesIfLessThanTolerance 
                # interp = np.ones(len(newPoints))*10**(-8)
                Pdf = np.append(Pdf, interp)
                triangulation = houseKeepingAfterAdjustingMesh(Mesh, triangulation)
    return Mesh, Pdf, triangulation, ChangedBool


def addPointsRadially(pointX, pointY, mesh, numPointsToAdd, minDistanceBetweenPoints, maxDistanceBetweenPoints):
    radius = minDistanceBetweenPoints/2 + maxDistanceBetweenPoints/2
    points = [] 
    noise = random.uniform(0, 1)*2*np.pi
    dTheta = 2*np.pi/numPointsToAdd
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
    return np.asarray(points)
    

# https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
def alpha_shape(points, triangulation, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"
    
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = triangulation
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        val = s * (s - a) * (s - b) * (s - c)
        if val <=0:
            circum_r = float('nan')
        else:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
            
    # plt.figure()
    # plt.plot(points[:, 0], points[:, 1], '.')
    # for i, j in edges:
    #     plt.plot(points[[i, j], 0], points[[i, j], 1], 'r')
    # plt.show()
    return edges
