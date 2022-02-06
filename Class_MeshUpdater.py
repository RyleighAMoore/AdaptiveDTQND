import numpy as np
import Functions as fun
from scipy.spatial import Delaunay
from itertools import chain
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LejaPoints as LP
import LejaPoints as LP
from scipy.interpolate import griddata
import random
import Circumsphere as CS
from itertools import combinations
from collections import defaultdict
import time

random.seed(10)

class MeshUpdater:
    def __init__(self, parameters, pdf,dimension):
        '''
        Used to update the mesh adaptively.

        Paramters:
        parameters: parameters defined by the user (class object)
        pdf: manages the probability density function of the solution of the SDE (PDF class object)
        dimension: dimension of the SDE
        '''
        self.changedBoolean = False
        self.addPointsToBoundaryIfBiggerThanTolerance = 10**(-parameters.beta)
        self.removeZerosValuesIfLessThanTolerance = 10**(-parameters.beta-0.5)
        if not dimension == 1:
            self.triangulation = Delaunay(pdf.meshCoordinates, incremental=True)


    def addPointsToBoundary(self, pdf, sde, parameters, dimension, integrator):
            numPointsAdded = 0
            if sde.dimension == 1: # 1D, add points to left and right is easy
                left = np.argmin(pdf.meshCoordinates)
                right = np.argmax(pdf.meshCoordinates)

                newPoints = [] # Use as temporary mesh
                numIters = int(parameters.h*10*20)
                if pdf.pdfVals[left] > self.addPointsToBoundaryIfBiggerThanTolerance:
                    radius = parameters.maxDistanceBetweenPoints
                    for i in range(1,numIters):
                        pdf.addPointsToMesh(np.asarray([[left-i*radius]]))

                if pdf.pdfVals[right] > self.addPointsToBoundaryIfBiggerThanTolerance:
                    radius = parameters.maxDistanceBetweenPoints

                    for i in range(1,numIters):
                        pdf.addPointsToMesh(np.asarray([[right+i*radius]]))

                if numPointsAdded > 0:
                    # pdflog = np.log(PdfOrig)
                    # interp = [griddata(MeshOrig, pdflog, np.asarray(newPoints), method='linear', fill_value=np.min(pdflog))][0]
                    # interp = np.exp(interp)
                    #interp1[interp1<=0] = 0.5*np.min(PdfOrig)
                    interp = np.ones((1,numPointsAdded))*np.min(pdf.pdfVals)
                    pdf.addPointsToPdf(interp)
                    self.changedBoolean = True
            else:
                boundaryPointsToAddAround = self.checkIntegrandForAddingPointsAroundBoundaryPoints(pdf, dimension, parameters, sde)
                iivals = np.expand_dims(np.arange(len(pdf.meshCoordinates)),1)
                index = iivals[boundaryPointsToAddAround]
                if len(index)>0:
                    candPoints = fun.nDGridMeshCenteredAtOrigin(sde.dimension, parameters.maxDistanceBetweenPoints, parameters.maxDistanceBetweenPoints, useNoiseBool = False, trimToCircle=True)
                for indx in index:
                    curr =  np.repeat(np.expand_dims(pdf.meshCoordinates[indx,:],1), np.size(candPoints,0), axis=1)
                    candPointsForIndx = -candPoints + curr.T
                    for i in range(len(candPointsForIndx)):
                        candPoint = candPointsForIndx[i,:]
                        nearestPoint,distToNearestPoint, idx = fun.findNearestPoint(candPoint, pdf.meshCoordinates)
                        if distToNearestPoint >= 0.99*parameters.minDistanceBetweenPoints and distToNearestPoint <= 1.01*parameters.maxDistanceBetweenPoints:
                            pdf.addPointsToMesh(np.expand_dims(candPoint,0))
                            numPointsAdded +=1

                if numPointsAdded > 0:
                    # newPoints = pdf.meshCoordinates[-numPointsAdded:,:]
                    # interp = [griddata(MeshOrig, np.log(PdfOrig), pdf.meshCoordinates[-pointsAdded:], method='linear', fill_value=np.log(np.min(pdf.pdfVals)))][0]
                    # interp = np.exp(interp)
                    # interp[interp<=0] = np.min(pdf.pdfVals)/2
                    interp = np.ones((1,numPointsAdded))*np.min(pdf.pdfVals)
                    pdf.addPointsToPdf(interp)
                    self.changedBoolean = True
                    self.triangulation = Delaunay(pdf.meshCoordinates, incremental=True)

    def addPointsToMeshProcedure(self, pdf, parameters, simulation, sde):
        '''If the mesh is changed, these become 1 so we know to recompute the triangulation'''
        self.changedBoolean = 0
        meshSizeBeforeUpdates = pdf.meshLength
        self.addPointsToBoundary(pdf, sde, parameters, sde.dimension, simulation.integrator)
        if self.changedBoolean==1:
            newMeshSize = len(pdf.meshCoordinates)
            if parameters.timeDiscretizationType == "EM":
                for i in range(meshSizeBeforeUpdates+1, newMeshSize+1):
                        simulation.timeDiscretizationMethod.AddPointToG(pdf.meshCoordinates[:i,:], i-1, parameters, sde, pdf, simulation.integrator, simulation)
            elif parameters.timeDiscretizationType == "AM":
                indices = list(range(meshSizeBeforeUpdates, newMeshSize))
                simulation.timeDiscretizationMethod.AddPointToG(simulation, indices, parameters, simulation.integrator, sde)

    def getBoundaryPoints(self, pdf, dimension, parameters, sde):
        if dimension ==1:
            pointsOnBoundary = ([np.min(pdf.meshVals), np.max(pdf.meshVals)])
        else:
            pointsOnBoundary = self.ND_Alpha_Shape(parameters.maxDistanceBetweenPoints*1.5, pdf, sde)
        return pointsOnBoundary

    def checkIntegrandForAddingPointsAroundBoundaryPoints(self, pdf, dimension, parameters, sde):
        '''Check if the points on the edge are too big and we need more points around them
        Uses alpha hull to get the boundary points if boundaryOnly is True.'''
        valueList = -1*np.ones(pdf.meshLength) # Set to small values for placeholder
        pointsOnEdge = self.getBoundaryPoints(pdf, dimension, parameters, sde)
        for i in pointsOnEdge:
            valueList[i]=pdf.pdfVals[i]
        addingAround = [np.asarray(valueList) >= self.addPointsToBoundaryIfBiggerThanTolerance]
        return np.asarray(addingAround).T


    def removeOutlierPoints(self, pdf, simulation, parameters, sde):
        pointIndicesToRemove = []
        for indx in reversed(range(len(pdf.meshCoordinates))):
            point = pdf.meshCoordinates[indx]
            nearestPoint,distToNearestPoint, idx = fun.findNearestPoint(point, pdf.meshCoordinates, CoordInAllPoints=True)
            # print(distToNearestPoint)
            if distToNearestPoint > 1.1*parameters.maxDistanceBetweenPoints:
                pointIndicesToRemove.append(indx)
                # print(distToNearestPoint)
        if len(pointIndicesToRemove)>0:
            pdf.removePointsFromMesh(pointIndicesToRemove)
            pdf.removePointsFromPdf(pointIndicesToRemove)
            simulation.removePoints(pointIndicesToRemove)
            simulation.houseKeepingStorageMatrices(pointIndicesToRemove)
            # print("removed", len(pointIndicesToRemove), "outlier(s)")
            if not sde.dimension ==1:
                self.triangulation = Delaunay(pdf.meshCoordinates, incremental=True)


    def removePointsFromMeshProcedure(self, pdf, simulation, parameters, sde):
        '''# Removing boundary points'''
        ZeroPointsBoolArray = np.asarray([np.asarray(pdf.pdfVals) < self.removeZerosValuesIfLessThanTolerance]).T
        iivals = np.expand_dims(np.arange(pdf.meshLength),1)
        indices = iivals[ZeroPointsBoolArray] # Points to remove
        if len(indices)>0:
            pdf.removePointsFromMesh(indices)
            pdf.removePointsFromPdf(indices)
            simulation.removePoints(indices)
            simulation.houseKeepingStorageMatrices(indices)

            if not sde.dimension ==1:
                self.triangulation = Delaunay(pdf.meshCoordinates, incremental=True)


    #Adapted from code here: https://stackoverflow.com/questions/64271678/3d-alpha-shape-for-finding-the-boundary-of-points-cloud-in-python
    def ND_Alpha_Shape(self, alpha, pdf, sde):
        # Del = Delaunay(mesh) # Form triangulation
        radii = []
        for verts in self.triangulation.simplices:
            c, r = CS.circumsphere(pdf.meshCoordinates[verts])
            radii.append(r)

        r = np.asarray(radii)
        r = np.nan_to_num(r)

        #List of all vertices associated to triangles where circumshpere r<alpha
        tetras = self.triangulation.vertices[r<alpha,:]

        vals = np.asarray(list(range(0,sde.dimension+1)))

        TriComb = np.asarray(list(combinations(vals, sde.dimension)))

        #List of all combinatations of edges that can make up the triangle
        Triangles = tetras[:,TriComb].reshape(-1,sde.dimension)
        Triangles = np.sort(Triangles,axis=1)

        # Remove triangles that occurs twice, because they are within shapes
        TrianglesDict = defaultdict(int)
        for tri in Triangles:
            TrianglesDict[tuple(tri)] += 1

        Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])

        #edges
        vals = np.asarray(list(range(0,sde.dimension)))
        EdgeComb = np.asarray(list(combinations(vals, sde.dimension-1)))

        Edges=Triangles[:,EdgeComb].reshape(-1,sde.dimension-1)
        Edges=np.sort(Edges,axis=1)
        Edges=np.unique(Edges,axis=0)

        Vertices = np.unique(Edges)
        return Vertices


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

