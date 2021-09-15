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
import ICMeshGenerator as M

random.seed(10)

class MeshUpdater:
    def __init__(self, parameters, pdf,dimension):
        self.addPointsToBoundaryIfBiggerThanTolerance = 10**(-parameters.beta)
        self.removeZerosValuesIfLessThanTolerance = 10**(-parameters.beta-0.5)
        self.changedBoolean = False
        if not dimension == 1:
            self.triangulation = Delaunay(pdf.meshCoordinates, incremental=True)


    def addPointsToBoundary(self, pdf, sde, parameters, dimension, integrator):
            count = 0
            MeshOrig = np.copy(pdf.meshCoordinates)
            PdfOrig = np.copy(pdf.pdfVals)
            if sde.dimension == 1: # 1D
                left = np.argmin(pdf.meshCoordinates)
                right = np.argmax(pdf.meshCoordinates)
                newPointsBool = False
                newPoints = [] # Use as temporary mesh
                if pdf.pdfVals[left] > self.addPointsToBoundaryIfBiggerThanTolerance:
                    radius = parameters.minDistanceBetweenPoints/2 + parameters.maxDistanceBetweenPoints/2
                    mm = np.min(pdf.meshCoordinates)
                    MM = np.max(pdf.meshCoordinates)
                    newPointsBool = True

                    for i in range(1,5):
                        pdf.addPointsToMesh(np.asarray([[mm-i*radius]]))
                        newPoints.append(np.asarray(mm-i*radius))
                if pdf.pdfVals[right] > self.addPointsToBoundaryIfBiggerThanTolerance:
                    radius = parameters.minDistanceBetweenPoints/2 + parameters.maxDistanceBetweenPoints/2
                    mm = np.min(pdf.meshCoordinates)
                    MM = np.max(pdf.meshCoordinates)
                    newPointsBool = True

                    for i in range(1,5):
                        pdf.addPointsToMesh(np.asarray([[MM+i*radius]]))
                        newPoints.append(np.asarray(MM+i*radius))
                if newPointsBool:
                    interp1 = [griddata(MeshOrig,PdfOrig, np.asarray(newPoints), method='linear', fill_value=np.min(pdf.pdfVals))][0]
                    interp1[interp1<0] = np.min(PdfOrig)
                    pdf.addPointsToPdf(interp1)
                    self.changedBoolean =1

            else:
                while count < 1:
                    count = count + 1
                    numPointsAdded = 0
                    boundaryPointsToAddAround = self.checkIntegrandForAddingPointsAroundBoundaryPoints(pdf, dimension, parameters, sde)
                    iivals = np.expand_dims(np.arange(len(pdf.meshCoordinates)),1)
                    index = iivals[boundaryPointsToAddAround]
                    if len(index)>0:
                        candPoints = M.NDGridMesh(sde.dimension,parameters.maxDistanceBetweenPoints*2, parameters.maxDistanceBetweenPoints*2.5, UseNoise = False)
                    for indx in index:
                        # newPoints = addPointsRadially(Mesh[indx,:], Mesh, 8, minDistanceBetweenPoints, maxDistanceBetweenPoints)
                        curr =  np.repeat(np.expand_dims(pdf.meshCoordinates[indx,:],1), np.size(candPoints,0), axis=1)
                        newPoints = -candPoints  + curr.T
                        points = []
                        for i in range(len(newPoints)):
                            newPoint = newPoints[i,:]
                            if len(points)>0:
                                mesh2 = np.vstack((pdf.meshCoordinates,points))
                                nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, mesh2)
                            else:
                                nearestPoint,distToNearestPoint, idx = UM.findNearestPoint(newPoint, pdf.meshCoordinates)

                            if distToNearestPoint >= parameters.minDistanceBetweenPoints:
                                points.append(newPoint)

                        newPoints = points
                        if len(newPoints)>0:
                            pdf.addPointsToMesh(newPoints)
                            self.changedBoolean = 1
                            numPointsAdded = numPointsAdded + len(newPoints)
                    if numPointsAdded > 0:
                        newPoints = pdf.meshCoordinates[-numPointsAdded:,:]
                        interp = [griddata(MeshOrig, PdfOrig, newPoints, method='linear', fill_value=np.min(pdf.pdfVals))][0]
                        interp[interp<0] = np.min(pdf.pdfVals)
                        pdf.addPointsToPdf(interp)
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
                        simulation.timeDiscretizationMethod.AddPointToG(pdf.meshCoordinates[:i,:], i-1, parameters, sde, pdf, simulation.integrator)
            elif parameters.timeDiscretizationType == "AM":
                indices = list(range(meshSizeBeforeUpdates, newMeshSize))
                simulation.timeDiscretizationMethod.AddPointToG(pdf, indices, parameters, simulation.integrator, sde)


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

    def removePointsFromMeshProcedure(self, pdf, simulation, parameters, sde):
        '''# Removing boundary points'''
        ZeroPointsBoolArray = np.asarray([np.asarray(pdf.pdfVals) < self.removeZerosValuesIfLessThanTolerance]).T
        iivals = np.expand_dims(np.arange(pdf.meshLength),1)
        indices = iivals[ZeroPointsBoolArray] # Points to remove
        if len(indices)>0:
            pdf.removePointsFromMesh(indices)
            pdf.removePointsFromPdf(indices)
            simulation.integrator.removePoints(indices)
            simulation.integrator.houseKeepingStorageMatrices(indices)

            if not sde.dimension ==1:
                self.triangulation = Delaunay(pdf.meshCoordinates, incremental=True)




    def ND_Alpha_Shape(self, alpha, pdf, sde):
        # Del = Delaunay(mesh) # Form triangulation
        radii = []
        for verts in self.triangulation.simplices:
            c, r = CS.circumsphere(pdf.meshCoordinates[verts])
            radii.append(r)

        r = np.asarray(radii)
        r = np.nan_to_num(r)

        tetras = self.triangulation.vertices[r<alpha,:]

        vals = np.asarray(list(range(0,sde.dimension+1)))
        TriComb = np.asarray(list(combinations(vals, sde.dimension)))

        Triangles = tetras[:,TriComb].reshape(-1,sde.dimension)
        Triangles = np.sort(Triangles,axis=1)
        # Remove triangles that occurs twice, because they are within shapes
        TrianglesDict = defaultdict(int)
        for tri in Triangles:TrianglesDict[tuple(tri)] += 1
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

