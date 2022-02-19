import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import random
from itertools import combinations
from collections import defaultdict

import Circumsphere as CS
import Functions as fun


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
                leftVal = pdf.meshCoordinates[left][0]
                rightVal = pdf.meshCoordinates[right][0]

                numIters = int(parameters.h*10*20)
                if pdf.pdfVals[left] > self.addPointsToBoundaryIfBiggerThanTolerance:
                    radius = parameters.maxDistanceBetweenPoints
                    for i in range(1,numIters):
                        pdf.addPointsToMesh(np.asarray([[leftVal-i*radius]]))
                        numPointsAdded +=1


                if pdf.pdfVals[right] > self.addPointsToBoundaryIfBiggerThanTolerance:
                    radius = parameters.maxDistanceBetweenPoints

                    for i in range(1,numIters):
                        pdf.addPointsToMesh(np.asarray([[rightVal+i*radius]]))
                        numPointsAdded +=1

                if numPointsAdded > 0:
                    # pdflog = np.log(PdfOrig)
                    # interp = [griddata(MeshOrig, pdflog, np.asarray(newPoints), method='linear', fill_value=np.min(pdflog))][0]
                    # interp = np.exp(interp)
                    #interp1[interp1<=0] = 0.5*np.min(PdfOrig)
                    interp = np.ones((1,numPointsAdded))*pdf.minPdfValue
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
                    interp = np.ones((1,numPointsAdded))*pdf.minPdfValue
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
        '''Currently checks for single straggling points to remove, could be neat to
        look for straggling clusters to remove in future verstions
        '''
        pointIndicesToRemove = []
        for indx in reversed(range(len(pdf.meshCoordinates))):
            point = pdf.meshCoordinates[indx]
            nearestPoint,distToNearestPoint, idx = fun.findNearestPoint(point, pdf.meshCoordinates, CoordInAllPoints=True)
            if distToNearestPoint > 1.1*parameters.maxDistanceBetweenPoints:
                pointIndicesToRemove.append(indx)
        if len(pointIndicesToRemove)>0:
            pdf.removePointsFromMesh(pointIndicesToRemove)
            pdf.removePointsFromPdf(pointIndicesToRemove)
            simulation.removePoints(pointIndicesToRemove)
            simulation.houseKeepingStorageMatrices(pointIndicesToRemove)
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
        radii = []
        for verts in self.triangulation.simplices:
            c, r = CS.circumsphere(pdf.meshCoordinates[verts])
            radii.append(r)

        r = np.asarray(radii)
        r = np.nan_to_num(r)

        '''List of all vertices associated to triangles where circumshpere r<alpha'''
        tetras = self.triangulation.vertices[r<alpha,:]

        vals = np.asarray(list(range(0,sde.dimension+1)))

        TriComb = np.asarray(list(combinations(vals, sde.dimension)))

        '''List of all combinatations of edges that can make up the triangle'''
        Triangles = tetras[:,TriComb].reshape(-1,sde.dimension)
        Triangles = np.sort(Triangles,axis=1)

        '''Remove triangles that occur twice, because they are within shapes'''
        TrianglesDict = defaultdict(int)
        for tri in Triangles:
            TrianglesDict[tuple(tri)] += 1

        Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])

        '''Get boundary edges and corresponding vertices'''
        vals = np.asarray(list(range(0,sde.dimension)))
        EdgeComb = np.asarray(list(combinations(vals, sde.dimension-1)))

        Edges=Triangles[:,EdgeComb].reshape(-1,sde.dimension-1)
        Edges=np.sort(Edges,axis=1)
        Edges=np.unique(Edges,axis=0)

        Vertices = np.unique(Edges)
        return Vertices


