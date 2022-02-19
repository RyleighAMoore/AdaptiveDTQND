



    # def computeTransitionMatrix1(self, pdf, sde, parameters):
    #     self.meshSpacingAM = 0.1
    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     for j in trange(pdf.meshLength):
    #         mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
    #         N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])

    #         product = N1

    #         self.integrator.laplaceApproximation.computeleastSquares(self.meshAM, product, sde.dimension)
    #         # print(self.integrator.laplaceApproximation.scalingForGaussian.cov)
    #         if self.integrator.laplaceApproximation.scalingForGaussian == None:
    #             value = 0
    #             condNumber = 1
    #         else:
    #             self.meshAM = map_from_canonical_space(self.integrator.altMethodLejaPoints, self.integrator.laplaceApproximation.scalingForGaussian)
    #             N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
    #             N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])
    #             # plt.scatter(self.meshAM, N2[i,:]*N1)
    #             for i in range(pdf.meshLength):
    #                 pdf.setIntegrandBeforeDividingOut(N2[i,:]*N1)

    #                 vals = self.integrator.laplaceApproximation.ComputeDividedOutAM(pdf, sde.dimension, self.meshAM)
    #                 pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
    #                 V = opolynd.opolynd_eval(self.integrator.altMethodLejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
    #                 vinv = np.linalg.inv(V)
    #                 if sde.dimension > 1:
    #                     L = np.linalg.cholesky((scale1.cov))
    #                     JacFactor = np.prod(np.diag(L))
    #                 if sde.dimension ==1:
    #                     L = np.sqrt(scale1.cov)
    #                     JacFactor = np.squeeze(L)

    #                 value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut)
    #                 condNumber = np.sum(np.abs(vinv[0,:]))
    #                 # print(condNumber)
    #                 matrix[i,j] = value
    #     # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
    #     return matrix

    # def computeTransitionMatrixE(self, pdf, sde, parameters):
    #     self.meshSpacingAM = 0.2

    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     for j in trange(pdf.meshLength):
    #         mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
    #         N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])

    #         for i in range(pdf.meshLength):
    #             # plt.scatter(self.meshAM, N2[i,:]*N1)
    #             product = N2[i,:]*N1
    #             self.integrator.laplaceApproximation.computeleastSquares(self.meshAM, product, sde.dimension)
    #             # print(self.integrator.laplaceApproximation.scalingForGaussian.mu)
    #             if self.integrator.laplaceApproximation.scalingForGaussian == None:
    #                 value = 0
    #                 condNumber = 1
    #             else:
    #                 meshAMr = map_from_canonical_space(self.integrator.altMethodLejaPoints, self.integrator.laplaceApproximation.scalingForGaussian)
    #                 N1r = scale1.ComputeGaussian(meshAMr, sde.dimension)
    #                 N2r = self.computeN2Row(pdf, sde, parameters.h, np.expand_dims(pdf.meshCoordinates[i],1).T, meshAMr)
    #                 # N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])

    #                 # plt.scatter(self.meshAM, N2[i,:]*N1)

    #                 pdf.setIntegrandBeforeDividingOut(N2r.T*N1r)

    #                 vals = self.integrator.laplaceApproximation.ComputeDividedOutAM(pdf, sde.dimension, meshAMr)
    #                 pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
    #                 V = opolynd.opolynd_eval(self.integrator.altMethodLejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
    #                 vinv = np.linalg.inv(V)
    #                 if sde.dimension > 1:
    #                     L = np.linalg.cholesky((scale1.cov))
    #                     JacFactor = np.prod(np.diag(L))
    #                 if sde.dimension ==1:
    #                     L = np.sqrt(scale1.cov)
    #                     JacFactor = np.squeeze(L)

    #                 value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut.T)
    #                 condNumber = np.sum(np.abs(vinv[0,:]))
    #                 # print(condNumber)
    #             matrix[i,j] = value
    #     # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
    #     return matrix


    # def computeTransitionMatrix1(self, pdf, sde, parameters):
    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     for j in trange(pdf.meshLength):
    #         mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
    #         N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)

    #         val = self.meshSpacingAM**sde.dimension*N2@np.expand_dims(N1,1)

    #         # self.integrator.laplaceApproximation.copmuteleastSquares(self.meshCoordinates, val, sde.dimension)
    #         # print(self.integrator.laplaceApproximation.scalingForGaussian)
    #         # fig = pyplot.figure()
    #         # ax = Axes3D(fig)
    #         # ax.scatter(self.meshAM[:,0], self.meshAM[:,1], N1)
    #         # # ax.scatter(pdf.meshCoordinates[:,0], pdf.meshCoordinates[:,1], val)

    #         # pyplot.show()
    #         matrix[:len(val),j] = np.squeeze(val)
    #     return matrix

    # # @profile
    # def computeTransitionMatrix1(self, pdf, sde, parameters):
    #     self.meshSpacingAM = 0.1
    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     meshDTQ = np.copy(pdf.meshCoordinates)
    #     for j in trange(len(meshDTQ)):
    #         pdf.meshCoordinates = meshDTQ
    #         mu1= meshDTQ[j]+sde.driftFunction(np.asarray([meshDTQ[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([meshDTQ[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
    #         N2 = self.computeN2(pdf, sde, parameters.h, meshDTQ[j])

    #         for i in range(len(meshDTQ)):
    #             # plt.scatter(self.meshAM, N2[i,:]*N1)
    #             product = N2[i,:]*N1
    #             self.integrator.laplaceApproximation.computeleastSquares(self.meshAM, product, sde.dimension)
    #             # print(self.integrator.laplaceApproximation.scalingForGaussian.mu)
    #             if self.integrator.laplaceApproximation.scalingForGaussian == None:
    #                 value = 0
    #                 condNumber = 1
    #             else:

    #                 mappedMesh = map_to_canonical_space(self.meshAM, self.integrator.laplaceApproximation.scalingForGaussian)
    #                 self.lejaPoints, self.lejaPointsPdfVals, self.indicesOfLejaPoints,self.lejaSuccess = LP.getLejaSetFromPoints(self.integrator.identityScaling, mappedMesh, parameters.numLejas, self.integrator.poly, pdf.pdfVals, sde.diffusionFunction, parameters.numPointsForLejaCandidates)

    #                 if self.lejaSuccess ==False: # Failed to get Leja points
    #                     self.lejaPoints = None
    #                     self.lejaPointsPdfVals = None
    #                     self.idicesOfLejaPoints = None
    #                     self.freshLejaPoints = True
    #                     value = 0
    #                     condNumber = 1
    #                 else:
    #                     mappedLejas = map_from_canonical_space(self.lejaPoints, self.integrator.laplaceApproximation.scalingForGaussian)


    #                     N2LP = N2[i,self.indicesOfLejaPoints]
    #                     N1LP = N1[self.indicesOfLejaPoints]
    #                     pdf.setIntegrandBeforeDividingOut(N2LP.T*N1LP)

    #                     pdf.meshCoordinates = mappedLejas
    #                     vals = self.integrator.laplaceApproximation.ComputeDividedOut(pdf, sde.dimension)

    #                     # values = self.integrator.laplaceApproximation.ComputeDividedOutAM(pdf, sde.dimension, self.lejaPoints)
    #                     pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
    #                     V = opolynd.opolynd_eval(self.integrator.altMethodLejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
    #                     vinv = np.linalg.inv(V)
    #                     if sde.dimension > 1:
    #                         L = np.linalg.cholesky((scale1.cov))
    #                         JacFactor = np.prod(np.diag(L))
    #                     if sde.dimension ==1:
    #                         L = np.sqrt(scale1.cov)
    #                         JacFactor = np.squeeze(L)

    #                     value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut.T)
    #                     condNumber = np.sum(np.abs(vinv[0,:]))
    #                     # print(condNumber)
    #             matrix[i,j] = value
    #     # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
    #     pdf.meshCoordinates = meshDTQ
    #     return matrix

    # # @profile
    # def computeTransitionMatrix1(self, pdf, sde, parameters):
    #     self.meshSpacingAM = 0.05
    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     meshDTQ = np.copy(pdf.meshCoordinates)
    #     for j in trange(len(meshDTQ)):
    #         pdf.meshCoordinates = meshDTQ
    #         mu1= meshDTQ[j]+sde.driftFunction(np.asarray([meshDTQ[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([meshDTQ[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
    #         N2 = self.computeN2(pdf, sde, parameters.h, meshDTQ[j])

    #         for i in range(len(meshDTQ)):
    #             # plt.scatter(self.meshAM, N2[i,:]*N1)
    #             product = N2[i,:]*N1
    #             self.integrator.laplaceApproximation.computeleastSquares(self.meshAM, product, sde.dimension)
    #             # print(self.integrator.laplaceApproximation.scalingForGaussian.mu)
    #             if self.integrator.laplaceApproximation.scalingForGaussian == None:
    #                 value = 0
    #                 condNumber = 1
    #             else:

    #                 # mappedMesh = map_to_canonical_space(self.meshAM, self.integrator.laplaceApproximation.scalingForGaussian)
    #                 # self.lejaPoints, self.lejaPointsPdfVals, self.indicesOfLejaPoints,self.lejaSuccess = LP.getLejaSetFromPoints(self.integrator.identityScaling, mappedMesh, parameters.numLejas, self.integrator.poly, pdf.pdfVals, sde.diffusionFunction, parameters.numPointsForLejaCandidates)


    #                 # if self.lejaSuccess ==False: # Failed to get Leja points
    #                 #     self.lejaPoints = None
    #                 #     self.lejaPointsPdfVals = None
    #                 #     self.idicesOfLejaPoints = None
    #                 #     self.freshLejaPoints = True
    #                 #     value = 0
    #                 #     condNumber = 1
    #                 # else:
    #                 #     mappedLejas = map_from_canonical_space(self.lejaPoints, self.integrator.laplaceApproximation.scalingForGaussian)

    #                     self.lejaPoints, distances, self.indicesOfLejaPoints = findNearestKPoints(self.integrator.laplaceApproximation.scalingForGaussian.mu, self.meshAM, parameters.numLejas, getIndices = True)


    #                     N2LP = N2[i,self.indicesOfLejaPoints]
    #                     N1LP = N1[self.indicesOfLejaPoints]
    #                     pdf.setIntegrandBeforeDividingOut(N2LP.T*N1LP)

    #                     pdf.meshCoordinates = self.lejaPoints
    #                     vals = self.integrator.laplaceApproximation.ComputeDividedOut(pdf, sde.dimension)

    #                     pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
    #                     V = opolynd.opolynd_eval(self.integrator.altMethodLejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
    #                     vinv = np.linalg.inv(V)
    #                     if sde.dimension > 1:
    #                         L = np.linalg.cholesky((scale1.cov))
    #                         JacFactor = np.prod(np.diag(L))
    #                     if sde.dimension ==1:
    #                         L = np.sqrt(scale1.cov)
    #                         JacFactor = np.squeeze(L)

    #                     value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut.T)
    #                     condNumber = np.sum(np.abs(vinv[0,:]))
    #                     # print(condNumber)
    #             matrix[i,j] = value
    #     # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
    #     pdf.meshCoordinates = meshDTQ
    #     return matrix

    # # @profile
    # def computeTransitionMatrix(self, pdf, sde, parameters):
    #     self.meshSpacingAM = 0.05
    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     meshDTQ = np.copy(pdf.meshCoordinates)
    #     for j in trange(len(meshDTQ)):
    #         pdf.meshCoordinates = meshDTQ
    #         mu1= meshDTQ[j]+sde.driftFunction(np.asarray([meshDTQ[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([meshDTQ[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1), Noise = True)
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)
    #         N2 = self.computeN2(pdf, sde, parameters.h, meshDTQ[j])

    #         for i in range(len(meshDTQ)):
    #             # plt.scatter(self.meshAM, N2[i,:]*N1)
    #             product = N2[i,:]*N1
    #             meanEst = meshDTQ[i,:]/2 + meshDTQ[j,:]/2
    #             self.lejaPoints, distances, self.indicesOfLejaPoints = findNearestKPoints(meanEst, self.meshAM, parameters.numLejas, getIndices = True)

    #             self.integrator.laplaceApproximation.computeleastSquares(self.lejaPoints, product[self.indicesOfLejaPoints], sde.dimension)
    #             # print(self.integrator.laplaceApproximation.scalingForGaussian.mu)
    #             if self.integrator.laplaceApproximation.scalingForGaussian == None:
    #                 value = 0
    #                 condNumber = 1
    #             else:
    #                 # self.lejaPoints, distances, self.indicesOfLejaPoints = findNearestKPoints(meanEst, self.meshAM, parameters.numLejas, getIndices = True)


    #                 N2LP = N2[i,self.indicesOfLejaPoints]
    #                 N1LP = N1[self.indicesOfLejaPoints]
    #                 pdf.setIntegrandBeforeDividingOut(N2LP.T*N1LP)

    #                 pdf.meshCoordinates = self.lejaPoints
    #                 vals = self.integrator.laplaceApproximation.ComputeDividedOut(pdf, sde.dimension)

    #                 pdf.integrandAfterDividingOut = pdf.integrandBeforeDividingOut/vals
    #                 V = opolynd.opolynd_eval(self.lejaPoints, self.integrator.poly.lambdas[:parameters.numLejas,:], self.integrator.poly.ab, self.integrator.poly)
    #                 vinv = np.linalg.inv(V)
    #                 if sde.dimension > 1:
    #                     L = np.linalg.cholesky((scale1.cov))
    #                     JacFactor = np.prod(np.diag(L))
    #                 if sde.dimension ==1:
    #                     L = np.sqrt(scale1.cov)
    #                     JacFactor = np.squeeze(L)

    #                 value = np.matmul(vinv[0,:], pdf.integrandAfterDividingOut.T)
    #                 condNumber = np.sum(np.abs(vinv[0,:]))
    #                 # print(condNumber)
    #             matrix[i,j] = value
    #     # matrix2 = self.computeTransitionMatrix2(pdf, sde, parameters)
    #     pdf.meshCoordinates = meshDTQ
    #     return matrix


    # def computeTransitionMatrix(self, pdf, sde, parameters):
    #     matrix = np.empty([self.sizeTransitionMatrixIncludingEmpty, self.sizeTransitionMatrixIncludingEmpty])*np.NaN
    #     for j in trange(pdf.meshLength):
    #         mu1= pdf.meshCoordinates[j]+sde.driftFunction(np.asarray([pdf.meshCoordinates[j]]))*self.theta*parameters.h
    #         sig1 = abs(sde.diffusionFunction(np.asarray([pdf.meshCoordinates[j]]))*np.sqrt(self.theta*parameters.h))
    #         scale1 = GaussScale(sde.dimension)
    #         scale1.setMu(np.asarray(mu1.T))
    #         scale1.setCov(np.asarray(sig1**2))

    #         self.setAndersonMattinglyMeshAroundPoint(mu1, sde, np.max(sig1))
    #         N2 = self.computeN2(pdf, sde, parameters.h, pdf.meshCoordinates[j])
    #         N1 = scale1.ComputeGaussian(self.meshAM, sde.dimension)

    #         val = self.meshSpacingAM**sde.dimension*N2@np.expand_dims(N1,1)

    #         # self.integrator.laplaceApproximation.copmuteleastSquares(self.meshCoordinates, val, sde.dimension)
    #         # print(self.integrator.laplaceApproximation.scalingForGaussian)
    #         # fig = pyplot.figure()
    #         # ax = Axes3D(fig)
    #         # ax.scatter(self.meshAM[:,0], self.meshAM[:,1], N1)
    #         # # ax.scatter(pdf.meshCoordinates[:,0], pdf.meshCoordinates[:,1], val)

    #         # pyplot.show()
    #         matrix[:len(val),j] = np.squeeze(val)
    #     return matrix


