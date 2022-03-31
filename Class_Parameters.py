class Parameters:
    def __init__(self, sde, beta, radius, kstepMin, kstepMax, h, useAdaptiveMesh, timeDiscretizationType = "EM", integratorType = "LQ", AMSpacing = 0.05, initialMeshCentering=None, OverideMesh = None, saveHistory = True):
        '''
        Manages parameters for the simulation for approximating the solution of the stochastic differential equation

        Parameters:
        sde: stochastic differential equation to solve (class object)
        beta: cutoff parameter used to adaptively change mesh in the MeshUpdater class
        radius: span of initial mesh
        kstepMin: spacing of initial mesh, also used to enforce min distance between new points when aadded
        kstepMax: used to enforce max distance between new points when aadded
        h: temportal step size
        useAdaptiveMesh: boolean to determine if the mesh should be adaptively updated (True) or static (False)
        timeDiscretizationType="EM": Either EM for Euler-Maruyama or AM for Anderson-Mattingly
        integratorType="LQ": Either TR for trapezoidal rule or LQ for Leja quadrature
        AMSpacing = 0.05: Spacing used for computing inegral in Anderson-Mattingly procedure
        initialMeshCentering=None: Used to adjust the mesh centering if not centered at origin
        OverideMesh: Used to overide default mesh with another mesh provided by the user, if None the proceudre will compute a mesh
        saveHistory: Determine if we should save all steps or only final (and possibly initial).
        '''

        self.conditionNumForAltMethod = 5
        self.conditionNumberForAcceptingLejaPointsAtNextTimeStep = 1.1
        self.h = h
        self.kstepMin = kstepMin
        self.minDistanceBetweenPoints = kstepMin
        self.kstepMax = kstepMax
        self.maxDistanceBetweenPoints = kstepMax
        self.beta = beta
        self.radius = radius
        self.initialMeshCentering = initialMeshCentering
        self.timeDiscretizationType = timeDiscretizationType
        self.setNumLejas(sde)
        self.setNumPointsForLejaCandidates(sde)
        self.setNumQuadFit(sde)
        self.useAdaptiveMesh = useAdaptiveMesh
        self.integratorType = integratorType
        self.AMMeshSpacing = AMSpacing
        self.OverideMesh = OverideMesh
        self.saveHistory = saveHistory
        self.eligibleToAddPointsTimeStep = 1
        self.eligibleToRemovePointsTimeStep = 10
        self.addPointsEveryNSteps = 1
        self.removePointsEveryNSteps = 10
        self.matrixSizeMultiple =10

        if timeDiscretizationType == "AM":
            print("Warning: The Anderson-Mattingly method is still under development and may not be ready for use. Please consider using Euler-Maruyama instead.")


    def setNumLejas(self, sde):
        if sde.dimension == 1:
            self.numLejas = 6
        elif sde.dimension == 2:
            self.numLejas = 10
        elif sde. dimension ==3:
            self.numLejas = 15
        elif sde.dimension ==4:
            self.numLejas = 15
        elif sde.dimension ==5:
            self.numLejas = 40

    def setNumPointsForLejaCandidates(self, sde):
        if sde.dimension == 1:
            self.numPointsForLejaCandidates = 50
        elif sde.dimension == 2:
            self.numPointsForLejaCandidates = 150
        elif sde.dimension == 3:
            self.numPointsForLejaCandidates = 150
        elif sde.dimension==4:
            self.numPointsForLejaCandidates = 250
        elif sde.dimension ==5:
            self.numPointsForLejaCandidates = 450

    def setNumQuadFit(self,sde):
        if sde.dimension == 1:
            self.numQuadFit = 20
        elif sde.dimension == 2:
            self.numQuadFit = 20
        elif sde.dimension==3:
            self.numQuadFit = 150
        elif sde.dimension == 4:
            self.numQuadFit = 200
        elif sde.dimension == 5:
            self.numQuadFit = 300






