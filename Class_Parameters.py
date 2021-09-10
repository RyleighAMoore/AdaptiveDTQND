class Parameters:
    def __init__(self,sde, beta, radius, kstepMin, kstepMax, h, timeDiscretizationType = "EM"):
        self.conditionNumForAltMethod = 10
        self.h = h
        self.kstepMin = kstepMin
        self.kstepMax = kstepMax
        self.beta = beta
        self.radius = radius
        self.timeDiscretizationType = timeDiscretizationType
        self.setNumLejas(sde)
        self.setNumPointsForLejaCandidates(sde)
        self.setNumQuadFit(sde)

    def tuneOnSdeUnlessDefined(self, sde):
        self.numberOfLejaPoints = self.getOptimalNumberOfLejaPoints(sde)

    def setNumLejas(self, sde):
        if sde.dimension == 1:
            self.numLejas = 5
        elif sde.dimension == 2:
            self.numLejas = 10
        else:
            self.numLejas = 15

    def setNumPointsForLejaCandidates(self, sde):
        if sde.dimension == 1:
            self.numPointsForLejaCandidates = 50
        elif sde.dimension == 2:
            self.numPointsForLejaCandidates = 150
        else:
            self.numPointsForLejaCandidates = 150

    def setNumQuadFit(self,sde):
        if sde.dimension == 1:
            self.numQuadFit = 30
        elif sde.dimension == 2:
            self.numQuadFit = 150
        else:
            self.numQuadFit = 150






