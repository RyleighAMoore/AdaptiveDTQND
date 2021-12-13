# from Class_Parameters import Parameters
# from Class_PDF import PDF
# from Class_SDE import SDE
# from Class_Simulation import Simulation
# import numpy as np
# import matplotlib.pyplot as plt
# import DriftDiffusionFunctionBank as functionBank
# from Errors import ErrorValsOneTime
# import time

# dimension = 2
# if dimension ==1:
#     beta = 5
#     radius = 3
#     # kstepMin= 0.08
#     # kstepMax = 0.09
#     kstepMin= 0.15
#     kstepMax = 0.2
#     h = 0.01

# if dimension ==2:
#     beta = 3
#     radius =2
#     # radius = 0.5
#     kstepMin= 0.08
#     kstepMax = 0.09
#     kstepMin= 0.13
#     kstepMax = 0.15
#     h = 0.05

# if dimension ==3:
#     beta = 3
#     radius = 0.5
#     kstepMin= 0.08
#     kstepMax = 0.085
#     h = 0.01

# # driftFunction = functionBank.zeroDrift
# # driftFunction = functionBank.erfDrift
# driftFunction = functionBank.oneDrift

# spatialDiff = False


# diffusionFunction = functionBank.ptSixDiffusion


# adaptive = True

# ApproxSolution =False

# sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)

# ErrorsAM = []
# ErrorsEM = []
# timesAM =[]
# timesEM = []
# betaVals = [2, 3, 4]
# # betaVals = [2.5]
# # radiusVals = [1, 4]
# # spacingVals = [0.08, 0.05]
# spacingVals = [0.2]

# # times = [4,8,10]
# # times = [12]
# endTime = 10
# sizes = [5,10,20]

# numPoints = []
# for size in sizes:
#     xmin = 0
#     xmax = size
#     ymin =0
#     ymax = size
#     for spacing in spacingVals:
#         buffer = 1
#         xstart = np.floor(xmin) - buffer
#         xs = []
#         xs.append(xstart)
#         while xstart< xmax + buffer:
#             xs.append(xstart+spacing)
#             xstart += spacing

#         ystart = np.floor(ymin) - buffer
#         ys = []
#         ys.append(ystart)

#         while ystart< ymax+ buffer:
#             ys.append(ystart+spacing)
#             ystart += spacing

#         mesh = []
#         for i in xs:
#             for j in ys:
#                 mesh.append([i,j])
#         mesh = np.asarray(mesh)
#         parametersAM = Parameters(sde, beta, radius, spacing, spacing, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = mesh)
#         startAM = time.time()
#         simulationAM = Simulation(sde, parametersAM, endTime)
#         # plt.figure()
#         # plt.scatter(simulationAM.pdf.meshCoordinates[:,0], simulationAM.pdf.meshCoordinates[:,1])
#         timeStartupAM = time.time() - startAM
#         startAMNoStartup = time.time()
#         simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
#         endAM =time.time()
#         timesAM.append(np.copy(endAM-startAM))


#         if not ApproxSolution:
#             meshApprox = simulationAM.meshTrajectory[-1]
#             pdfApprox = sde.exactSolution(simulationAM.meshTrajectory[-1], endTime)
#         LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
#         ErrorsAM.append(np.copy(L2wErrors))
#         # del simulationAM
#         numPoints.append(np.copy(simulationAM.pdf.meshLength))



# plt.figure()
# simulation = simulationAM
# Meshes = simulation.meshTrajectory
# plt.loglog(np.asarray(numPoints),np.asarray(timesAM), 'or')
# plt.ylabel("time seconds")
# plt.xlabel("Number of Points")



from Class_Parameters import Parameters
from Class_PDF import PDF
from Class_SDE import SDE
from Class_Simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import DriftDiffusionFunctionBank as functionBank
from Errors import ErrorValsOneTime
import time

dimension = 2
if dimension ==1:
    beta = 5
    radius = 3
    # kstepMin= 0.08
    # kstepMax = 0.09
    kstepMin= 0.15
    kstepMax = 0.2
    h = 0.01

if dimension ==2:
    beta = 3
    radius =2
    # radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.09
    kstepMin= 0.13
    kstepMax = 0.15
    h = 0.05

if dimension ==3:
    beta = 3
    radius = 0.5
    kstepMin= 0.08
    kstepMax = 0.085
    h = 0.01

# driftFunction = functionBank.zeroDrift
# driftFunction = functionBank.erfDrift
driftFunction = functionBank.oneDrift

spatialDiff = False


diffusionFunction = functionBank.ptSixDiffusion


adaptive = True

ApproxSolution =False

sde = SDE(dimension, driftFunction, diffusionFunction, spatialDiff)

ErrorsAM = []
ErrorsEM = []
timesAM =[]
timesEM = []
betaVals = [2, 3, 4]
# betaVals = [2.5]
# radiusVals = [1, 4]
# spacingVals = [0.08, 0.05]
spacingVals = [0.2]

times = [4,8,10]
# times = [12]
sizes = [5,10,20]
size = 10
numPoints = []
for endTime in times:
    xmin = 0
    xmax = size
    ymin =0
    ymax = size
    for spacing in spacingVals:
        buffer = 1
        xstart = np.floor(xmin) - buffer
        xs = []
        xs.append(xstart)
        while xstart< xmax + buffer:
            xs.append(xstart+spacing)
            xstart += spacing

        ystart = np.floor(ymin) - buffer
        ys = []
        ys.append(ystart)

        while ystart< ymax+ buffer:
            ys.append(ystart+spacing)
            ystart += spacing

        mesh = []
        for i in xs:
            for j in ys:
                mesh.append([i,j])
        mesh = np.asarray(mesh)
        parametersAM = Parameters(sde, beta, radius, spacing, spacing, h,useAdaptiveMesh =False, timeDiscretizationType = "EM", integratorType="TR", OverideMesh = mesh)
        startAM = time.time()
        simulationAM = Simulation(sde, parametersAM, endTime)
        # plt.figure()
        # plt.scatter(simulationAM.pdf.meshCoordinates[:,0], simulationAM.pdf.meshCoordinates[:,1])
        timeStartupAM = time.time() - startAM
        startAMNoStartup = time.time()
        simulationAM.computeAllTimes(sde, simulationAM.pdf, parametersAM)
        endAM =time.time()
        timesAM.append(np.copy(endAM-startAMNoStartup))


        if not ApproxSolution:
            meshApprox = simulationAM.meshTrajectory[-1]
            pdfApprox = sde.exactSolution(simulationAM.meshTrajectory[-1], endTime)
        LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsOneTime(simulationAM.meshTrajectory[-1], simulationAM.pdfTrajectory[-1], meshApprox, pdfApprox, ApproxSolution)
        ErrorsAM.append(np.copy(L2wErrors))
        # del simulationAM
        numPoints.append(np.copy(simulationAM.pdf.meshLength))



plt.figure()
simulation = simulationAM
Meshes = simulation.meshTrajectory
plt.loglog(np.asarray(times),np.asarray(timesAM), 'or')
plt.ylabel("time seconds")
plt.xlabel("Number of Points")













