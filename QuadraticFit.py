
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from pyopoly1.Scaling import GaussScale
import Functions as fun


def fitQuad(mesh, pdf):
    zobs = np.log(pdf)
    zobs = np.squeeze(zobs)
    xy = mesh.T
    x, y = mesh.T
    try:
        pred_params, uncert_cov = opt.curve_fit(quad, xy, zobs, p0 = [0,0,0,0,0,0])
    except:
        return float('nan'),float('nan'),float('nan'),float('nan')
    
    c = pred_params
    A = np.asarray([[c[0], c[2]],[c[2],c[1]]])
    B = np.expand_dims(np.asarray([c[3], c[4]]),1)
    
    if np.linalg.det(A)<= 0:
         return float('nan'),float('nan'),float('nan'),float('nan')
         
    sigma = np.linalg.inv(A)
    Lam, U = np.linalg.eigh(A)
    if np.min(Lam) <= 0:
        return float('nan'),float('nan'),float('nan'),float('nan')
    
    La = np.diag(Lam)
    mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T    
    Const = np.exp(-c[5]+1/4*B.T@U@np.linalg.inv(La)@U.T@B)
    
    if math.isfinite(mu[0][0]) and math.isfinite(mu[1][0]) and math.isfinite(np.sqrt(sigma[0,0])) and math.isfinite(np.sqrt(sigma[1,1])):
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[mu[0][0],mu[1][0]]]).T)
        scaling.setCov(sigma)
        
    # cc=pred_params
    # x,y = xy   
    # vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const[0][0]
    
    return scaling, pdf, pred_params, Const


def quad(xy, a, b, c, d, e, f):
    x, y = xy
    # A= np.asarray([[a,c],[c,b]])
    # B=np.asarray([[d, e]]).T
    quad = -(a*x**2+ b*y**2 + 2*c*x*y + d*x + e*y + f)
    return quad



def leastSquares(mesh, pdf):
    x = mesh[:,0]
    y = mesh[:,1]
    A = np.zeros((len(mesh), 6))
    
    A[:,0] = -x**2
    A[:,1] = -y**2
    A[:,2] = -2*x*y
    A[:,3] = -x
    A[:,4] = -y
    A[:,5] = -np.ones(len(mesh))
    
    AT = A.T
    const = np.linalg.inv(AT@A)@(AT@np.log(pdf))
    c=const.T[0]
    
    A = np.asarray([[c[0], c[2]],[c[2],c[1]]])
    B = np.expand_dims(np.asarray([c[3], c[4]]),1)
    
    if np.linalg.det(A)<= 0:
         return float('nan'),float('nan'),float('nan')
         
    sigma = np.linalg.inv(A)
    Lam, U = np.linalg.eigh(A)
    if np.min(Lam) <= 0:
        return float('nan'),float('nan'),float('nan')
    
    La = np.diag(Lam)
    mu = -1/2*U @ np.linalg.inv(La) @ (B.T @ U).T    
    Const = np.exp(-c[5]+1/4*B.T@U@np.linalg.inv(La)@U.T@B)
    
    if math.isfinite(mu[0][0]) and math.isfinite(mu[1][0]) and math.isfinite(np.sqrt(sigma[0,0])) and math.isfinite(np.sqrt(sigma[1,1])):
        scaling = GaussScale(2)
        scaling.setMu(np.asarray([[mu[0][0],mu[1][0]]]).T)
        scaling.setCov(sigma)
    else:
        return float('nan'),float('nan'),float('nan')
    # cc=pred_params
    # x,y = xy   
    # vals = np.exp(-(cc[0]*x**2+ cc[1]*y**2 + 2*cc[2]*x*y + cc[3]*x + cc[4]*y + cc[5]))/Const[0][0]
    
    return scaling, c, Const



    
    


