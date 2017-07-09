#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:24:32 2017
    Test OPS derivations
@author: amit
"""
import sys
sys.path.insert(0,'../')

from numpy import array, log, logspace, zeros, mean, absolute, \
    set_printoptions, concatenate
from src.lib.OPS import pointEnergyJacobian, totEnergyJacobian

import numpy.random as rd
import matplotlib.pyplot as plt

set_printoptions(threshold=110)

# Morse parameters
E = 1.0
re = 1.0
sigma = log(2.0)/0.1

# Kernel parameters
K = 1.0
a = 3.0
b = 2.0

alphaM = 1.0
alphaP = 1.0
alphaN = 1.0
alphaC = 1.0

params=array([alphaM, alphaP, alphaN, alphaC, E, re, sigma, K, a, b])

N = 2 # DON'T CHANGE THIS!!!

x = 10.0*rd.random((N*6,))

enTot, JTot = totEnergyJacobian(x, params )

v0 = x[0:3]; x0 = x[3:6]; v1 = x[6:9]; x1 = x[9:12]
points1 = concatenate([v0,x0,v1,x1])
points2 = concatenate([v1,x1,v0,x0])

temp = pointEnergyJacobian( points1, params )
enPoint1 = temp[0]
J1 = temp[1:]
temp = pointEnergyJacobian( points2, params )
enPoint2 = temp[0]
J2 = temp[1:]
enPoint = enPoint1 + enPoint2

print("Energy difference = {0}".format(enTot - enPoint))

JPoint = zeros(N*6)
JPoint += J1
JPoint[0:6] += J2[6:12]
JPoint[6:12] += J2[0:6]

print("Jacobian difference = ")
print(JTot - JPoint)

"""
Test consistency
"""

h = logspace(-15,-1,100)
#h = [1e-6]
err = zeros(len(h))
for hi in range(len(h)):
    hc = h[hi]
    Jh = zeros(N*6)
    J = zeros(N*6)
    x = 10.0*rd.random((N*6,))
    v0 = x[0:3]; x0 = x[3:6]; v1 = x[6:9]; x1 = x[9:12]
    
    points = concatenate([v0,x0,v1,x1])
    temp = pointEnergyJacobian( points, params)
    J0 = temp[1:]
    J += J0
    points = concatenate([v1,x1,v0,x0])
    temp = pointEnergyJacobian( points, params )
    J1 = temp[1:]
    J[0:6] += J1[6:12]
    J[6:12] += J1[0:6]
    
    for i in range(N*6):
        x[i] += hc
        v0 = x[0:3]; x0 = x[3:6]; v1 = x[6:9]; x1 = x[9:12]
        Ep = 0.0
        points = concatenate([v0,x0,v1,x1])
        temp = pointEnergyJacobian( points, params, calcJac=False )
        Ep += temp[0]
        points = concatenate([v1,x1,v0,x0])
        temp = pointEnergyJacobian( points, params, calcJac=False )
        Ep += temp[0]
        x[i] -= 2*hc
        v0 = x[0:3]; x0 = x[3:6]; v1 = x[6:9]; x1 = x[9:12]
        Em = 0.0
        points = concatenate([v0,x0,v1,x1])
        temp = pointEnergyJacobian( points, params, calcJac=False )
        Em += temp[0]
        points = concatenate([v1,x1,v0,x0])
        temp = pointEnergyJacobian( points, params, calcJac=False )
        Em += temp[0]
        Jh[i] = (Ep - Em)/(2.0*hc)
    errJ = J - Jh
    err[hi] = mean(absolute(errJ))

plt.loglog(h,err)
