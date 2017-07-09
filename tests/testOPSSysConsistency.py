#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 20:24:32 2017
    Test OPS derivations
@author: amit
"""
import sys
sys.path.insert(0,'../')

from numpy import array, log, logspace, zeros, mean, absolute, set_printoptions
from src.lib.OPS import sysEnergyJacobian, totEnergyJacobian
from numpy.linalg import norm
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

sr = 20.0

params=array([alphaM, alphaP, alphaN, alphaC, E, re, sigma, K, a, b, sr])

N = 2 # DON'T CHANGE THIS!!!

x = 10.0*rd.random((N*6,))

pos1 = x[3:6]
pos2 = x[9:12]
dist = norm( pos1 - pos2 )

print("Distance between the points = {0}".format(dist))

enTot, JTot = totEnergyJacobian(x, params )
enPoint,_ = sysEnergyJacobian( x, params, calcJac=False )
_, JPoint = sysEnergyJacobian( x, params )

print("Energy difference = {0}".format(enTot - enPoint))
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
    x = 10.0*rd.random((N*6,))    
    _,J = sysEnergyJacobian( x, params)    
    for i in range(N*6):
        x[i] += hc        
        Ep,_ = sysEnergyJacobian( x, params, calcJac=False )        
        x[i] -= 2*hc        
        Em,_ = sysEnergyJacobian( x, params, calcJac=False )       
        Jh[i] = (Ep - Em)/(2.0*hc)
    errJ = J - Jh
    err[hi] = mean(absolute(errJ))

plt.loglog(h,err)
