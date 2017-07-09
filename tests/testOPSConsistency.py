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
from src.lib.OPS import totEnergyJacobian

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

N = 10

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
    _,J = totEnergyJacobian(x,params)
    for i in range(N*6):
        x[i] += hc
        Ep,_ = totEnergyJacobian(x,params,calcJac=False)
        x[i] -= 2*hc
        Em,_ = totEnergyJacobian(x,params,calcJac=False)
        Jh[i] = (Ep - Em)/(2.0*hc)
    errJ = J - Jh
    err[hi] = mean(absolute(errJ))

plt.loglog(h,err)