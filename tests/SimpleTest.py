#!/usr/bin/env python3
# -*- coding: utf-8 -*-
        
"""
Created on Sun Jun 25 20:09:56 2017
    Simple N=2 example test
@author: amit
"""

import sys
sys.path.insert(0,'../')

from sympy import symbols, diff
from numpy import array, zeros, set_printoptions, mean, absolute
from numpy.random import random

from src.derivations import PotentialsSym as ps
from src.lib import OPS as op1

set_printoptions(threshold=20)

x0,x1,x2,y0,y1,y2 = symbols('x0,x1,x2,y0,y1,y2')
p0,p1,p2,q0,q1,q2 = symbols('p0,p1,p2,q0,q1,q2')
E,r,s,K,a,b = symbols('E,r,s,K,a,b')

x = [x0,x1,x2]
y = [y0,y1,y2]
p = [p0,p1,p2]
q = [q0,q1,q2]

Am,Ap,An,Ac = symbols('Am,Ap,An,Ac')

enSym = Am*(ps.morse(x,y,E,r,s) + ps.morse(y,x,E,r,s)) +\
     Ap*(ps.phi_p(p,x,y)*ps.psi(p,x,y,K,a,b) + \
         ps.phi_p(q,y,x)*ps.psi(q,y,x,K,a,b)) +\
     An*(ps.phi_n(p,q)*ps.psi(p,x,y,K,a,b) + \
         ps.phi_n(q,p)*ps.psi(q,y,x,K,a,b)) +\
     Ac*(ps.phi_c(p,q,x,y)*ps.psi(p,x,y,K,a,b) + \
         ps.phi_c(q,p,y,x)*ps.psi(q,y,x,K,a,b))

JSym = [0,0,0,0,0,0,0,0,0,0,0,0]
v = [p0,p1,p2,x0,x1,x2,q0,q1,q2,y0,y1,y2]

for i in range(len(v)):
    JSym[i] = diff(enSym,v[i])

varVal = random(len(v))
subsList = [(v[i], varVal[i]) for i in range(len(v))]
paramList = [Am,Ap,An,Ac,E,r,s,K,a,b]
paramVal = [1.0,1.0,1.0,1.0,1.0,1.0,6.931,10.0,3.0,2.0]

for i in range(len(paramList)):
    subsList.append((paramList[i],paramVal[i]))

enDir = enSym.subs(subsList)
str1 = "Direct evaluation of energy gives {0}".format(enDir)
print(str1)

JDir = array([JSym[i].subs(subsList) for i in range(len(v))])
print("\nJacobian evaluated directly gives:")
print(JDir)

params = paramVal
x_in = varVal
N = 2

enFun, JFun = op1.totEnergyJacobian(x_in, params)
str2 = "\nEnergy calculated using our function is {0}".format(enFun)
print(str2)
print("\nJacobian evaluated using function: ")
print(JFun)

# Numerical Jacobian
hc = 1e-6
Jh = zeros(N*6)
for i in range(N*6):
    x_in[i] += hc
    Ep,_ = op1.totEnergyJacobian(x_in,params,calcJac=False)
    x_in[i] -= 2*hc
    Em,_ = op1.totEnergyJacobian(x_in,params,calcJac=False)
    Jh[i] = (Ep - Em)/(2.0*hc)
print("Numerical Jacobian:")
print(Jh)

str1 = "Mean abs diff between JDir and Jh : {0}".format(mean(absolute(JDir -\
                                            Jh)))
print(str1)
str1 = "Mean abs diff between JDir and JFun : {0}".format(mean(absolute(JDir -\
                                            JFun)))
print(str1)
str1 = "Mean abs diff between JFun and Jh : {0}".format(mean(absolute(JFun -\
                                            Jh)))
print(str1)