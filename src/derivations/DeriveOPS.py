#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../../')

import sympy as sy

from src.derivations import PotentialsSym as ps

# Convenience function to write our results to file
def writeListToFile(fileName, listIn, varName):
    fileStream = open(fileName, 'w')
    fileStream.write('    vi_mag_sqr = u0**2 + u1**2 + u2**2\n')
    fileStream.write('    vj_mag_sqr = v0**2 + v1**2 + v2**2\n')
    fileStream.write('    vi_mag = sqrt(vi_mag_sqr)\n')
    fileStream.write('    vj_mag = sqrt(vj_mag_sqr)\n')
    fileStream.write('    sin_alpha_i = sin( 0.5*vi_mag )\n')
    fileStream.write('    sin_alpha_j = sin( 0.5*vj_mag )\n')
    fileStream.write('    cos_alpha_i = cos( 0.5*vi_mag )\n')
    fileStream.write('    cos_alpha_j = cos( 0.5*vj_mag )\n')
    str1 = '    rij = sqrt( (-x0+y0)**2 + (-x1+y1)**2 + (-x2+y2)**2 )\n'
    fileStream.write(str1)
    fileStream.write('\n')
    for i in range(len(listIn)):
        var = '    ' + varName + '{0}'.format(i) + ' = '
        fileStream.write(var)
        fileStream.write(str(listIn[i]))
        fileStream.write('\n\n')
    fileStream.close()

# Define the symbolic variables to be used
u0,u1,u2,v0,v1,v2,x0,x1,x2,y0,y1,y2 = \
    sy.symbols('u0,u1,u2,v0,v1,v2,x0,x1,x2,y0,y1,y2', real=True)

# Rotation vector
vi = [u0, u1, u2]
vj = [v0, v1, v2]

# Points in the system
xi = [x0,x1,x2]
xj = [y0,y1,y2]

# The Morse potential
epsilon, l0, a = sy.symbols('epsilon, l0, a')
morsePot = ps.morse(xi, xj, epsilon, l0, a)

# The kernel
a,b,K = sy.symbols('a,b,K', real=True)
kernel = ps.psi(vi, xi, xj, K, a, b)

# The co-planarity potential
planarPot = ps.phi_p(vi, xi, xj)

# The co-normality potential
normalPot = ps.phi_n(vi, vj)

# The co-circularity potential
circPot = ps.phi_c(vi, vj, xi, xj)

# The Jacobian of morsePot wrt xi. Driver must ensure that xi != xj
DmorseDxi = [sy.diff(morsePot, i) for i in [x0,x1,x2]]
DmorseDxj = [sy.diff(morsePot, i) for i in [y0,y1,y2]]

# The Jacobian of kernel wrt vi
DpsiDvi = [sy.diff(kernel, i) for i in [u0,u1,u2]]

# The Jacobian of kernel wrt xi
DpsiDxi = [sy.diff(kernel, i) for i in [x0,x1,x2]]
DpsiDxj = [sy.diff(kernel, i) for i in [y0,y1,y2]]

# The Jacobian of planarPot wrt vi
Dphi_pDvi = [sy.diff(planarPot, i) for i in [u0,u1,u2]]

# The Jacobian of planarPot wrt xi and xj
Dphi_pDxi = [sy.diff(planarPot,i) for i in [x0,x1,x2]]
Dphi_pDxj = [sy.diff(planarPot,i) for i in [y0,y1,y2]]

# The Jacobian of normalPot wrt vi and vj
Dphi_nDvi = [sy.diff(normalPot, i) for i in [u0,u1,u2]]
Dphi_nDvj = [sy.diff(normalPot, i) for i in [v0,v1,v2]]

# The Jacobian of circPot wrt vi and vj
Dphi_cDvi = [sy.diff(circPot, i) for i in [u0,u1,u2]]
Dphi_cDvj = [sy.diff(circPot, i) for i in [v0,v1,v2]]

# The Jacobian of circPot wrt xi and xj
Dphi_cDxi = [sy.diff(circPot,i) for i in [x0,x1,x2]]
Dphi_cDxj = [sy.diff(circPot,i) for i in [y0,y1,y2]]

# Make simplifying substitutions
vi_mag_sqr,vj_mag_sqr,vi_mag,vj_mag = \
    sy.symbols('vi_mag_sqr,vj_mag_sqr,vi_mag,vj_mag')
sin_alpha_i,cos_alpha_i,sin_alpha_j,cos_alpha_j, rij = \
    sy.symbols('sin_alpha_i,cos_alpha_i,sin_alpha_j,cos_alpha_j,rij')

subsList = [(u0**2 + u1**2 + u2**2, vi_mag_sqr), \
            (v0**2 + v1**2 + v2**2, vj_mag_sqr), \
            (sy.sqrt(vi_mag_sqr), vi_mag), (sy.sqrt(vj_mag_sqr), vj_mag), \
            (sy.sin(vi_mag/2), sin_alpha_i),(sy.cos(vi_mag/2), cos_alpha_i), \
            (sy.sin(vj_mag/2), sin_alpha_j),(sy.cos(vj_mag/2), cos_alpha_j), \
            (sy.sqrt((-x0+y0)**2 + (-x1+y1)**2 + (-x2+y2)**2), rij)]

eqns = [kernel, planarPot, normalPot, circPot, DmorseDxi, DmorseDxj, DpsiDvi,\
        DpsiDxi, DpsiDxj, Dphi_pDvi, Dphi_pDxi, Dphi_pDxj, Dphi_nDvi, \
        Dphi_nDvj, Dphi_cDvi, Dphi_cDvj, Dphi_cDxi, Dphi_cDxj]

fileNames = ['psi','phi_p','phi_n','phi_c','DmorseDxi',\
            'DmorseDxj','DpsiDvi','DpsiDxi','DpsiDxj','Dphi_pDvi',\
            'Dphi_pDxi','Dphi_pDxj','Dphi_nDvi','Dphi_nDvj','Dphi_cDvi',\
            'Dphi_cDvj','Dphi_cDxi','Dphi_cDxj']

varNames = ['psi','phi_p','phi_n','phi_c','Dmorse_Dxi','Dmorse_Dxj','Dpsi_Du',\
            'Dpsi_Dxi','Dpsi_Dxj','Dphi_pDu','Dphi_pDxi','Dphi_pDxj',\
            'Dphi_nDu','Dphi_nDv','Dphi_cDu','Dphi_cDv','Dphi_cDxi',\
            'Dphi_cDxj']

newEqns = []

for eqn in eqns:    
    if type(eqn) != list:
        temp = eqn.subs(subsList)        
    else:
        temp = []
        for term in eqn:
            temp2 = term.subs(subsList)
            temp.append(temp2)
    newEqns.append(temp)

for i in range(len(newEqns)):
    eqn = newEqns[i]
    fileName = fileNames[i] + '.txt'
    if type(eqn) != list:        
        writeListToFile( fileName, [eqn], varNames[i] )
    else:        
        writeListToFile( fileName, eqn, varNames[i] )
