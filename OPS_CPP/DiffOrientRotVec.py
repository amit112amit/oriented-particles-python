#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:09:42 2017

@author: amit
"""

import sympy as sy
from sympy import sin, cos, sqrt, symbols, Rational, diff, latex

# Convenience function to write our results to file
def writeListToFile(fileName, listIn, varName):
    fileStream = open(fileName, 'w')
    fileStream.write('    u_2 = u0**2 + u1**2 + u2**2\n')    
    fileStream.write('    u = sqrt(u_2)\n')
    fileStream.write('    u_3 = u**3\n')    
    fileStream.write('    u_4 = u**4\n')    
    fileStream.write('    sin_alpha = sin( 0.5*u )\n')    
    for i in range(len(listIn)):
        var = '    ' + varName + '{0}'.format(i) + ' = '
        fileStream.write(var)
        #fileStream.write(str(listIn[i]))
        fileStream.write(str(latex(listIn[i])))
        fileStream.write('\n\n')
    fileStream.close()

# Define the symbolic variables to be used
u0,u1,u2,v0,v1,v2,x0,x1,x2,y0,y1,y2 = \
    symbols('u0,u1,u2,v0,v1,v2,x0,x1,x2,y0,y1,y2', real=True)
    
#Derivatives with respect to x0,x1,x2

u = sqrt( u0**2 + u1**2 + u2**2 )
alpha = Rational(1,2)*u

p0 = (2*sin(alpha)/u**2)*( u1*u*cos(alpha) + u0*u2*sin(alpha) )
p1 = (2*sin(alpha)/u**2)*( -u0*u*cos(alpha) + u1*u2*sin(alpha) )
p2 = cos(alpha)**2 +(sin(alpha)**2/u**2)*( u2**2 -u1**2 - u0**2 )

dp0du0 = diff(p0,u0)
dp1du0 = diff(p1,u0)
dp2du0 = diff(p2,u0)

dp0du1 = diff(p0,u1)
dp1du1 = diff(p1,u1)
dp2du1 = diff(p2,u1)

dp0du2 = diff(p0,u2)
dp1du2 = diff(p1,u2)
dp2du2 = diff(p2,u2)

dpdu = [ dp0du0,dp1du0, dp2du0,\
        dp0du1,dp1du1, dp2du1,\
        dp0du2,dp1du2, dp2du2 ]


U,A = symbols('U,A')

subsList = [ (sqrt(u0**2 + u1**2 + u2**2)/2, A),\
            (sqrt(u0**2 + u1**2 + u2**2), U)]

derivatives = []
for i in range(9):
    derivatives.append( dpdu[i].subs(subsList) )

writeListToFile( 'Latex.txt',derivatives, 'dpdu' );