#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:48:17 2017

@author: amit
"""
import sympy as sy
from PotentialsSym import conjugation, phi_p, phi_n, phi_c

# Define the symbolic variables to be used
u0,u1,u2,v0,v1,v2,x0,x1,x2,y0,y1,y2 = \
    sy.symbols('u0,u1,u2,v0,v1,v2,x0,x1,x2,y0,y1,y2', real=True)

# Convert rotation vector vi to unit quaternion   
u = sy.sqrt(u0**2 + u1**2 + u2**2)
p0 = sy.cos( sy.Rational(1,2)*u )
p1 = sy.sin( sy.Rational(1,2)*u )*(u0/u)
p2 = sy.sin( sy.Rational(1,2)*u )*(u1/u)
p3 = sy.sin( sy.Rational(1,2)*u )*(u2/u)
p = [p0, p1, p2, p3]

# Convert rotation vector vj to unit quaternion    
v = sy.sqrt(v0**2 + v1**2 + v2**2)
q0 = sy.cos( sy.Rational(1,2)*v )
q1 = sy.sin( sy.Rational(1,2)*v )*(v0/v)
q2 = sy.sin( sy.Rational(1,2)*v )*(v1/v)
q3 = sy.sin( sy.Rational(1,2)*v )*(v2/v)
q = [q0, q1, q2, q3]

qz = [0,0,1]
ni = conjugation( p, qz )
nj = conjugation( q, qz )

# Make simplifying substitutions
u,v,u_2,v_2,u_3,v_3,u_4,v_4 = \
sy.symbols('u,v,u_2,v_2,u_3,v_3,u_4,v_4')
sin_alpha_i,cos_alpha_i,sin_alpha_j,cos_alpha_j = \
    sy.symbols('sin_alpha_i,cos_alpha_i,sin_alpha_j,cos_alpha_j')

subsList = [(u0**2 + u1**2 + u2**2, u_2), \
            (v0**2 + v1**2 + v2**2, v_2), \
            (sy.sqrt(u_2), u), (sy.sqrt(v_2), v), \
            (u**2, u_2),(v**2, v_2),\
            (u**3, u_3),(v**3, v_3), \
            (u**4, u_4),(v**4, v_4 ), \
            (sy.sin(u/2), sin_alpha_i),(sy.cos(u/2), cos_alpha_i), \
            (sy.sin(v/2), sin_alpha_j),(sy.cos(v/2), cos_alpha_j)]

ni_s =[]
nj_s = []
for i in range( 3 ):
    ni_s.append( ni[i].subs(subsList) )
    nj_s.append( nj[i].subs(subsList) )

