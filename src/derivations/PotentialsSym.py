#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 18:29:22 2017
    Oriented Partilce System potentials symbolic functions
@author: amit
"""

import sympy as sy

# Quaternion multiplication
def quatMul(p,q,i,j,k):
    p0,p1,p2,p3 = p
    q0,q1,q2,q3 = q
    out = [p0*q0 - p1*q1 - p2*q2 - p3*q3, p0*q1 + p1*q0 + p2*q3 - p3*q2, \
           p0*q2 - p1*q3 + p2*q0 + p3*q1, p0*q3 + p1*q2 - p2*q1 + p3*q0]
    return out

def conjugation(q,p):
    # We will assume that q is quaternion and p is vector
    q0,q1,q2,q3 = q
    p1,p2,p3 = p
    ans = [p1*q0**2 + p1*q1**2 - p1*q2**2 - p1*q3**2 - 2*p2*q0*q3 +\
           2*p2*q1*q2 + 2*p3*q0*q2 + 2*p3*q1*q3, 2*p1*q0*q3 + 2*p1*q1*q2 +\
           p2*q0**2 - p2*q1**2 + p2*q2**2 - p2*q3**2 - 2*p3*q0*q1 +\
           2*p3*q2*q3, -2*p1*q0*q2 + 2*p1*q1*q3 + 2*p2*q0*q1 + 2*p2*q2*q3 +\
           p3*q0**2 - p3*q1**2 - p3*q2**2 + p3*q3**2]
    return ans

def rotateByQuat(q,z):
    q0,q1,q2,q3 = q
    z0,z1,z2 = z
    Qq = sy.Matrix([[1, 0, 0, 0],
          [0, q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
          [0, 2*(q1*q2-q0*q3),q0*q0-q1*q1+q2*q2-q3*q3,2*(q2*q3+q0*q1)],
          [0, 2*(q1*q3+q0*q2),2*(q2*q3-q0*q1),q0*q0-q1*q1-q2*q2+q3*q3]
         ])
    z_mod = sy.Matrix([0, z0, z1, z2])
    out = Qq*z_mod
    out_final = []
    for i in range(4):
        out_final.append( sy.expand(out[i]) )
    return out_final

# Morse potential
def morse(xi, xj, epsilon, l0, a):
    a1,b1,c1 = xi
    a2,b2,c2 = xj
    out = epsilon*(-2*sy.exp(-a*(-l0 + sy.sqrt((-a1 + a2)**2 + (-b1 + b2)**2 +\
    (-c1 + c2)**2))) + sy.exp(-2*a*(-l0 + sy.sqrt((-a1 + a2)**2 +\
    (-b1 + b2)**2 + (-c1 + c2)**2))))
    return out

# Define the kernel function
def psi(vi, xi, xj, K, a, b):
    # Convert rotation vector to unit quaternion
    v0, v1, v2 = vi
    v = sy.sqrt(v0**2 + v1**2 + v2**2)
    q0 = sy.cos( sy.Rational(1,2)*v )
    q1 = sy.sin( sy.Rational(1,2)*v)*(v0/v)
    q2 = sy.sin( sy.Rational(1,2)*v)*(v1/v)
    q3 = sy.sin( sy.Rational(1,2)*v)*(v2/v)
    qi = [q0, q1, q2, q3]

    rij = [xj[0]-xi[0],xj[1]-xi[1],xj[2]-xi[2]]
    dij = conjugation(qi, rij)
    x,y,z = dij
    out = K*sy.exp( -(x**2 + y**2)/(2*a**2) - z**2/(2*b**2) )
    return out

# In the following function the input qi, xi, xj must be of list type
def phi_p(vi, xi, xj):
    # Convert rotation vector to unit quaternion
    u0, u1, u2 = vi
    u = sy.sqrt(u0**2 + u1**2 + u2**2)
    q0 = sy.cos( sy.Rational(1,2)*u )
    q1 = sy.sin( sy.Rational(1,2)*u )*(u0/u)
    q2 = sy.sin( sy.Rational(1,2)*u )*(u1/u)
    q3 = sy.sin( sy.Rational(1,2)*u )*(u2/u)
    qi = [q0, q1, q2, q3]

    #Get normal ni for point xi by rotating z-axis of Global Coordinate System
    qz = [0,0,1]
    ni = conjugation( qi, qz )

    #Rotate rij(global coords) to dij(local coords)
    rij = [0,0,0]
    for z in range(len(xj)):
        rij[z] = xj[z] - xi[z]

    # Calculate dot product of ni and rij
    ni_dot_rij = ni[0]*rij[0] + ni[1]*rij[1] + ni[2]*rij[2]

    # Construct the potential value
    out = ni_dot_rij**2
    return out

# Co-normality potential
def phi_n(vi,vj):
    # Convert rotation vector vi to unit quaternion
    u0, u1, u2 = vi
    u = sy.sqrt(u0**2 + u1**2 + u2**2)
    p0 = sy.cos( sy.Rational(1,2)*u )
    p1 = sy.sin( sy.Rational(1,2)*u )*(u0/u)
    p2 = sy.sin( sy.Rational(1,2)*u )*(u1/u)
    p3 = sy.sin( sy.Rational(1,2)*u )*(u2/u)
    p = [p0, p1, p2, p3]
    # Convert rotation vector vj to unit quaternion
    v0, v1, v2 = vj
    v = sy.sqrt(v0**2 + v1**2 + v2**2)
    q0 = sy.cos( sy.Rational(1,2)*v )
    q1 = sy.sin( sy.Rational(1,2)*v )*(v0/v)
    q2 = sy.sin( sy.Rational(1,2)*v )*(v1/v)
    q3 = sy.sin( sy.Rational(1,2)*v )*(v2/v)
    q = [q0, q1, q2, q3]

    #Get normal ni, nj for points xi, xj by rotating z-axis of Global Coord Sys
    qz = [0,0,1]
    ni = conjugation( p, qz )
    nj = conjugation( q, qz )
    out = 0
    for i in range(len(ni)):
        out += (ni[i] - nj[i])**2
    return out

# Co-circularity potential
def phi_c(vi,vj,xi,xj):
    # Convert rotation vector vi to unit quaternion
    u0, u1, u2 = vi
    u = sy.sqrt(u0**2 + u1**2 + u2**2)
    p0 = sy.cos( sy.Rational(1,2)*u )
    p1 = sy.sin( sy.Rational(1,2)*u )*(u0/u)
    p2 = sy.sin( sy.Rational(1,2)*u )*(u1/u)
    p3 = sy.sin( sy.Rational(1,2)*u )*(u2/u)
    p = [p0, p1, p2, p3]
    # Convert rotation vector vj to unit quaternion
    v0, v1, v2 = vj
    v = sy.sqrt(v0**2 + v1**2 + v2**2)
    q0 = sy.cos( sy.Rational(1,2)*v )
    q1 = sy.sin( sy.Rational(1,2)*v )*(v0/v)
    q2 = sy.sin( sy.Rational(1,2)*v )*(v1/v)
    q3 = sy.sin( sy.Rational(1,2)*v )*(v2/v)
    q = [q0, q1, q2, q3]

    #Get normal ni, nj for points xi, xj by rotating z-axis of Global Coord Sys
    qz = [0,0,1]
    ni = conjugation( p, qz )
    nj = conjugation( q, qz )

    #Rotate rij(global coords) to dij(local coords)
    rij = [0,0,0]
    for z in range(len(xj)):
        rij[z] = xj[z] - xi[z]

    # Calculate ni + nj
    ni_p_nj = [ (ni[i] + nj[i]) for i in range(len(ni)) ]

    # Calculate dot product of ni and rij
    ni_p_nj_dot_rij = ni_p_nj[0]*rij[0] + ni_p_nj[1]*rij[1] + ni_p_nj[2]*rij[2]

    # Construct the potential value
    out = ni_p_nj_dot_rij**2

    return out
