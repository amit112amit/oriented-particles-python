#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:06:27 2017
    Quaternion multiplication results
@author: amit
"""
from numba import jit
from numpy import array, sqrt, arccos, sin, cos, dot, zeros

"""
quatMul(q,p): returns Hamiltonian product of q and p
"""
@jit(cache=True,nopython=True)
def quatMul(p,q):    
    # We will assume that q and p are quaternions    
    p0,p1,p2,p3 = p
    q0,q1,q2,q3 = q
    
    ans = array([p0*q0 - p1*q1 - p2*q2 - p3*q3, p0*q1 + p1*q0 + p2*q3 - p3*q2,
           p0*q2 - p1*q3 + p2*q0 + p3*q1, p0*q3 + p1*q2 - p2*q1 + p3*q0])
    
    return ans

"""
conjugation(q,v): returns a quaternion
    q is the rotation quaternion, v is vector to be rotated
"""
@jit(cache=True,nopython=True)
def conjugation(q,p):
    # We will assume that q and p are quaternions
    q0,q1,q2,q3 = q
    p1,p2,p3 = p
    ans = array([p1*q0**2 + p1*q1**2 - p1*q2**2 - p1*q3**2 - 2*p2*q0*q3 +\
           2*p2*q1*q2 + 2*p3*q0*q2 + 2*p3*q1*q3, 2*p1*q0*q3 + 2*p1*q1*q2 +\
           p2*q0**2 - p2*q1**2 + p2*q2**2 - p2*q3**2 - 2*p3*q0*q1 +\
           2*p3*q2*q3, -2*p1*q0*q2 + 2*p1*q1*q3 + 2*p2*q0*q1 + 2*p2*q2*q3 +\
           p3*q0**2 - p3*q1**2 - p3*q2**2 + p3*q3**2])
    return ans

"""
rotMatToUQuat(R): converts 3x3 rotation matrix R to a unit-quaternion
"""
@jit(cache=True,nopython=True)
def rotMatToUQuat(R):
    # Extract all elements from R
    r11,r12,r13 = R[0]
    r21,r22,r23 = R[1]
    r31,r32,r33 = R[2]
    
    if r22 > -r33 and r11 > -r22 and r11 > -r33:
        r = sqrt(1 + r11 + r22 + r33)
        q = 0.5*array([ r, (r23 - r32)/r, (r31 - r13)/r ,(r12 - r21)/r ])
    elif r22 < -r33 and r11 > r22 and r11 > r33:
        r = sqrt(1 + r11 - r22 - r33)
        q = 0.5*array([ (r23 - r32)/r, r, (r12 + r21)/r, (r31 + r13)/r ])
    elif r22 > r33 and r11 < r22 and r11 < -r33:
        r = sqrt(1 - r11 + r22 - r33)
        q = 0.5*array([ (r31 - r13)/r, (r12 + r21)/r, r, (r23 + r32)/r ])
    elif r22 < r33 and r11 < -r22 and r11 < r33:
        r = sqrt(1 - r11 - r22 + r33)
        q = 0.5*array([ (r12 - r21)/r, (r31 + r13)/r, (r23 + r32)/r, r ])
        
    return q

"""
uQuatToRotVec(q): returns a rotation vector from a unit quaternion q
"""
@jit(cache=True,nopython=True)
def uQuatToRotVec(q):
    q0,q1,q2,q3 = q
    q123 = array( [q1,q2,q3] ) 
    v = (2*arccos(q0))/sqrt(1 - q0*q0)*q123
    return v

"""
rotVecToUQuat(r): returns a unit quaternion from a rotation vector v
"""
@jit(cache=True,nopython=True)
def rotVecToUQuat(v):    
    v_norm = sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] )
    q = array([ cos(0.5*v_norm), 0.0, 0.0, 0.0])
    q[1:4] = (sin( 0.5*v_norm )/v_norm)*v
    return q

"""
 rotByUQuat(q,z): rotates the vector z by unit quaternion q and returns 
     a vector instead of a quaternion by stripping the 0 first component
"""
@jit(cache=True,nopython=True)
def rotByUQuat(q,z):
    q0,q1,q2,q3 = q
    Rq_flat = array([q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2),
                2*(q1*q2-q0*q3),q0*q0-q1*q1+q2*q2-q3*q3,2*(q2*q3+q0*q1),
                2*(q1*q3+q0*q2),2*(q2*q3-q0*q1),q0*q0-q1*q1-q2*q2+q3*q3
            ])
    Rq = Rq_flat.reshape((3,3))
    Qp = zeros((4,4))
    Qp[0] = array([1.0,0.0,0.0,0.0])
    Qp[1:4,0] = array([0.0,0.0,0.0])
    Qp[1:4,1:4] = Rq
    z_mod = array([0.0, z[0],z[1],z[2]])
    out = dot(Qp,z_mod)    
    return out

"""
# Test if rotating y-axis by pi/2 gives z-axis
from numpy import sqrt

s = 1.0/sqrt(2.0)
q = array([s,s,0.,0.])
v1 = array([0.,1.,0.])
v0 = array([0.,0.,1.])
ans0 = conjugation(q,v0)
ans1 = conjugation(q,v1)
print(ans0)
print(ans1)

# Test Hamiltonian product
q1 = array([1.0,2.0,3.0,4.0])
q2 = array([5.0,6.0,7.0,8.0])
ans2 = quatMul(q1,q2)
print(ans2)
"""