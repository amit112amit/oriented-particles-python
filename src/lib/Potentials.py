#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 18:29:22 2017
    Oriented Partilce System potentials NUMERICAL functions
@author: amit
"""

from numba import jit
from numpy import exp, sqrt, cos, sin

# Morse potential
@jit(cache=True,nopython=True)
def morse(xi, xj, epsilon, l0, a):
    a1,b1,c1 = xi
    a2,b2,c2 = xj
    r = sqrt((-a1 + a2)**2 + (-b1 + b2)**2 + (-c1 + c2)**2)
    out = epsilon*(-2*exp(-a*(-l0 + r)) + exp(-2*a*(-l0 + r)))
    return out

# The kernel function
@jit(cache=True,nopython=True)
def psi(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0**2 + u1**2 + u2**2    
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )

    psi0 = K*exp(-(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)**2/(2*b**2) + (-\
        (cos_alpha_i**2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +\
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2)**2 - (cos_alpha_i**2*(-\
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2)**2)/(2*a**2))    
    
    return psi0

# The co-planarity potential
@jit(cache=True,nopython=True)
def phi_p(vi, xi, xj):    
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vi_mag = sqrt(vi_mag_sqr)    
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    
    phi_p0 = ((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))**2
    
    return phi_p0

# The co-normality potential
@jit(cache=True,nopython=True)
def phi_n(vi, vj):    
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    
    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vj_mag_sqr = v0**2 + v1**2 + v2**2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )

    phi_n0 = (-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2 -\
         2*sin_alpha_j**2*v1*v2/vj_mag**2)**2 +\
         (2*cos_alpha_i*sin_alpha_i*u1/vi_mag -\
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2 -\
         2*sin_alpha_j**2*v0*v2/vj_mag**2)**2 + (cos_alpha_i**2 -\
         cos_alpha_j**2 - sin_alpha_i**2*u0**2/vi_mag**2 -\
         sin_alpha_i**2*u1**2/vi_mag**2 + sin_alpha_i**2*u2**2/vi_mag**2 +\
         sin_alpha_j**2*v0**2/vj_mag**2 + sin_alpha_j**2*v1**2/vj_mag**2 -\
         sin_alpha_j**2*v2**2/vj_mag**2)**2

    return phi_n0

# The co-circularity potential
@jit(cache=True,nopython=True)
def phi_c(vi, vj, xi, xj):    
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vj_mag_sqr = v0**2 + v1**2 + v2**2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )

    phi_c0 = ((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2 +\
         2*sin_alpha_j**2*v0*v2/vj_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -\
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2 +\
         2*sin_alpha_j**2*v1*v2/vj_mag**2) + (-x2 + y2)*(cos_alpha_i**2 +\
         cos_alpha_j**2 - sin_alpha_i**2*u0**2/vi_mag**2 -\
         sin_alpha_i**2*u1**2/vi_mag**2 + sin_alpha_i**2*u2**2/vi_mag**2 -\
         sin_alpha_j**2*v0**2/vj_mag**2 - sin_alpha_j**2*v1**2/vj_mag**2 +\
         sin_alpha_j**2*v2**2/vj_mag**2))**2
    
    return phi_c0
