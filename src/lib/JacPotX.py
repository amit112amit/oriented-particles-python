#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:18:19 2017
    Numerical values for derivatives of the potentials
@author: amit
"""

from numba import jit, f8
from numpy import exp, sqrt, array, cos, sin

"""
Derivative of Morse potential wrt xi
"""
@jit( f8[:](f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dmorse_Dxi(xi, xj, epsilon, l0, a):
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    rij = sqrt( (-x0+y0)**2 + (-x1+y1)**2 + (-x2+y2)**2 )
    
    Dmorse_Dxi0 = epsilon*(2*a*(x0 - y0)*exp(-\
        a*(-l0 + rij))/rij - 2*a*(x0 - y0)*exp(-2*a*(-l0 +\
         rij))/rij)

    Dmorse_Dxi1 = epsilon*(2*a*(x1 - y1)*exp(-a*(-l0 +\
         rij))/rij - 2*a*(x1 - y1)*exp(-2*a*(-l0 +\
         rij))/rij)

    Dmorse_Dxi2 = epsilon*(2*a*(x2 - y2)*exp(-a*(-l0 +\
         rij))/rij - 2*a*(x2 - y2)*exp(-2*a*(-l0 + rij))/rij)
    
    return array([Dmorse_Dxi0,Dmorse_Dxi1,Dmorse_Dxi2])

"""
Derivative of Morse potential wrt xj
"""
@jit( f8[:](f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dmorse_Dxj(xi, xj, epsilon, l0, a):
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    rij = sqrt( (-x0+y0)**2 + (-x1+y1)**2 + (-x2+y2)**2 )
    
    Dmorse_Dxj0 = epsilon*(2*a*(-x0 +\
         y0)*exp(-a*(-l0 + rij))/rij - 2*a*(-x0 + y0)*exp(-2*a*(-l0 +\
         rij))/rij)

    Dmorse_Dxj1 = epsilon*(2*a*(-x1 + y1)*exp(-a*(-l0 +\
         rij))/rij - 2*a*(-x1 + y1)*exp(-2*a*(-l0 +\
         rij))/rij)

    Dmorse_Dxj2 = epsilon*(2*a*(-x2 + y2)*exp(-a*(-l0 +\
         rij))/rij - 2*a*(-x2 + y2)*exp(-2*a*(-l0 + rij))/rij)    

    return array([Dmorse_Dxj0,Dmorse_Dxj1,Dmorse_Dxj2])

"""
Derivative of co-planarity potential wrt xi
"""
@jit( f8[:](f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_pDxi(vi, xi, xj):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
        
    Dphi_pDxi0 = (-\
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag -\
         4*sin_alpha_i**2*u0*u2/vi_mag**2)*((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))

    Dphi_pDxi1 = (4*cos_alpha_i\
        *sin_alpha_i*u0/vi_mag - 4*sin_alpha_i**2*u1*u2/vi_mag**2)*((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))

    Dphi_pDxi2 = ((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))*(-2*cos_alpha_i**2 +\
         2*sin_alpha_i**2*u0**2/vi_mag**2 + 2*sin_alpha_i**2*u1**2/vi_mag**2 -\
         2*sin_alpha_i**2*u2**2/vi_mag**2)
        
    return array([Dphi_pDxi0,Dphi_pDxi1,Dphi_pDxi2])

"""
Derivative of co-planarity potential wrt xj
"""
@jit( f8[:](f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_pDxj(vi, xi, xj):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    
    Dphi_pDxj0 = (4*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         4*sin_alpha_i**2*u0*u2/vi_mag**2)*((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))

    Dphi_pDxj1 = (-\
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         4*sin_alpha_i**2*u1*u2/vi_mag**2)*((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))

    Dphi_pDxj2 = ((-x0 +\
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         2*sin_alpha_i**2*u0*u2/vi_mag**2) + (-x1 + y1)*(-\
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         2*sin_alpha_i**2*u1*u2/vi_mag**2) + (-x2 + y2)*(cos_alpha_i**2 -\
         sin_alpha_i**2*u0**2/vi_mag**2 - sin_alpha_i**2*u1**2/vi_mag**2 +\
         sin_alpha_i**2*u2**2/vi_mag**2))*(2*cos_alpha_i**2 -\
         2*sin_alpha_i**2*u0**2/vi_mag**2 - 2*sin_alpha_i**2*u1**2/vi_mag**2 +\
         2*sin_alpha_i**2*u2**2/vi_mag**2)

    return array([Dphi_pDxj0,Dphi_pDxj1,Dphi_pDxj2])

"""
Derivative of co-circularity potential wrt xi
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_cDxi(vi, vj, xi, xj):
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
    
    Dphi_cDxi0 = ((-x0 +\
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
         sin_alpha_j**2*v2**2/vj_mag**2))*(-\
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag -\
         4*cos_alpha_j*sin_alpha_j*v1/vj_mag -\
         4*sin_alpha_i**2*u0*u2/vi_mag**2 -\
         4*sin_alpha_j**2*v0*v2/vj_mag**2)

    Dphi_cDxi1 = ((-x0 +\
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
         sin_alpha_j**2*v2**2/vj_mag**2))*(4*cos_alpha_i*sin_alpha_i*\
        u0/vi_mag + 4*cos_alpha_j*sin_alpha_j*v0/vj_mag -\
         4*sin_alpha_i**2*u1*u2/vi_mag**2 -\
         4*sin_alpha_j**2*v1*v2/vj_mag**2)

    Dphi_cDxi2 = ((-x0 +\
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
         sin_alpha_j**2*v2**2/vj_mag**2))*(-2*cos_alpha_i**2 -\
         2*cos_alpha_j**2 + 2*sin_alpha_i**2*u0**2/vi_mag**2 +\
         2*sin_alpha_i**2*u1**2/vi_mag**2 - 2*sin_alpha_i**2*u2**2/vi_mag**2 +\
         2*sin_alpha_j**2*v0**2/vj_mag**2 + 2*sin_alpha_j**2*v1**2/vj_mag**2 -\
         2*sin_alpha_j**2*v2**2/vj_mag**2)
 
    return array([Dphi_cDxi0,Dphi_cDxi1,Dphi_cDxi2])

"""
Derivative of co-circularity potential wrt xj
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_cDxj(vi, vj, xi, xj):
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
    
    Dphi_cDxj0 = ((-x0 +\
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
         sin_alpha_j**2*v2**2/vj_mag**2))*(4*cos_alpha_i*sin_alpha_i*\
        u1/vi_mag + 4*cos_alpha_j*sin_alpha_j*v1/vj_mag +\
         4*sin_alpha_i**2*u0*u2/vi_mag**2 +\
         4*sin_alpha_j**2*v0*v2/vj_mag**2)

    Dphi_cDxj1 = ((-x0 +\
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
         sin_alpha_j**2*v2**2/vj_mag**2))*(-\
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag -\
         4*cos_alpha_j*sin_alpha_j*v0/vj_mag +\
         4*sin_alpha_i**2*u1*u2/vi_mag**2 +\
         4*sin_alpha_j**2*v1*v2/vj_mag**2)

    Dphi_cDxj2 = ((-x0 +\
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
         sin_alpha_j**2*v2**2/vj_mag**2))*(2*cos_alpha_i**2 +\
         2*cos_alpha_j**2 - 2*sin_alpha_i**2*u0**2/vi_mag**2 -\
         2*sin_alpha_i**2*u1**2/vi_mag**2 + 2*sin_alpha_i**2*u2**2/vi_mag**2 -\
         2*sin_alpha_j**2*v0**2/vj_mag**2 - 2*sin_alpha_j**2*v1**2/vj_mag**2 +\
         2*sin_alpha_j**2*v2**2/vj_mag**2)

    return array([Dphi_cDxj0,Dphi_cDxj1,Dphi_cDxj2])