#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:18:19 2017
    Numerical values for derivatives of the potentials
@author: amit
"""

from numba import jit, f8
from numpy import sqrt, array, cos, sin

"""
Derivative of co-planarity potential wrt vi
"""
@jit( f8[:](f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_pDvi(vi, xi, xj):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2    
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    
    Dphi_pDu0 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_i_2*u0*u1/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u0*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u0*u1/pow(vi_mag,3) -
         4*sin_alpha_i_2*u0*u0*u2/pow(vi_mag,4) -
         sin_alpha_i_2*u0*u1/vi_mag_sqr + 2*sin_alpha_i_2*u2/vi_mag_sqr) +
         2*(-x1 + y1)*(-cos_alpha_i_2*u0*u0/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u0/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u0*u1*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i/vi_mag + sin_alpha_i_2*u0*u0/vi_mag_sqr -
         4*sin_alpha_i_2*u0*u1*u2/pow(vi_mag,4)) + 2*(-x2 + y2)*(-
        cos_alpha_i*sin_alpha_i*pow(u0,3)/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u0*u1*u1/pow(vi_mag,3) +
         cos_alpha_i*sin_alpha_i*u0*u2*u2/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*pow(u0,3)/pow(vi_mag,4) + 
        2*sin_alpha_i_2*u0*u1*u1/pow(vi_mag,4) -
         2*sin_alpha_i_2*u0*u2*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*u0/vi_mag_sqr))

    Dphi_pDu1 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_i_2*u1*u1/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u1*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u1*u1/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i/vi_mag -
         4*sin_alpha_i_2*u0*u1*u2/pow(vi_mag,4) -
         sin_alpha_i_2*u1*u1/vi_mag_sqr) + 2*(-x1 + y1)*(-
        cos_alpha_i_2*u0*u1/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u1/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u1*u1*u2/pow(vi_mag,3) +
         sin_alpha_i_2*u0*u1/vi_mag_sqr -
         4*sin_alpha_i_2*u1*u1*u2/pow(vi_mag,4) + 2*sin_alpha_i_2*u2/vi_mag_sqr) + 2*(-x2 + y2)*(-
        cos_alpha_i*sin_alpha_i*u0*u0*u1/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*pow(u1,3)/pow(vi_mag,3) +
         cos_alpha_i*sin_alpha_i*u1*u2*u2/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u0*u1/pow(vi_mag,4) +
         2*sin_alpha_i_2*pow(u1,3)/pow(vi_mag,4) -
         2*sin_alpha_i_2*u1*u2*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*u1/vi_mag_sqr))

    Dphi_pDu2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_i_2*u1*u2/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u2*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u1*u2/pow(vi_mag,3) -
         4*sin_alpha_i_2*u0*u2*u2/pow(vi_mag,4) + 2*sin_alpha_i_2*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u2/vi_mag_sqr) + 2*(-x1 + y1)*(-
        cos_alpha_i_2*u0*u2/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u2/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u1*u2*u2/pow(vi_mag,3) +
         sin_alpha_i_2*u0*u2/vi_mag_sqr -
         4*sin_alpha_i_2*u1*u2*u2/pow(vi_mag,4) + 2*sin_alpha_i_2*u1/vi_mag_sqr) + 2*(-x2 + y2)*(-
        cos_alpha_i*sin_alpha_i*u0*u0*u2/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u1*u1*u2/pow(vi_mag,3) +
         cos_alpha_i*sin_alpha_i*pow(u2,3)/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u2/vi_mag +
         2*sin_alpha_i_2*u0*u0*u2/pow(vi_mag,4) +
         2*sin_alpha_i_2*u1*u1*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*pow(u2,3)/pow(vi_mag,4) + 2*sin_alpha_i_2*u2/vi_mag_sqr))
        
    return array([Dphi_pDu0,Dphi_pDu1,Dphi_pDu2])

"""
Derivative of co-normality potential wrt vi
"""
@jit( f8[:](f8[:], f8[:]), cache=True, nopython=True )
def Dphi_nDvi(vi, vj):
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )

    Dphi_nDu0 = (-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr)*(-
        2*cos_alpha_i_2*u0*u0/vi_mag_sqr + 4*
        cos_alpha_i*sin_alpha_i*u0*u0/pow(vi_mag,3) +
         4*cos_alpha_i*sin_alpha_i*u0*u1*u2/pow(vi_mag,3) -
         4*cos_alpha_i*sin_alpha_i/vi_mag + 2*sin_alpha_i_2*u0*u0/vi_mag_sqr -
         8*sin_alpha_i_2*u0*u1*u2/pow(vi_mag,4)) +
         (2*cos_alpha_i*sin_alpha_i*u1/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr)*(2*cos_alpha_i_2*u0*u1/vi_mag_sqr +
         4*cos_alpha_i*sin_alpha_i*u0*u0*u2/pow(vi_mag,3) -
         4*cos_alpha_i*sin_alpha_i*u0*u1/pow(vi_mag,3) -
         8*sin_alpha_i_2*u0*u0*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*u0*u1/vi_mag_sqr + 4*sin_alpha_i_2*u2/vi_mag_sqr) +
         (cos_alpha_i_2 - cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr +
         sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr -
         sin_alpha_j_2*v2*v2/vj_mag_sqr)*(-
        2*cos_alpha_i*sin_alpha_i*pow(u0,3)/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u0*u1*u1/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u0*u2*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         4*sin_alpha_i_2*pow(u0,3)/pow(vi_mag,4) +
         4*sin_alpha_i_2*u0*u1*u1/pow(vi_mag,4) -
         4*sin_alpha_i_2*u0*u2*u2/pow(vi_mag,4) -
         4*sin_alpha_i_2*u0/vi_mag_sqr)

    Dphi_nDu1 = (-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr)*(-
        2*cos_alpha_i_2*u0*u1/vi_mag_sqr + 4*
        cos_alpha_i*sin_alpha_i*u0*u1/pow(vi_mag,3) +
         4*cos_alpha_i*sin_alpha_i*u1*u1*u2/pow(vi_mag,3) +
         2*sin_alpha_i_2*u0*u1/vi_mag_sqr -
         8*sin_alpha_i_2*u1*u1*u2/pow(vi_mag,4) +
         4*sin_alpha_i_2*u2/vi_mag_sqr) + (2*
                                      cos_alpha_i*sin_alpha_i*u1/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr)*(2*cos_alpha_i_2*u1*u1/vi_mag_sqr +
         4*cos_alpha_i*sin_alpha_i*u0*u1*u2/pow(vi_mag,3) -
         4*cos_alpha_i*sin_alpha_i*u1*u1/pow(vi_mag,3) +
         4*cos_alpha_i*sin_alpha_i/vi_mag -
         8*sin_alpha_i_2*u0*u1*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*u1*u1/vi_mag_sqr) + (cos_alpha_i_2 -
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr +
         sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr -
         sin_alpha_j_2*v2*v2/vj_mag_sqr)*(-
        2*cos_alpha_i*sin_alpha_i*u0*u0*u1/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*pow(u1,3)/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u1*u2*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         4*sin_alpha_i_2*u0*u0*u1/pow(vi_mag,4) +
         4*sin_alpha_i_2*pow(u1,3)/pow(vi_mag,4) -
         4*sin_alpha_i_2*u1*u2*u2/pow(vi_mag,4) -
         4*sin_alpha_i_2*u1/vi_mag_sqr)

    Dphi_nDu2 = (-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr)*(-
        2*cos_alpha_i_2*u0*u2/vi_mag_sqr + 4*
        cos_alpha_i*sin_alpha_i*u0*u2/pow(vi_mag,3) +
         4*cos_alpha_i*sin_alpha_i*u1*u2*u2/pow(vi_mag,3) +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         8*sin_alpha_i_2*u1*u2*u2/pow(vi_mag,4) +
         4*sin_alpha_i_2*u1/vi_mag_sqr) + (2*
                                      cos_alpha_i*sin_alpha_i*u1/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr)*(2*cos_alpha_i_2*u1*u2/vi_mag_sqr +
         4*cos_alpha_i*sin_alpha_i*u0*u2*u2/pow(vi_mag,3) -
         4*cos_alpha_i*sin_alpha_i*u1*u2/pow(vi_mag,3) -
         8*sin_alpha_i_2*u0*u2*u2/pow(vi_mag,4) + 4*sin_alpha_i_2*u0/vi_mag_sqr -
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (cos_alpha_i_2 -
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr +
         sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr -
         sin_alpha_j_2*v2*v2/vj_mag_sqr)*(-
        2*cos_alpha_i*sin_alpha_i*u0*u0*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u1*u1*u2/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*pow(u2,3)/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u2/vi_mag +
         4*sin_alpha_i_2*u0*u0*u2/pow(vi_mag,4) +
         4*sin_alpha_i_2*u1*u1*u2/pow(vi_mag,4) -
         4*sin_alpha_i_2*pow(u2,3)/pow(vi_mag,4) + 4*sin_alpha_i_2*u2/vi_mag_sqr)

    return array([Dphi_nDu0,Dphi_nDu1,Dphi_nDu2])

"""
Derivative of co-normality potential wrt vj
"""
@jit( f8[:](f8[:], f8[:]), cache=True, nopython=True )
def Dphi_nDvj(vi, vj):
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )
    
    Dphi_nDv0 = (-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr)*(2*cos_alpha_j_2*v0*v0/vj_mag_sqr -
         4*cos_alpha_j*sin_alpha_j*v0*v0/pow(vj_mag,3) -
         4*cos_alpha_j*sin_alpha_j*v0*v1*v2/pow(vj_mag,3) +
         4*cos_alpha_j*sin_alpha_j/vj_mag - 2*sin_alpha_j_2*v0*v0/vj_mag_sqr +
         8*sin_alpha_j_2*v0*v1*v2/pow(vj_mag,4)) +
         (2*cos_alpha_i*sin_alpha_i*u1/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr)*(-
        2*cos_alpha_j_2*v0*v1/vj_mag_sqr - 4*
        cos_alpha_j*sin_alpha_j*v0*v0*v2/pow(vj_mag,3) +
         4*cos_alpha_j*sin_alpha_j*v0*v1/pow(vj_mag,3) +
         8*sin_alpha_j_2*v0*v0*v2/pow(vj_mag,4) +
         2*sin_alpha_j_2*v0*v1/vj_mag_sqr - 4*sin_alpha_j_2*v2/vj_mag_sqr) +
         (cos_alpha_i_2 - cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr +
         sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr -
         sin_alpha_j_2*v2*v2/vj_mag_sqr)*(2*cos_alpha_j*sin_alpha_j*pow(v0,3)/
        pow(vj_mag,3) + 2*cos_alpha_j*sin_alpha_j*v0*v1*v1/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j*v0*v2*v2/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag -
         4*sin_alpha_j_2*pow(v0,3)/pow(vj_mag,4) -
         4*sin_alpha_j_2*v0*v1*v1/pow(vj_mag,4) +
         4*sin_alpha_j_2*v0*v2*v2/pow(vj_mag,4) +
         4*sin_alpha_j_2*v0/vj_mag_sqr)

    Dphi_nDv1 = (-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr)*(2*cos_alpha_j_2*v0*v1/vj_mag_sqr -
         4*cos_alpha_j*sin_alpha_j*v0*v1/pow(vj_mag,3) -
         4*cos_alpha_j*sin_alpha_j*v1*v1*v2/pow(vj_mag,3) -
         2*sin_alpha_j_2*v0*v1/vj_mag_sqr +
         8*sin_alpha_j_2*v1*v1*v2/pow(vj_mag,4) -
         4*sin_alpha_j_2*v2/vj_mag_sqr) + (2*
                                      cos_alpha_i*sin_alpha_i*u1/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr)*(-
        2*cos_alpha_j_2*v1*v1/vj_mag_sqr - 4*
        cos_alpha_j*sin_alpha_j*v0*v1*v2/pow(vj_mag,3) +
         4*cos_alpha_j*sin_alpha_j*v1*v1/pow(vj_mag,3) -
         4*cos_alpha_j*sin_alpha_j/vj_mag +
         8*sin_alpha_j_2*v0*v1*v2/pow(vj_mag,4) +
         2*sin_alpha_j_2*v1*v1/vj_mag_sqr) + (cos_alpha_i_2 -
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr +
         sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr -
         sin_alpha_j_2*v2*v2/vj_mag_sqr)*(2*cos_alpha_j*sin_alpha_j*v0*v0*
        v1/pow(vj_mag,3) + 2*cos_alpha_j*sin_alpha_j*pow(v1,3)/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j*v1*v2*v2/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag -
         4*sin_alpha_j_2*v0*v0*v1/pow(vj_mag,4) -
         4*sin_alpha_j_2*pow(v1,3)/pow(vj_mag,4) +
         4*sin_alpha_j_2*v1*v2*v2/pow(vj_mag,4) +
         4*sin_alpha_j_2*v1/vj_mag_sqr)

    Dphi_nDv2 = (-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr)*(2*cos_alpha_j_2*v0*v2/vj_mag_sqr -
         4*cos_alpha_j*sin_alpha_j*v0*v2/pow(vj_mag,3) -
         4*cos_alpha_j*sin_alpha_j*v1*v2*v2/pow(vj_mag,3) -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr +
         8*sin_alpha_j_2*v1*v2*v2/pow(vj_mag,4) -
         4*sin_alpha_j_2*v1/vj_mag_sqr) + (2*
                                      cos_alpha_i*sin_alpha_i*u1/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr)*(-
        2*cos_alpha_j_2*v1*v2/vj_mag_sqr - 4*
        cos_alpha_j*sin_alpha_j*v0*v2*v2/pow(vj_mag,3) +
         4*cos_alpha_j*sin_alpha_j*v1*v2/pow(vj_mag,3) +
         8*sin_alpha_j_2*v0*v2*v2/pow(vj_mag,4) - 4*sin_alpha_j_2*v0/vj_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (cos_alpha_i_2 -
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr +
         sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr -
         sin_alpha_j_2*v2*v2/vj_mag_sqr)*(2*cos_alpha_j*sin_alpha_j*v0*v0*
        v2/pow(vj_mag,3) + 2*cos_alpha_j*sin_alpha_j*v1*v1*v2/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j*pow(v2,3)/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j*v2/vj_mag -
         4*sin_alpha_j_2*v0*v0*v2/pow(vj_mag,4) -
         4*sin_alpha_j_2*v1*v1*v2/pow(vj_mag,4) +
         4*sin_alpha_j_2*pow(v2,3)/pow(vj_mag,4) - 4*sin_alpha_j_2*v2/vj_mag_sqr)    
    
    return array([Dphi_nDv0,Dphi_nDv1,Dphi_nDv2])

"""
Derivative of co-circularity potential wrt vi
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_cDvi(vi, vj, xi, xj):
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )
    
    Dphi_cDu0 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_i_2*u0*u1/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u0*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u0*u1/pow(vi_mag,3) -
         4*sin_alpha_i_2*u0*u0*u2/pow(vi_mag,4) -
         sin_alpha_i_2*u0*u1/vi_mag_sqr + 2*sin_alpha_i_2*u2/vi_mag_sqr) +
         2*(-x1 + y1)*(-cos_alpha_i_2*u0*u0/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u0/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u0*u1*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i/vi_mag + sin_alpha_i_2*u0*u0/vi_mag_sqr -
         4*sin_alpha_i_2*u0*u1*u2/pow(vi_mag,4)) + 2*(-x2 + y2)*(-
        cos_alpha_i*sin_alpha_i*pow(u0,3)/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u0*u1*u1/pow(vi_mag,3) +
         cos_alpha_i*sin_alpha_i*u0*u2*u2/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*pow(u0,3)/pow(vi_mag,4) + 
        2*sin_alpha_i_2*u0*u1*u1/pow(vi_mag,4) -
         2*sin_alpha_i_2*u0*u2*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*u0/vi_mag_sqr))

    Dphi_cDu1 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_i_2*u1*u1/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u1*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u1*u1/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i/vi_mag -
         4*sin_alpha_i_2*u0*u1*u2/pow(vi_mag,4) -
         sin_alpha_i_2*u1*u1/vi_mag_sqr) + 2*(-x1 + y1)*(-
        cos_alpha_i_2*u0*u1/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u1/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u1*u1*u2/pow(vi_mag,3) +
         sin_alpha_i_2*u0*u1/vi_mag_sqr -
         4*sin_alpha_i_2*u1*u1*u2/pow(vi_mag,4) + 2*sin_alpha_i_2*u2/vi_mag_sqr) + 2*(-x2 + y2)*(-
        cos_alpha_i*sin_alpha_i*u0*u0*u1/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*pow(u1,3)/pow(vi_mag,3) +
         cos_alpha_i*sin_alpha_i*u1*u2*u2/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u0*u1/pow(vi_mag,4) +
         2*sin_alpha_i_2*pow(u1,3)/pow(vi_mag,4) -
         2*sin_alpha_i_2*u1*u2*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*u1/vi_mag_sqr))

    Dphi_cDu2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_i_2*u1*u2/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u2*u2/pow(vi_mag,3) -
         2*cos_alpha_i*sin_alpha_i*u1*u2/pow(vi_mag,3) -
         4*sin_alpha_i_2*u0*u2*u2/pow(vi_mag,4) + 2*sin_alpha_i_2*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u2/vi_mag_sqr) + 2*(-x1 + y1)*(-
        cos_alpha_i_2*u0*u2/vi_mag_sqr +
         2*cos_alpha_i*sin_alpha_i*u0*u2/pow(vi_mag,3) +
         2*cos_alpha_i*sin_alpha_i*u1*u2*u2/pow(vi_mag,3) +
         sin_alpha_i_2*u0*u2/vi_mag_sqr -
         4*sin_alpha_i_2*u1*u2*u2/pow(vi_mag,4) + 2*sin_alpha_i_2*u1/vi_mag_sqr) + 2*(-x2 + y2)*(-
        cos_alpha_i*sin_alpha_i*u0*u0*u2/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u1*u1*u2/pow(vi_mag,3) +
         cos_alpha_i*sin_alpha_i*pow(u2,3)/pow(vi_mag,3) -
         cos_alpha_i*sin_alpha_i*u2/vi_mag +
         2*sin_alpha_i_2*u0*u0*u2/pow(vi_mag,4) +
         2*sin_alpha_i_2*u1*u1*u2/pow(vi_mag,4) -
         2*sin_alpha_i_2*pow(u2,3)/pow(vi_mag,4) + 2*sin_alpha_i_2*u2/vi_mag_sqr))    
    
    return array([Dphi_cDu0,Dphi_cDu1,Dphi_cDu2])

"""
Derivative of co-circularity potential wrt vj
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8[:]), cache=True, nopython=True )
def Dphi_cDvj(vi, vj, xi, xj):
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )

    Dphi_cDv0 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_j_2*v0*v1/vj_mag_sqr +
         2*cos_alpha_j*sin_alpha_j*v0*v0*v2/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j*v0*v1/pow(vj_mag,3) -
         4*sin_alpha_j_2*v0*v0*v2/pow(vj_mag,4) -
         sin_alpha_j_2*v0*v1/vj_mag_sqr + 2*sin_alpha_j_2*v2/vj_mag_sqr) +
         2*(-x1 + y1)*(-cos_alpha_j_2*v0*v0/vj_mag_sqr +
         2*cos_alpha_j*sin_alpha_j*v0*v0/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j*v0*v1*v2/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j/vj_mag + sin_alpha_j_2*v0*v0/vj_mag_sqr -
         4*sin_alpha_j_2*v0*v1*v2/pow(vj_mag,4)) + 2*(-x2 + y2)*(-
        cos_alpha_j*sin_alpha_j*pow(v0,3)/pow(vj_mag,3) -
         cos_alpha_j*sin_alpha_j*v0*v1*v1/pow(vj_mag,3) +
         cos_alpha_j*sin_alpha_j*v0*v2*v2/pow(vj_mag,3) -
         cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_j_2*pow(v0,3)/pow(vj_mag,4) + 
        2*sin_alpha_j_2*v0*v1*v1/pow(vj_mag,4) -
         2*sin_alpha_j_2*v0*v2*v2/pow(vj_mag,4) -
         2*sin_alpha_j_2*v0/vj_mag_sqr))

    Dphi_cDv1 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_j_2*v1*v1/vj_mag_sqr +
         2*cos_alpha_j*sin_alpha_j*v0*v1*v2/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j*v1*v1/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j/vj_mag -
         4*sin_alpha_j_2*v0*v1*v2/pow(vj_mag,4) -
         sin_alpha_j_2*v1*v1/vj_mag_sqr) + 2*(-x1 + y1)*(-
        cos_alpha_j_2*v0*v1/vj_mag_sqr +
         2*cos_alpha_j*sin_alpha_j*v0*v1/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j*v1*v1*v2/pow(vj_mag,3) +
         sin_alpha_j_2*v0*v1/vj_mag_sqr -
         4*sin_alpha_j_2*v1*v1*v2/pow(vj_mag,4) + 2*sin_alpha_j_2*v2/vj_mag_sqr) + 2*(-x2 + y2)*(-
        cos_alpha_j*sin_alpha_j*v0*v0*v1/pow(vj_mag,3) -
         cos_alpha_j*sin_alpha_j*pow(v1,3)/pow(vj_mag,3) +
         cos_alpha_j*sin_alpha_j*v1*v2*v2/pow(vj_mag,3) -
         cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_j_2*v0*v0*v1/pow(vj_mag,4) +
         2*sin_alpha_j_2*pow(v1,3)/pow(vj_mag,4) -
         2*sin_alpha_j_2*v1*v2*v2/pow(vj_mag,4) -
         2*sin_alpha_j_2*v1/vj_mag_sqr))

    Dphi_cDv2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*(-x0 +
         y0)*(cos_alpha_j_2*v1*v2/vj_mag_sqr +
         2*cos_alpha_j*sin_alpha_j*v0*v2*v2/pow(vj_mag,3) -
         2*cos_alpha_j*sin_alpha_j*v1*v2/pow(vj_mag,3) -
         4*sin_alpha_j_2*v0*v2*v2/pow(vj_mag,4) + 2*sin_alpha_j_2*v0/vj_mag_sqr -
         sin_alpha_j_2*v1*v2/vj_mag_sqr) + 2*(-x1 + y1)*(-
        cos_alpha_j_2*v0*v2/vj_mag_sqr +
         2*cos_alpha_j*sin_alpha_j*v0*v2/pow(vj_mag,3) +
         2*cos_alpha_j*sin_alpha_j*v1*v2*v2/pow(vj_mag,3) +
         sin_alpha_j_2*v0*v2/vj_mag_sqr -
         4*sin_alpha_j_2*v1*v2*v2/pow(vj_mag,4) + 2*sin_alpha_j_2*v1/vj_mag_sqr) + 2*(-x2 + y2)*(-
        cos_alpha_j*sin_alpha_j*v0*v0*v2/pow(vj_mag,3) -
         cos_alpha_j*sin_alpha_j*v1*v1*v2/pow(vj_mag,3) +
         cos_alpha_j*sin_alpha_j*pow(v2,3)/pow(vj_mag,3) -
         cos_alpha_j*sin_alpha_j*v2/vj_mag +
         2*sin_alpha_j_2*v0*v0*v2/pow(vj_mag,4) +
         2*sin_alpha_j_2*v1*v1*v2/pow(vj_mag,4) -
         2*sin_alpha_j_2*pow(v2,3)/pow(vj_mag,4) + 2*sin_alpha_j_2*v2/vj_mag_sqr))

    return array([Dphi_cDv0,Dphi_cDv1,Dphi_cDv2])
    
