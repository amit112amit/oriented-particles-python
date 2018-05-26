#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:06:30 2017
    Numerical Jacobian of Kernel
@author: amit
"""
from numba import jit, f8
from numpy import array, exp, sqrt, sin, cos

"""
 Derivative of kernel with respect to xi
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dpsi_Dxi(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj

    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    cos_alpha_i = cos( 0.5*vi_mag )

    Dpsi_Dxi0 = K*(-\
        (4*cos_alpha_i*sin_alpha_i*u1/vi_mag -\
         4*sin_alpha_i**2*u0*u2/vi_mag**2)*(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-(-\
        4*cos_alpha_i*sin_alpha_i*u2/vi_mag -\
         4*sin_alpha_i**2*u0*u1/vi_mag**2)*(cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2) - (-2*cos_alpha_i**2 -\
         2*sin_alpha_i**2*u0**2/vi_mag**2 + 2*sin_alpha_i**2*u1**2/vi_mag**2 +\
         2*sin_alpha_i**2*u2**2/vi_mag**2)*(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2))/(2*a**2))*exp(-\
        (cos_alpha_i**2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +\
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
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
         sin_alpha_i**2*u2**2*(-x1 +\
         y1)/vi_mag**2)**2)/(2*a**2))

    Dpsi_Dxi1 = K*(-(-\
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag -\
         4*sin_alpha_i**2*u1*u2/vi_mag**2)*(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-\
        (4*cos_alpha_i*sin_alpha_i*u2/vi_mag -\
         4*sin_alpha_i**2*u0*u1/vi_mag**2)*(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2) - (-2*cos_alpha_i**2 +\
         2*sin_alpha_i**2*u0**2/vi_mag**2 - 2*sin_alpha_i**2*u1**2/vi_mag**2 +\
         2*sin_alpha_i**2*u2**2/vi_mag**2)*(cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2))/(2*a**2))*exp(-\
        (cos_alpha_i**2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +\
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
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
         sin_alpha_i**2*u2**2*(-x1 +\
         y1)/vi_mag**2)**2)/(2*a**2))

    Dpsi_Dxi2 = K*(-(-2*cos_alpha_i**2 +\
         2*sin_alpha_i**2*u0**2/vi_mag**2 + 2*sin_alpha_i**2*u1**2/vi_mag**2 -\
         2*sin_alpha_i**2*u2**2/vi_mag**2)*(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-\
        (4*cos_alpha_i*sin_alpha_i*u0/vi_mag -\
         4*sin_alpha_i**2*u1*u2/vi_mag**2)*(cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2) - (-\
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag -\
         4*sin_alpha_i**2*u0*u2/vi_mag**2)*(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2))/(2*a**2))*exp(-\
        (cos_alpha_i**2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +\
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
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

    return array([Dpsi_Dxi0, Dpsi_Dxi1, Dpsi_Dxi2])

"""
 Derivative of kernel with respect to xj
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dpsi_Dxj(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj

    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    cos_alpha_i = cos( 0.5*vi_mag )

    Dpsi_Dxj0 = K*(-(-\
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         4*sin_alpha_i**2*u0*u2/vi_mag**2)*(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-\
        (4*cos_alpha_i*sin_alpha_i*u2/vi_mag +\
         4*sin_alpha_i**2*u0*u1/vi_mag**2)*(cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2) - (2*cos_alpha_i**2 +\
         2*sin_alpha_i**2*u0**2/vi_mag**2 - 2*sin_alpha_i**2*u1**2/vi_mag**2 -\
         2*sin_alpha_i**2*u2**2/vi_mag**2)*(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2))/(2*a**2))*exp(-\
        (cos_alpha_i**2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +\
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
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
         sin_alpha_i**2*u2**2*(-x1 +\
         y1)/vi_mag**2)**2)/(2*a**2))

    Dpsi_Dxj1 = K*(-\
        (4*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         4*sin_alpha_i**2*u1*u2/vi_mag**2)*(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-(-\
        4*cos_alpha_i*sin_alpha_i*u2/vi_mag +\
         4*sin_alpha_i**2*u0*u1/vi_mag**2)*(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2) - (2*cos_alpha_i**2 -\
         2*sin_alpha_i**2*u0**2/vi_mag**2 + 2*sin_alpha_i**2*u1**2/vi_mag**2 -\
         2*sin_alpha_i**2*u2**2/vi_mag**2)*(cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2))/(2*a**2))*exp(-\
        (cos_alpha_i**2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +\
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
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
         sin_alpha_i**2*u2**2*(-x1 +\
         y1)/vi_mag**2)**2)/(2*a**2))

    Dpsi_Dxj2 = K*(-(2*cos_alpha_i**2 -\
         2*sin_alpha_i**2*u0**2/vi_mag**2 - 2*sin_alpha_i**2*u1**2/vi_mag**2 +\
         2*sin_alpha_i**2*u2**2/vi_mag**2)*(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-(-\
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag +\
         4*sin_alpha_i**2*u1*u2/vi_mag**2)*(cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2) -\
         (4*cos_alpha_i*sin_alpha_i*u1/vi_mag +\
         4*sin_alpha_i**2*u0*u2/vi_mag**2)*(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2))/(2*a**2))*exp(-\
        (cos_alpha_i**2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +\
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
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

    return array([Dpsi_Dxj0, Dpsi_Dxj1, Dpsi_Dxj2])

"""
 Derivative of kernel with respect to qi
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dpsi_Dvi(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj

    vi_mag_sqr = u0**2 + u1**2 + u2**2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    cos_alpha_i = cos( 0.5*vi_mag )

    Dpsi_Du0 = K*(-(cos_alpha_i**2*(-x2 + y2) +\
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)*(cos_alpha_i**2*u0**2*(-\
        2*x1 + 2*y1)/vi_mag**2 + cos_alpha_i**2*u0*u1*(2*x0 -\
         2*y0)/vi_mag**2 - 2*cos_alpha_i*sin_alpha_i*u0**3*(-x2 +\
         y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0**2*u2*(-2*x0 +\
         2*y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0**2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u1**2*(-x2 +\
         y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u1*(2*x0 -\
         2*y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u2**2*(-x2 +\
         y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*(-x2 + y2)/vi_mag +\
         2*cos_alpha_i*sin_alpha_i*(-2*x1 + 2*y1)/vi_mag +\
         4*sin_alpha_i**2*u0**3*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u0**2*u2*(-2*x0 + 2*y0)/vi_mag**4 -\
         sin_alpha_i**2*u0**2*(-2*x1 + 2*y1)/vi_mag**2 +\
         4*sin_alpha_i**2*u0*u1**2*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1*u2*(-2*x1 + 2*y1)/vi_mag**4 -\
         sin_alpha_i**2*u0*u1*(2*x0 - 2*y0)/vi_mag**2 -\
         4*sin_alpha_i**2*u0*u2**2*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*(-x2 + y2)/vi_mag**2 + 2*sin_alpha_i**2*u2*(-\
        2*x0 + 2*y0)/vi_mag**2)/(2*b**2) + (-(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2)*(cos_alpha_i**2*u0*u1*(-\
        2*x2 + 2*y2)/vi_mag**2 - cos_alpha_i**2*u0*u2*(-2*x1 +\
         2*y1)/vi_mag**2 + 2*cos_alpha_i*sin_alpha_i*u0**3*(-x0 +\
         y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0**2*u1*(-2*x1 +\
         2*y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0**2*u2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u1**2*(-x0 +\
         y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u1*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u2**2*(-x0 +\
         y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*(-x0 + y0)/vi_mag -\
         4*sin_alpha_i**2*u0**3*(-x0 + y0)/vi_mag**4 -\
         4*sin_alpha_i**2*u0**2*u1*(-2*x1 + 2*y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0**2*u2*(-2*x2 + 2*y2)/vi_mag**4 +\
         4*sin_alpha_i**2*u0*u1**2*(-x0 + y0)/vi_mag**4 -\
         sin_alpha_i**2*u0*u1*(-2*x2 + 2*y2)/vi_mag**2 +\
         4*sin_alpha_i**2*u0*u2**2*(-x0 + y0)/vi_mag**4 +\
         sin_alpha_i**2*u0*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         4*sin_alpha_i**2*u0*(-x0 + y0)/vi_mag**2 + 2*sin_alpha_i**2*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + 2*sin_alpha_i**2*u2*(-2*x2 +\
         2*y2)/vi_mag**2) - (cos_alpha_i**2*(-x1 + y1) -\
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2)*(-cos_alpha_i**2*u0**2*(-\
        2*x2 + 2*y2)/vi_mag**2 + cos_alpha_i**2*u0*u2*(-2*x0 +\
         2*y0)/vi_mag**2 - 2*cos_alpha_i*sin_alpha_i*u0**3*(-x1 +\
         y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0**2*u1*(-2*x0 +\
         2*y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0**2*(-2*x2 +\
         2*y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1**2*(-x1 +\
         y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u2**2*(-x1 +\
         y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x0 +\
         2*y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*(-x1 + y1)/vi_mag -\
         2*cos_alpha_i*sin_alpha_i*(-2*x2 + 2*y2)/vi_mag +\
         4*sin_alpha_i**2*u0**3*(-x1 + y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0**2*u1*(-2*x0 + 2*y0)/vi_mag**4 +\
         sin_alpha_i**2*u0**2*(-2*x2 + 2*y2)/vi_mag**2 -\
         4*sin_alpha_i**2*u0*u1**2*(-x1 + y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1*u2*(-2*x2 + 2*y2)/vi_mag**4 +\
         4*sin_alpha_i**2*u0*u2**2*(-x1 + y1)/vi_mag**4 -\
         sin_alpha_i**2*u0*u2*(-2*x0 + 2*y0)/vi_mag**2 -\
         4*sin_alpha_i**2*u0*(-x1 + y1)/vi_mag**2 + 2*sin_alpha_i**2*u1*(-\
        2*x0 + 2*y0)/vi_mag**2))/(2*a**2))*exp(-(cos_alpha_i**2*(-x2 + y2) +\
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
         sin_alpha_i**2*u2**2*(-x1 +\
         y1)/vi_mag**2)**2)/(2*a**2))

    Dpsi_Du1 = K*(-(cos_alpha_i**2*(-\
        x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)*(cos_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + cos_alpha_i**2*u1**2*(2*x0 -\
         2*y0)/vi_mag**2 - 2*cos_alpha_i*sin_alpha_i*u0**2*u1*(-x2 +\
         y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x0 +\
         2*y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u1*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1**3*(-x2 +\
         y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1**2*u2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1**2*(2*x0 -\
         2*y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1*u2**2*(-x2 +\
         y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*(-x2 + y2)/vi_mag +\
         2*cos_alpha_i*sin_alpha_i*(2*x0 - 2*y0)/vi_mag +\
         4*sin_alpha_i**2*u0**2*u1*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1*u2*(-2*x0 + 2*y0)/vi_mag**4 -\
         sin_alpha_i**2*u0*u1*(-2*x1 + 2*y1)/vi_mag**2 +\
         4*sin_alpha_i**2*u1**3*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u1**2*u2*(-2*x1 + 2*y1)/vi_mag**4 -\
         sin_alpha_i**2*u1**2*(2*x0 - 2*y0)/vi_mag**2 -\
         4*sin_alpha_i**2*u1*u2**2*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u1*(-x2 + y2)/vi_mag**2 + 2*sin_alpha_i**2*u2*(-\
        2*x1 + 2*y1)/vi_mag**2)/(2*b**2) + (-(cos_alpha_i**2*(-x0 + y0) +\
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -\
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2)*(cos_alpha_i**2*u1**2*(-\
        2*x2 + 2*y2)/vi_mag**2 - cos_alpha_i**2*u1*u2*(-2*x1 +\
         2*y1)/vi_mag**2 + 2*cos_alpha_i*sin_alpha_i*u0**2*u1*(-x0 +\
         y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1**2*(-2*x1 +\
         2*y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1**3*(-x0 +\
         y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1**2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*u2**2*(-x0 +\
         y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1*u2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*(-x0 + y0)/vi_mag +\
         2*cos_alpha_i*sin_alpha_i*(-2*x2 + 2*y2)/vi_mag -\
         4*sin_alpha_i**2*u0**2*u1*(-x0 + y0)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1**2*(-2*x1 + 2*y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1*u2*(-2*x2 + 2*y2)/vi_mag**4 +\
         2*sin_alpha_i**2*u0*(-2*x1 + 2*y1)/vi_mag**2 +\
         4*sin_alpha_i**2*u1**3*(-x0 + y0)/vi_mag**4 - sin_alpha_i**2*u1**2*(-\
        2*x2 + 2*y2)/vi_mag**2 + 4*sin_alpha_i**2*u1*u2**2*(-x0 +\
         y0)/vi_mag**4 + sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 -\
         4*sin_alpha_i**2*u1*(-x0 + y0)/vi_mag**2) - (cos_alpha_i**2*(-x1 +\
         y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2)*(-cos_alpha_i**2*u0*u1*(-\
        2*x2 + 2*y2)/vi_mag**2 + cos_alpha_i**2*u1*u2*(-2*x0 +\
         2*y0)/vi_mag**2 - 2*cos_alpha_i*sin_alpha_i*u0**2*u1*(-x1 +\
         y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1**2*(-2*x0 +\
         2*y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*(-2*x2 +\
         2*y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1**3*(-x1 +\
         y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1**2*u2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*u2**2*(-x1 +\
         y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*u2*(-2*x0 +\
         2*y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*(-x1 + y1)/vi_mag +\
         4*sin_alpha_i**2*u0**2*u1*(-x1 + y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1**2*(-2*x0 + 2*y0)/vi_mag**4 +\
         sin_alpha_i**2*u0*u1*(-2*x2 + 2*y2)/vi_mag**2 +\
         2*sin_alpha_i**2*u0*(-2*x0 + 2*y0)/vi_mag**2 -\
         4*sin_alpha_i**2*u1**3*(-x1 + y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u1**2*u2*(-2*x2 + 2*y2)/vi_mag**4 +\
         4*sin_alpha_i**2*u1*u2**2*(-x1 + y1)/vi_mag**4 -\
         sin_alpha_i**2*u1*u2*(-2*x0 + 2*y0)/vi_mag**2 +\
         4*sin_alpha_i**2*u1*(-x1 + y1)/vi_mag**2 + 2*sin_alpha_i**2*u2*(-\
        2*x2 + 2*y2)/vi_mag**2))/(2*a**2))*exp(-(cos_alpha_i**2*(-x2 + y2) +\
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
         sin_alpha_i**2*u2**2*(-x1 +\
         y1)/vi_mag**2)**2)/(2*a**2))

    Dpsi_Du2 = K*(-(cos_alpha_i**2*(-\
        x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +\
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x2 + y2)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-\
        2*x0 + 2*y0)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x2 + y2)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         sin_alpha_i**2*u2**2*(-x2 + y2)/vi_mag**2)*(cos_alpha_i**2*u0*u2*(-\
        2*x1 + 2*y1)/vi_mag**2 + cos_alpha_i**2*u1*u2*(2*x0 -\
         2*y0)/vi_mag**2 - 2*cos_alpha_i*sin_alpha_i*u0**2*u2*(-x2 +\
         y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u2**2*(-2*x0 +\
         2*y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1**2*u2*(-x2 +\
         y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1*u2**2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*u2*(2*x0 -\
         2*y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u2**3*(-x2 +\
         y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u2*(-x2 + y2)/vi_mag +\
         4*sin_alpha_i**2*u0**2*u2*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u2**2*(-2*x0 + 2*y0)/vi_mag**4 -\
         sin_alpha_i**2*u0*u2*(-2*x1 + 2*y1)/vi_mag**2 +\
         2*sin_alpha_i**2*u0*(-2*x0 + 2*y0)/vi_mag**2 +\
         4*sin_alpha_i**2*u1**2*u2*(-x2 + y2)/vi_mag**4 -\
         4*sin_alpha_i**2*u1*u2**2*(-2*x1 + 2*y1)/vi_mag**4 -\
         sin_alpha_i**2*u1*u2*(2*x0 - 2*y0)/vi_mag**2 + 2*sin_alpha_i**2*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 - 4*sin_alpha_i**2*u2**3*(-x2 + y2)/vi_mag**4 +\
         4*sin_alpha_i**2*u2*(-x2 + y2)/vi_mag**2)/(2*b**2) + (-\
        (cos_alpha_i**2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +\
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +\
         sin_alpha_i**2*u0**2*(-x0 + y0)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x1 + 2*y1)/vi_mag**2 + sin_alpha_i**2*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**2 - sin_alpha_i**2*u1**2*(-x0 + y0)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x0 + y0)/vi_mag**2)*(cos_alpha_i**2*u1*u2*(-\
        2*x2 + 2*y2)/vi_mag**2 - cos_alpha_i**2*u2**2*(-2*x1 +\
         2*y1)/vi_mag**2 + 2*cos_alpha_i*sin_alpha_i*u0**2*u2*(-x0 +\
         y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x1 +\
         2*y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u2**2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1**2*u2*(-x0 +\
         y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u1*u2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u2**3*(-x0 +\
         y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u2**2*(-2*x1 +\
         2*y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u2*(-x0 + y0)/vi_mag -\
         2*cos_alpha_i*sin_alpha_i*(-2*x1 + 2*y1)/vi_mag -\
         4*sin_alpha_i**2*u0**2*u2*(-x0 + y0)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1*u2*(-2*x1 + 2*y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u2**2*(-2*x2 + 2*y2)/vi_mag**4 +\
         2*sin_alpha_i**2*u0*(-2*x2 + 2*y2)/vi_mag**2 +\
         4*sin_alpha_i**2*u1**2*u2*(-x0 + y0)/vi_mag**4 -\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 +\
         4*sin_alpha_i**2*u2**3*(-x0 + y0)/vi_mag**4 + sin_alpha_i**2*u2**2*(-\
        2*x1 + 2*y1)/vi_mag**2 - 4*sin_alpha_i**2*u2*(-x0 + y0)/vi_mag**2) -\
         (cos_alpha_i**2*(-x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 +\
         2*y2)/vi_mag + cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -\
         sin_alpha_i**2*u0**2*(-x1 + y1)/vi_mag**2 + sin_alpha_i**2*u0*u1*(-\
        2*x0 + 2*y0)/vi_mag**2 + sin_alpha_i**2*u1**2*(-x1 + y1)/vi_mag**2 +\
         sin_alpha_i**2*u1*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         sin_alpha_i**2*u2**2*(-x1 + y1)/vi_mag**2)*(-cos_alpha_i**2*u0*u2*(-\
        2*x2 + 2*y2)/vi_mag**2 + cos_alpha_i**2*u2**2*(-2*x0 +\
         2*y0)/vi_mag**2 - 2*cos_alpha_i*sin_alpha_i*u0**2*u2*(-x1 +\
         y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x0 +\
         2*y0)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x2 +\
         2*y2)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1**2*u2*(-x1 +\
         y1)/vi_mag**3 + 2*cos_alpha_i*sin_alpha_i*u1*u2**2*(-2*x2 +\
         2*y2)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u2**3*(-x1 +\
         y1)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u2**2*(-2*x0 +\
         2*y0)/vi_mag**3 - 2*cos_alpha_i*sin_alpha_i*u2*(-x1 + y1)/vi_mag +\
         2*cos_alpha_i*sin_alpha_i*(-2*x0 + 2*y0)/vi_mag +\
         4*sin_alpha_i**2*u0**2*u2*(-x1 + y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u0*u1*u2*(-2*x0 + 2*y0)/vi_mag**4 +\
         sin_alpha_i**2*u0*u2*(-2*x2 + 2*y2)/vi_mag**2 -\
         4*sin_alpha_i**2*u1**2*u2*(-x1 + y1)/vi_mag**4 -\
         4*sin_alpha_i**2*u1*u2**2*(-2*x2 + 2*y2)/vi_mag**4 +\
         2*sin_alpha_i**2*u1*(-2*x2 + 2*y2)/vi_mag**2 +\
         4*sin_alpha_i**2*u2**3*(-x1 + y1)/vi_mag**4 - sin_alpha_i**2*u2**2*(-\
        2*x0 + 2*y0)/vi_mag**2 - 4*sin_alpha_i**2*u2*(-x1 +\
         y1)/vi_mag**2))/(2*a**2))*exp(-(cos_alpha_i**2*(-x2 + y2) +\
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

    return array([Dpsi_Du0, Dpsi_Du1, Dpsi_Du2])
