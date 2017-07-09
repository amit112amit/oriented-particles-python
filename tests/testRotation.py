#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:05:34 2017
    Test conversion between rotation vector and unit normal
@author: amit
"""
import sys
sys.path.insert(0,'../')

import numpy.random as rd
import numpy.linalg as la
import numpy as np
import src.lib.OPS as op

# Create a random 3-D unit normal vector
th = rd.uniform(0.0, 0.5*np.pi)
x,y = rd.random( (2,) )
n_in1 = np.array( [x, np.sin(th), y] )
n_in1 = n_in1/la.norm( n_in1 )

n_in2 = np.array( [x, np.sin(th), -y] )
n_in2 = n_in2/la.norm( n_in2 )

str1 = "Input 1 = {0}, {1}, {2}".format( n_in1[0], n_in1[1], n_in1[2] )
print( str1 )
str1 = "Input 2 = {0}, {1}, {2}".format( n_in2[0], n_in2[1], n_in2[2] )
print( str1 )

# Convert unit normal to rotation vector using OPS function ptNormalToRotVector
r1 = op.ptNormalToRotVector( n_in1 )
str1 = "Rotation Vector 1 = {0}, {1}, {2}".format( r1[0], r1[1], r1[2] )
print( str1 )
r2 = op.ptNormalToRotVector( n_in2 )
str1 = "Rotation Vector 2 = {0}, {1}, {2}".format( r2[0], r2[1], r2[2] )
print( str1 )

# Try to convert rotation vector back to unit normal
n_out1 = op.rotVectorToPtNormal( r1 )
str1 = "Output 1 = {0}, {1}, {2}".format( n_out1[0], n_out1[1], \
                                  n_out1[2] )
print( str1 )
n_out2 = op.rotVectorToPtNormal( r2 )
str1 = "Output 2 = {0}, {1}, {2}".format( n_out2[0], n_out2[1], \
                                  n_out2[2] )
print( str1 )
