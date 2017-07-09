#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:02:32 2017
    Driver file for asphericity calculations
@author: amit
"""
import sys
sys.path.insert(0,'../../')

import vtk as v
import numpy as np
import numpy.linalg as la
import src.lib.OPS as ops
#import matplotlib.pyplot as plt

from scipy.optimize import minimize

############################### DRIVER BEGINS ################################

# Set simulation parameters
alphaM = np.logspace(0,5,100)
alphaP = 1.0
alphaN = 1.0
alphaC = 1.0
E = 1.0
re = 1.0
s = np.log(2)/0.1
K = 1.0
a = 1.0
b = 1.0
searchRad = 1.5

params = np.array([ alphaM[0], alphaP, alphaN, alphaC, E, re, s, K, a, b, \
                   searchRad])

# Read a input file
reader = v.vtkPolyDataReader()
reader.SetFileName('T7.vtk')
reader.Update()

# Fetch the polydata
pd = reader.GetOutput()
numPoints = pd.GetNumberOfPoints()

# Compute the average distance between neighboring particles for normalization.
kdt = v.vtkKdTree()
kdt.BuildLocatorFromPoints( pd.GetPoints() )
avgLen = 0.0
numberOfEdges = 0
for i in range( numPoints ):
    currPoint = np.array( pd.GetPoint( i ) )
    neighbors = v.vtkIdList()
    kdt.FindPointsWithinRadius( 1.1, currPoint, neighbors )
    # One of the points is the input itself. So we need to skip it
    numNeighbors = neighbors.GetNumberOfIds() - 1
    numberOfEdges += numNeighbors
    for j in range( numNeighbors + 1 ):
        neighborId = neighbors.GetId( j )
        if neighborId != i:
            neighbor = np.array( pd.GetPoint( neighborId ) )
            avgLen += la.norm( neighbor - currPoint )
avgLen /= numberOfEdges
print( "Avg Length = {0}".format( avgLen ) )

# Compute point normals and create normalized vertices
verts = v.vtkCellArray()
newPoints = v.vtkPoints()
normals = v.vtkDoubleArray()
normals.SetNumberOfComponents( 3 )
normals.SetNumberOfTuples( numPoints )
for i in range(numPoints):
    currPoint = np.array( pd.GetPoint(i) )
    newCurrPoint = currPoint / avgLen
    currNormal = currPoint/la.norm( currPoint )
    normals.SetTuple( i, currNormal )
    newPoints.InsertNextPoint( newCurrPoint )
    verts.InsertNextCell(1)
    verts.InsertCellPoint( i )

# Assign points, normals and vertices
pd.SetPoints( newPoints )
pd.SetVerts( verts )
pd.GetPointData().SetNormals( normals )

# Keep a writer ready
pdw = v.vtkPolyDataWriter()

# Print the initial configurations
pdw.SetFileName('Initial.vtk')
pdw.SetInputData(pd)
pdw.Update()
pdw.Write()

# Convert points and normals to a x-vector
x0 = np.zeros(numPoints*6)
for i in range( numPoints ):
    currNormal = np.array( normals.GetTuple( i ) )
    currRotVec = ops.ptNormalToRotVector( currNormal )
    currPoint = np.array( pd.GetPoint(i) )
    si = i*6
    x0[si:si+3] = currRotVec
    x0[si+3:si+6] = currPoint

#solFile = open('Solution.txt','a')

# Main solution loop: Change parameter, solve, print to file
for i in range( len(alphaM) ):
    # Change parameter alphaM
    params[0] = alphaM[i]
    
    str1 = "i = {0} alphaM = {1}".format(i,alphaM[i])
    print(str1)
    
    # Solve
    res = minimize(ops.sysEnergyJacobian, x0, args=(params,), jac=True, \
               method='L-BFGS-B')    
    if res.success == True:
        print("\tMinimization successful!")
        #Update the guess for next iteration to current solution
        x0 = res.x
    else:
        print("\tMinimization failed")
    
    # Print solution vtk file
    sol = res.x
    str1 = "T7-step-{0}.txt".format(i)
    np.savetxt(str1, sol)