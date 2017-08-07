#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:02:32 2017
    Driver file for asphericity calculations
@author: amit
"""
import sys
sys.path.insert(0,\
            '/home/amit/GoogleDrive/Research/Code/oriented-particles-python')
import os
import vtk as v
import numpy as np
import numpy.linalg as la
import src.lib.OPS as ops
import matplotlib.pyplot as plt

from scipy.optimize import minimize

############################### DRIVER BEGINS ################################

#fileNameExt = sys.argv[1]
#initialRad = float(sys.argv[2])
#makePlots = True
fileNameExt = 'T7.vtk'
initialRad = 1.1
makePlots = True

# Separate file name from extension
fileName, fileExt = os.path.splitext(fileNameExt)

# Set simulation parameters
alphaM = np.logspace(0,2,100)
#alphaM = np.array([1.4174741629268053])
alphaP = 18.0
alphaN = 1.0
alphaC = 1.0
E = 1.0
re = 1.0
s = np.log(2)/0.1
K = 1.0
a = 1.0
b = 1.0
searchRad = 1.5
numAttempts = 3

params = np.array([ alphaM[0], alphaP, alphaN, alphaC, E, re, s, K, a, b, \
                   searchRad])

# Read a input file
reader = v.vtkPolyDataReader()
reader.SetFileName(fileNameExt)
reader.Update()

# Fetch the polydata
pd = reader.GetOutput()
numPoints = pd.GetNumberOfPoints()

# Compute the average distance between neighboring particles for normalization.
kdt = v.vtkKdTree()
kdt.BuildLocatorFromPoints( pd )
avgLen = 0.0
numberOfEdges = 0
for i in range( numPoints ):
    currPoint = np.array( pd.GetPoint( i ) )
    neighbors = v.vtkIdList()
    kdt.FindPointsWithinRadius( initialRad, currPoint, neighbors )
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
pdw.SetFileName('Before-step.vtk')
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

# Main solution loop: Change parameter, solve, print to file
asphericity = np.zeros( len(alphaM) )
radius = np.zeros( len(alphaM) )
for i in range( len(alphaM) ):
    # Change parameter alphaM
    params[0] = alphaM[i]
    
    str1 = "i = {0} alphaM = {1}".format(i,alphaM[i])
    print(str1)
    # Make 2 attempts to minimize
    attempt = 0
    while attempt < numAttempts:
        # Solve
        res = minimize(ops.sysEnergyJacobian, x0, args=(params,), jac=True, \
                   method='L-BFGS-B')    
        if res.success == False and attempt < numAttempts:
            print("\tMinimization failed. Trying again.")
            attempt += 1
        elif res.success == False and attempt == numAttempts:
            print("\t\tFailed again at i = {0}. Check output.".format(i))
            sol = res.x
            str1 = "{0}-step-failed-{1}.vtk".format(fileName,i)
            ops.writeToVTK(sol,str1)
            attempt += 1
        else:
            print("\tMinimization successful!")
            # Print solution vtk file
            sol = res.x
            radius[i], asphericity[i] = ops.getRadialStats(sol)
            str1 = "{0}-step-{1}.vtk".format(fileName,i)
            ops.writeToVTK(sol,str1)
            attempt = numAttempts
        #Update the guess for next iteration to current solution
        x0 = res.x        
# Plot Asphericity
if makePlots is True:
    plt.rc('text', usetex=True)
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.semilogx(alphaM, asphericity)
    ax1.set_ylabel('Asphericity')
    ax2.semilogx(alphaM, radius)
    ax2.set_ylabel('Radius')
    ax2.set_xlabel(r'$\alpha_M$')
    figFile = "{0}_Asph.pdf".format(fileName)
    plt.savefig(figFile)