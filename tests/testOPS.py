#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 20:46:42 2017
    Test lbfgsb
@author: amit
"""
import sys
sys.path.insert(0,'../')
import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
import vtk as v
import matplotlib.pyplot as plt
from src.lib.OPS import totEnergyJacobian,ptNormalToRotVector,\
    rotVectorToPtNormal

np.set_printoptions(threshold=1000)

# Set number of particles and generate initial degrees of freedom
Nx = 4
Ny = 4
Nz = 1
N = Nx*Ny*Nz

# Prepare a vtk points to store position and a double array to store normals
pts = v.vtkPoints()
pointNormals = v.vtkDoubleArray()
pointNormals.SetNumberOfComponents(3)
pointNormals.SetNumberOfTuples(N)

points = np.zeros((N,3))
normals = np.zeros((N,3))
index = 0
for i in range(Nx):
    xc = 0.0 + 1.5*i
    for j in range(Ny):
        yc = 0.0 + 1.5*j
        for k in range(Nz):
            zc = 0.0 + 1.5*k
            points[index] = np.array([xc,yc,zc])
            normals[index] = np.array([0.0,0.0,1.0])
            index += 1

# Assign values from x to pts and normals
for i in range(N):
    xi = points[i]
    ni = normals[i]
    pts.InsertNextPoint(xi)
    pointNormals.SetTuple(i, ni)

# Prepare writer and PolyData object
pd = v.vtkPolyData()
writer = v.vtkPolyDataWriter()

pd.SetPoints(pts)
pd.GetPointData().SetNormals(pointNormals)

writer.SetFileName("BeforePerturb.vtk")
writer.SetInputData(pd)
writer.Write()

#Now perturb
ptPerturb = 0.1*np.random.random((N,3))
normPerturb = np.random.random((N,3))

points = points + ptPerturb
normals = normals + normPerturb

for i in range(N):
    temp = normals[i]    
    normals[i] = temp/la.norm(temp)

# Convert point normals to rotation vectors
rotVec = np.zeros((N,3))
zaxis = np.array([0.0,0.0,1.0])
for i in range(N):
    rotVec[i] = ptNormalToRotVector(normals[i])

# Combine the normals and points
x0 = np.zeros((N*6))

for i in range(N):
    si = i*6
    x0[si:si+3] = rotVec[i]
    x0[si+3:si+6] = points[i]

pts.Initialize()
pointNormals.Reset()
pointNormals.SetNumberOfTuples(N)

# Assign values from x to pts and normals
for i in range(N):
    si = i*6
    ni = normals[i]
    xi = x0[si+3:si+6]
    pts.InsertNextPoint(xi)
    pointNormals.SetTuple(i, ni)

# Prepare writer and PolyData object
pd.SetPoints(pts)
pd.GetPointData().SetNormals(pointNormals)
writer.SetFileName("Points-0.vtk")
writer.SetInputData(pd)
writer.Write()

# Set Morse potential parameters
E = 1.0; l0 = 1.0; s = 6.0

# Set kernel parameters
K = 1.0; a = 3.0; b = 2.0

# Set energy coefficients
Am = 1.0; Ap = 1.0; An = 1.0; Ac = 1.0

# Assemble parameters in an array
params = np.array([Am, Ap, An, Ac, E, l0, s, K, a, b])

enTrend = []
jacTrend = []
def energyTrend(x):
    energy, jacobian = totEnergyJacobian(x, params)    
    jac_abs = np.absolute( jacobian )
    jac_max = np.max( jac_abs )
    enTrend.append(energy)
    jacTrend.append( jac_max )

# Minimize energy
res = minimize(totEnergyJacobian, x0, args=(params,), jac=True, \
               method='L-BFGS-B', callback=energyTrend)

# Prepare plot data
iterCount = len(enTrend)
iters = np.array( [ it for it in range(iterCount) ], dtype=np.int64 )
enTrend = np.array( enTrend )
jacTrend = np.array( jacTrend )

# Plot the energy and jacobian behavior
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.set_ylabel("Energy")
ax2.set_ylabel("Jac_Norm")
ax2.set_xlabel("Iterations")
ax1.semilogy(iters, enTrend)
ax2.semilogy(iters, jacTrend)

if res.success is True:
    print("Minimization successful!")
else:
    print("Minimization failed!")

# Convert rotation vectors to normals
for i in range(N):
    si = i*6
    vi = res.x[si:si+3]
    normals[i] = rotVectorToPtNormal(vi)

# Prepare a vtk points to store position and a double array to store normals
pts.Initialize()
pointNormals.Reset()
pointNormals.SetNumberOfTuples(N)

# Assign values from x to pts and normals
for i in range(N):
    si = i*6
    ni = normals[i]
    xi = res.x[si+3:si+6]
    pts.InsertNextPoint(xi)
    pointNormals.SetTuple(i, ni)
pd.SetPoints(pts)
pd.GetPointData().SetNormals(pointNormals)
# Write the final output
writer.SetFileName("Points-1.vtk")
writer.SetInputData(pd)
writer.Write()
