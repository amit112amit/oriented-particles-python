#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derivative of Volume of a Closed shell wrt position of the points
"""

import vtk as v
import numpy as np
import matplotlib.pyplot as plt

"""
The input of this class should be a polydata type with consistently labeled
triangles. It returns volume of the closed mesh formed by the triangles.
"""
def Vol(poly):    
    cells = poly.GetPolys()
    cells.InitTraversal()
    verts = v.vtkIdList()
    V = 0.0
    while( cells.GetNextCell( verts ) ):
        coords = []
        assert( verts.GetNumberOfIds() == 3 )
        for i in range( 3 ):
            coords.append( poly.GetPoint( verts.GetId(i) ) )
        a,b,c = coords
        V += (1/6)*np.dot( a, np.cross(b,c) )
    return V

"""
The input to this class is a polydata object with triangles ordered 
consistently. It returns a Matrix with N rows and 3 columns where N is the 
number of points in the polydata
"""
def VolDiff(poly):
    N = poly.GetNumberOfPoints()    
    poly.BuildLinks()
    Out = np.zeros( (N,3) )
    cells = poly.GetPolys()
    cells.InitTraversal()
    cellPts = v.vtkIdList()
    while( cells.GetNextCell(cellPts) ):
        ida = cellPts.GetId(0)
        idb = cellPts.GetId(1)
        idc = cellPts.GetId(2)        
        a = poly.GetPoint( ida )
        b = poly.GetPoint( idb )
        c = poly.GetPoint( idc )
        dVa = np.array([b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]])
        dVb = np.array([a[2]*c[1]-a[1]*c[2],a[0]*c[2]-a[2]*c[0],a[1]*c[0]-a[0]*c[1]])
        dVc = np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
        Out[ida,:] += (dVa*(1/6))
        Out[idb,:] += (dVb*(1/6))
        Out[idc,:] += (dVc*(1/6))
    return Out

"""
Consistency test
"""
sp = v.vtkSphereSource()
sp.SetCenter(0.0,0.0,0.0)
sp.SetRadius(1.0)
sp.SetThetaResolution(5)
sp.SetPhiResolution(5)
sp.Update()
pd = sp.GetOutput()
writer = v.vtkPolyDataWriter()
writer.SetFileName('Sphere.vtk')
writer.SetInputData(pd)
writer.Write()

N = pd.GetNumberOfPoints()
hvec = np.logspace(-7,-2)
err = np.zeros(50)
dV = VolDiff(pd)
points = pd.GetPoints()
for z in range(50):
    h = hvec[z]
    dVh = np.zeros((N,3))
    errh = np.zeros(N)
    for i in range(N):        
        pt = np.array(points.GetPoint(i))
        for j in range(3):
            pt[j] += h
            points.SetPoint(i, pt)            
            Vp = Vol(pd)
            pt[j] += -2*h
            points.SetPoint(i, pt)                      
            Vm = Vol(pd)
            dVh[i,j] = (Vp-Vm)/(2*h)            
            pt[j] += -2*h
            points.SetPoint(i, pt)            
        errh[i] = np.linalg.norm(dV[i,:] - dVh[i,:])
    err[z] = np.max( errh )

fig, ax = plt.subplots()
ax.loglog(hvec,err)
        