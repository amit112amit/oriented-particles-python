#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 19:00:24 2017
    Convert NumPy array output to VTK files
@author: amit
"""
import vtk as v
import numpy as np
import OPS as ops
import sys
from os.path import splitext

# Convert .txt files to vtk
fileName, file_ext = splitext(sys.argv[1])
sol = np.loadtxt(sys.argv[1])  
N = int( len(sol)/6 )
finalPts = v.vtkPoints()
finalNormals = v.vtkDoubleArray()
finalNormals.SetNumberOfComponents( 3 )
finalNormals.SetNumberOfTuples( N )
verts = v.vtkCellArray()
for i in range(N):
    si = i*6
    currRotVec = sol[si:si+3]
    currNormal = ops.rotVectorToPtNormal( currRotVec )
    currPoint = sol[si+3:si+6]
    finalNormals.SetTuple( i, currNormal )
    finalPts.InsertNextPoint( currPoint )
    verts.InsertNextCell(1)
    verts.InsertCellPoint( i )

# Create new polydata
pd = v.vtkPolyData()
pd.SetPoints( finalPts )
pd.SetVerts( verts )
pd.GetPointData().SetNormals( finalNormals )

# Write the output file
pdw = v.vtkPolyDataWriter()
str1 = "{0}.vtk".format(fileName)
pdw.SetFileName(str1)
pdw.SetInputData(pd)
pdw.Update()
pdw.Write()