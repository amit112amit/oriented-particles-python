#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:32:55 2017
   Mesh Point Cloud
@author: amit
"""
import numpy as np
import vtk as v

"""
Read a polydata vtk file and insert triangles in it
"""
def meshPointCloud( inputVTKfile, searchRad ):
    reader = v.vtkPolyDataReader()
    reader.SetFileName( inputVTKfile )
    reader.Update()
    pd = reader.GetOutput()
    N = pd.GetNumberOfPoints()
    kdt = v.vtkKdTree()
    kdt.BuildLocatorFromPoints( pd.GetPoints() )
    
    polys = v.vtkCellArray()
    for i in range( N ):
        currPoint = pd.GetPoint(i)
        idList = v.vtkIdList()        
        kdt.FindPointsWithinRadius( searchRad, currPoint, idList )
        
        neighbors = []
        for j in range( idList.GetNumberOfIds() ):
            currId = idList.GetId(j)
            if j != i:
                neighbors.append(currId)
        
        