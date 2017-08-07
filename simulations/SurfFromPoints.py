#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 13:58:08 2017
    Test creating a surface from a point cloud
@author: amit
"""
import vtk as v

# Read input file
rd = v.vtkPolyDataReader()
rd.SetFileName('T7.vtk')
