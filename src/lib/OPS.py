#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:18:21 2017
    Energy and Jacobian
@author: amit
"""
import vtk as v
import matplotlib.pyplot as plt

from numba import jit
from numpy import zeros, array, arcsin, sin, cos, pi, logspace, mean, \
    absolute, sqrt
from numpy.linalg import norm
from src.lib.Quaternion import conjugation

from src.lib.Potentials import morse, psi, phi_p, phi_n, phi_c
from src.lib.JacKernel import Dpsi_Dvi, Dpsi_Dxi, Dpsi_Dxj
from src.lib.JacPotV import Dphi_pDvi, Dphi_nDvi, Dphi_nDvj, Dphi_cDvi, Dphi_cDvj
from src.lib.JacPotX import Dmorse_Dxi, Dmorse_Dxj, Dphi_pDxi, Dphi_pDxj,\
    Dphi_cDxi, Dphi_cDxj

"""
Energy and Jacobian calculation for the full N^2 case
"""
@jit(nopython=True,cache=True)
def totEnergyJacobian(x_in, params, calcJac=True):
    N_tot = len(x_in) / 6
    N = int(N_tot)
    alphaM = params[0]
    alphaP = params[1]
    alphaN = params[2]
    alphaC = params[3]
    E = params[4]
    re = params[5]
    s = params[6]
    K = params[7]
    a = params[8]
    b = params[9]

    v = zeros((N,3))
    x = zeros((N,3))

    for i in range(N):
        si = i*6
        v[i] = x_in[si:si+3]
        x[i] = x_in[si+3:si+6]

    energy = 0.0
    J = zeros(N*6)
    if calcJac is False:
        for i in range(N):
            xi = x[i]
            vi = v[i]
            for j in range(N):
                if i != j :
                    xj = x[j]
                    vj = v[j]
                    # Calculate the energies
                    morseVal = morse( xi, xj, E, re, s)
                    KerXi = psi(vi,xi,xj,K,a,b)
                    Phi_pXi = phi_p(vi,xi,xj)
                    Phi_nXi = phi_n(vi,vj)
                    Phi_cXi = phi_c(vi,vj,xi,xj)
                    energy += alphaM*morseVal + (alphaP*Phi_pXi +\
                    alphaN*Phi_nXi + alphaC*Phi_cXi)*KerXi
    else:
        Jv = zeros((N,3))
        Jx = zeros((N,3))

        for i in range(N):
            xi = x[i]
            vi = v[i]
            for j in range(N):
                if i != j :
                    xj = x[j]
                    vj = v[j]
                    # Calculate the energies
                    morseVal = morse( xi, xj, E, re, s)

                    # Evaluate morse potential derivatives
                    dMorseXi = Dmorse_Dxi(xi,xj,E,re,s)
                    dMorseXj = Dmorse_Dxj(xj,xi,E,re,s)

                    # Evaluate kernel and its derivatives
                    KerXi = psi(vi,xi,xj,K,a,b)
                    KerXj = psi(vj,xj,xi,K,a,b)
                    dKerXi = Dpsi_Dxi(vi,xi,xj,K,a,b)
                    dKerXj = Dpsi_Dxj(vj,xj,xi,K,a,b)
                    dKerVi = Dpsi_Dvi(vi,xi,xj,K,a,b)

                    # Evaluate co-planarity potential and its derivatives
                    Phi_pXi = phi_p(vi,xi,xj)
                    Phi_pXj = phi_p(vj,xj,xi)
                    dPhi_pXi = Dphi_pDxi(vi,xi,xj)
                    dPhi_pXj = Dphi_pDxj(vj,xj,xi)
                    dPhi_pVi = Dphi_pDvi(vi,xi,xj)

                    # Evaluate co-normality potential and its derivatives
                    Phi_nXi = phi_n(vi,vj)
                    Phi_nXj = phi_n(vj,vi)
                    dPhi_nVi = Dphi_nDvi(vi,vj)
                    dPhi_nVj = Dphi_nDvj(vj,vi)

                    # Evaluate co-circularity potential and its derivatives
                    Phi_cXi = phi_c(vi,vj,xi,xj)
                    Phi_cXj = phi_c(vj,vi,xj,xi)
                    dPhi_cXi = Dphi_cDxi(vi,vj,xi,xj)
                    dPhi_cXj = Dphi_cDxj(vj,vi,xj,xi)
                    dPhi_cVi = Dphi_cDvi(vi,vj,xi,xj)
                    dPhi_cVj = Dphi_cDvj(vj,vi,xj,xi)

                    # Collect derivatives wrt xi and xj
                    Jx[i] += alphaM*( dMorseXi + dMorseXj )
                    Jx[i] += alphaP*( dPhi_pXi*KerXi + Phi_pXi*dKerXi +\
                          dPhi_pXj*KerXj + Phi_pXj*dKerXj )
                    Jx[i] += alphaN*( Phi_nXi*dKerXi + Phi_nXj*dKerXj )
                    Jx[i] += alphaC*( dPhi_cXi*KerXi + Phi_cXi*dKerXi +\
                          dPhi_cXj*KerXj + Phi_cXj*dKerXj )

                    # Collect derivatives wrt vi and vj
                    Jv[i] += alphaP*( dPhi_pVi*KerXi + Phi_pXi*dKerVi )
                    Jv[i] += alphaN*( dPhi_nVi*KerXi + Phi_nXi*dKerVi +\
                      dPhi_nVj*KerXj )
                    Jv[i] += alphaC*( dPhi_cVi*KerXi + Phi_cXi*dKerVi +\
                      dPhi_cVj*KerXj )

                    # Sum the energies
                    energy += alphaM*morseVal + (alphaP*Phi_pXi +\
                        alphaN*Phi_nXi + alphaC*Phi_cXi)*KerXi

        for i in range(N):
            si = i*6
            J[si:si+3] = Jv[i]
            J[si+3:si+6] = Jx[i]

    return energy, J

"""
Convert Point Normal to Rotation Vector
"""
@jit(nopython=True,cache=True)
def ptNormalToRotVector(ptNormal):
    # Assume z-axis of Global Coord Sys is the reference for ptNormal rotation
    # Cross-product of ptNormal with z-axis.
    cross_prod = array( [-ptNormal[1], ptNormal[0], 0.0] )
    cross_prod_norm = norm( cross_prod )
    # Check if ptNormal is parallel or anti-parallel to z-axis
    p3 = ptNormal[2]
    if cross_prod_norm < 1e-10:
        axis = array([ 1.0, 0.0, 0.0 ]) # Arbitrarily choose the x-axis
        angle = 0.0 if p3 > 0.0 else pi
    else:
        angle = arcsin( cross_prod_norm )
        angle = (pi - angle) if p3 < 0.0 else angle
        axis = cross_prod/cross_prod_norm
    rotVec = angle*axis
    return rotVec

"""
Convert Rotation Vector to Point Normal
"""
@jit(nopython=True,cache=True)
def rotVectorToPtNormal(rotVec):
    # Assume z-axis of Global Coord Sys is the reference for ptNormal rotation
    zaxis = array([0.0,0.0,1.0])
    vi = norm(rotVec) # Angle of rotation
    # Check 0-angle case
    if vi < 1e-10:
        ptNormal = array( [0.0, 0.0, 1.0] )
    elif abs(vi - pi) < 1e-10:
        ptNormal = array( [0.0, 0.0, -1.0] )
    else:
        # Make unit quaternion with angle vi and axis along rotVec
        w = sin( 0.5*vi )*(rotVec)/vi
        q = array( [ cos( 0.5*vi ), w[0], w[1], w[2] ])
        ptNormal = conjugation(q,zaxis)
    return ptNormal

"""
Calculate energy and Jacobian for a point and its neighbors. 'points' is a 1-d
array of rotation vector and position vector components. First 6 elements are
assumed to belong to the center node and rest belong to the neighbors. 'points'
has the form of consequtive rotation vector and position vectors like
[u0,v0,z0,x0,y0,z0,u1,v1,z1,x1,y1,z1....]. Output has first element as energy
remaining elements are Jacobian
"""
@jit(nopython=True,cache=True)
def pointEnergyJacobian(points, params, calcJac=True):
    N_tot = len(points) / 6
    N = int(N_tot)
    # Unpack params. Sequence is very important!!
    alphaM = params[0]
    alphaP = params[1]
    alphaN = params[2]
    alphaC = params[3]
    E = params[4]
    re = params[5]
    s = params[6]
    K = params[7]
    a = params[8]
    b = params[9]

    vi = points[0:3]
    xi = points[3:6]

    energy = 0.0
    Jx = zeros((N,3))
    Jv = zeros((N,3))

    EAndJ = zeros(N*6 + 1)

    if calcJac is False:
        for i in range(1,N):
            si = i*6
            vj = points[si:si+3]
            xj = points[si+3:si+6]

            morseEn = morse(xi,xj,E,re,s)
            KerXi = psi( vi, xi, xj, K, a, b )
            Phi_pXi = phi_p( vi, xi, xj )
            Phi_nXi = phi_n( vi, vj )
            Phi_cXi = phi_c( vi, vj, xi, xj )

            energy += alphaM*morseEn + ( alphaP*Phi_pXi + alphaN*Phi_nXi + \
                                    alphaC*Phi_cXi )*KerXi
    else:
        for i in range(1,N):
            si = i*6
            vj = points[si:si+3]
            xj = points[si+3:si+6]

            # Evaluate morse potential derivatives
            morseEn = morse(xi,xj,E,re,s)
            dMorseXi = Dmorse_Dxi( xi, xj, E, re, s )
            dMorseXj = Dmorse_Dxj( xi, xj, E, re, s)

            # Evaluate kernel and its derivatives
            KerXi = psi( vi, xi, xj, K, a, b )
            dKerXi = Dpsi_Dxi( vi, xi, xj, K, a, b )
            dKerXj = Dpsi_Dxj( vi, xi, xj, K, a, b )
            dKerVi = Dpsi_Dvi( vi, xi, xj, K, a, b )

            # Evaluate co-planarity potential and its derivatives
            Phi_pXi = phi_p( vi, xi, xj )
            dPhi_pXi = Dphi_pDxi( vi, xi, xj )
            dPhi_pXj = Dphi_pDxj( vi, xi, xj )
            dPhi_pVi = Dphi_pDvi( vi, xi, xj )

            # Evaluate co-normality potential and its derivatives
            Phi_nXi = phi_n( vi, vj )
            dPhi_nVi = Dphi_nDvi( vi, vj )
            dPhi_nVj = Dphi_nDvj( vi, vj )

            # Evaluate co-circularity potential and its derivatives
            Phi_cXi = phi_c( vi, vj, xi, xj )
            dPhi_cXi = Dphi_cDxi( vi, vj, xi, xj )
            dPhi_cXj = Dphi_cDxj( vi, vj, xi, xj )
            dPhi_cVi = Dphi_cDvi( vi, vj, xi, xj )
            dPhi_cVj = Dphi_cDvj( vi, vj, xi, xj )

            Jv[0] += alphaP*( dPhi_pVi*KerXi + Phi_pXi*dKerVi ) + \
                alphaN*( dPhi_nVi*KerXi + Phi_nXi*dKerVi ) + \
                alphaC*( dPhi_cVi*KerXi + Phi_cXi*dKerVi )
            Jx[0] += alphaM*dMorseXi + alphaP*( dPhi_pXi*KerXi + \
                  Phi_pXi*dKerXi ) + alphaN*( Phi_nXi*dKerXi ) \
                + alphaC*( dPhi_cXi*KerXi + Phi_cXi*dKerXi )

            Jv[i] += KerXi*( alphaN*dPhi_nVj + alphaC*dPhi_cVj )
            Jx[i] += alphaM*dMorseXj + alphaP*( dPhi_pXj*KerXi + \
                  Phi_pXi*dKerXj ) + alphaN*( Phi_nXi*dKerXj ) \
                + alphaC*( dPhi_cXj*KerXi + Phi_cXi*dKerXj )

            energy += alphaM*morseEn + ( alphaP*Phi_pXi + alphaN*Phi_nXi + \
                                    alphaC*Phi_cXi )*KerXi

        # Assemble Jacobian output
        for i in range(N):
            si = i*6 + 1
            EAndJ[si:si+3] = Jv[i]
            EAndJ[si+3:si+6] = Jx[i]

    EAndJ[0] = energy
    return EAndJ

"""
For a given x vector, identify points and for each point find neighbors and
calculate energy and Jacobian.
"""
def sysEnergyJacobian(x,params,calcJac=True):
    N_tot = len(x) / 6
    N = int(N_tot)
    # Create vtkPoints from x
    pts = v.vtkPoints()
    for i in range(N):
        si = i*6
        currPoint = x[si+3:si+6]
        pts.InsertNextPoint(currPoint)
    # Build k-d tree to search for neighbors
    kdt = v.vtkKdTree()
    kdt.BuildLocatorFromPoints( pts )
    # Get the search radius from params
    searchRad = params[10]
    params_internal = params[0:10]
    # Calculate energy and Jacobian
    energy = 0.0
    jacobian = zeros( N*6 )

    if calcJac is False:
        for i in range( N ):
            si = i*6
            centerRotVec = x[si:si+3]
            centerPoint = x[si+3:si+6]

            # Fetch neighbor ids
            currPointSet = v.vtkIdList()
            kdt.FindPointsWithinRadius( searchRad, centerPoint, currPointSet )
            # Separate center-point from neighbors list
            currNeighbors = []
            currNumPts = 1
            for j in range( currPointSet.GetNumberOfIds() ):
                currId = currPointSet.GetId( j )
                if( currId != i ):
                    currNeighbors.append( currId )
                    currNumPts += 1
            # Create ith point-set x-vector
            currX = zeros( currNumPts*6 )
            currX[0:3] = centerRotVec
            currX[3:6] = centerPoint
            sj = 6
            for j in currNeighbors:
                lsi = j*6
                currX[sj:sj+3] = x[lsi:lsi+3]
                currX[sj+3:sj+6] = x[lsi+3:lsi+6]
                sj += 6
            EAndJ =  pointEnergyJacobian(currX, params_internal, calcJac=False)
            energy += EAndJ[0]
    else:
        for i in range( N ):
            si = i*6
            centerRotVec = x[si:si+3]
            centerPoint = x[si+3:si+6]

            # Fetch neighbor ids
            currPointSet = v.vtkIdList()
            kdt.FindPointsWithinRadius( searchRad, centerPoint, currPointSet )
            # Separate center-point from neighbors list
            currNeighbors = []
            currNumPts = 1
            for j in range( currPointSet.GetNumberOfIds() ):
                currId = currPointSet.GetId( j )
                if( currId != i ):
                    currNeighbors.append( currId )
                    currNumPts += 1
            # Create ith point-set x-vector
            currX = zeros( currNumPts*6 )
            currX[0:3] = centerRotVec
            currX[3:6] = centerPoint
            sj = 6
            for j in currNeighbors:
                lsi = j*6
                currX[sj:sj+3] = x[lsi:lsi+3]
                currX[sj+3:sj+6] = x[lsi+3:lsi+6]
                sj += 6
            EAndJ =  pointEnergyJacobian( currX, params_internal)
            energy += EAndJ[0]
            currJacobian = EAndJ[1:]
            # Assemble into total jacobian
            jacobian[si:si+3] += currJacobian[0:3]
            jacobian[si+3:si+6] += currJacobian[3:6]
            sj = 6
            for j in currNeighbors:
                lsi = j*6
                jacobian[lsi:lsi+3] += currJacobian[sj:sj+3]
                jacobian[lsi+3:lsi+6] += currJacobian[sj+3:sj+6]
                sj += 6

    return energy, jacobian

"""
Define a consistency test function for sysEnergyJacobian
"""
def checkConsistency( x, params ):
    x0 = x.copy()
    _,J = sysEnergyJacobian(x0, params)
    h = logspace(-15,-1,20)
    #h = [1e-6]
    err = zeros(len(h))
    N = int( len(x0)/ 6.0 )
    for hi in range(len(h)):
        hc = h[hi]
        Jh = zeros(N*6)
        for i in range(N*6):
            x0[i] += hc
            Ep,_ = sysEnergyJacobian( x0, params, calcJac=False )
            x0[i] -= 2*hc
            Em,_ = sysEnergyJacobian( x0, params, calcJac=False )
            Jh[i] = (Ep - Em)/(2.0*hc)
            x0[i] += hc
        errJ = J - Jh
        err[hi] = mean(absolute(errJ))
    fig1, ax = plt.subplots()
    ax.loglog(h,err)
    ax.set_ylabel("Error")
    ax.set_xlabel("h")

"""
A function to calculate asphericity and average radius from solution array x
"""
@jit(nopython=True,cache=True)
def getRadialStats(x):
    N_tot = len(x)/6.0
    N = int(N_tot)
    # Calculate average radius
    ravg = 0.0
    rad = zeros(N)
    for i in range(N):
        si = i*6
        pt = x[si+3:si+6]
        rad[i] = sqrt( pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2] )
        ravg += rad[i]
    ravg /= N
    asph = 0.0
    for i in range(N):
        si = i*6
        pt = x[si+3:si+6]
        asph += (rad[i] - ravg)**2/N
    asph /= ravg**2
    return ravg, asph

"""
Convert x-array to a vtk file
"""
def writeToVTK(x,fileName):
    # Calculate the total number of points in the particle system.
    N = int( len(x)/6 )

    # Compute the point normals and create points and vertices for vtkPolyData
    finalPts = v.vtkPoints()
    unitSpherePts = v.vtkPoints()
    finalNormals = v.vtkDoubleArray()
    finalNormals.SetNumberOfComponents( 3 )
    finalNormals.SetNumberOfTuples( N )
    #verts = v.vtkCellArray()
    for i in range(N):
        si = i*6
        currRotVec = x[si:si+3]
        currNormal = rotVectorToPtNormal( currRotVec )
        currPoint = x[si+3:si+6]
        finalNormals.SetTuple( i, currNormal )
        finalPts.InsertNextPoint( currPoint )
        #verts.InsertNextCell(1)
        #verts.InsertCellPoint( i )
        normalizedPoint = currPoint / norm(currPoint)
        unitSpherePts.InsertNextPoint( normalizedPoint )

    unitSphere = v.vtkPolyData()
    ids = v.vtkIdFilter()
    d3D = v.vtkDelaunay3D()
    dssf = v.vtkDataSetSurfaceFilter()
    cellIds = v.vtkIdList()
    finalTris = v.vtkCellArray()

    """
    Project all the points from finalPts to a unit sphere.
    Generate a convex hull on the unit sphere. Map the topology
    of triangles from unitSphere to finalPts by making use of
    vtkIdFilter to track the original Ids of the points in finalPts
    """
    unitSphere.SetPoints( unitSpherePts )
    ids.SetIdsArrayName("origIds")
    ids.PointIdsOn()
    ids.SetInputData( unitSphere )
    d3D.SetInputConnection( ids.GetOutputPort() )
    dssf.SetInputConnection( d3D.GetOutputPort() )
    dssf.Update()
    triangulation = dssf.GetOutput()
    cells = triangulation.GetPolys()
    cells.InitTraversal()
    origIds = v.vtkIdTypeArray.SafeDownCast(
            triangulation.GetPointData().GetArray("origIds") )
    while cells.GetNextCell( cellIds ):
        if cellIds.GetNumberOfIds() == 3:
            finalTris.InsertNextCell( 3 )
            for i in range( 3 ):
                finalTris.InsertCellPoint(int(
                        origIds.GetTuple1(cellIds.GetId(i))))

    # Create new polydata
    pd = v.vtkPolyData()
    pd.SetPoints( finalPts )
    #pd.SetVerts( verts )
    pd.GetPointData().SetNormals( finalNormals )
    pd.SetPolys( finalTris )

    # Write the output file
    pdw = v.vtkPolyDataWriter()
    pdw.SetFileName(fileName)
    pdw.SetInputData(pd)
    pdw.Update()
    pdw.Write()
