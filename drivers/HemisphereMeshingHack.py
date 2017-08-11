import vtk
import numpy as np

reader = vtk.vtkPolyDataReader()
reader.SetFileName('T7.vtk')

ids = vtk.vtkIdFilter()
ids.PointIdsOn()
ids.SetIdsArrayName('OriginalIds')
ids.SetInputConnection( reader.GetOutputPort() )

plane = vtk.vtkPlane()
plane.SetOrigin(0.0,0.0,0.0)
plane.SetNormal(0.0,0.0,1.0)

clipper = vtk.vtkClipPolyData()
clipper.SetClipFunction( plane )
clipper.SetValue(0.0)
clipper.SetInputConnection( ids.GetOutputPort() )
clipper.Update()
clipped = clipper.GetOutput()

oldIds = clipped.GetCellData().GetScalars('OriginalIds')
projPts = vtk.vtkPoints()
for i in range( clipped.GetNumberOfPoints() ):
    currPoint = clipped.GetPoint( i )
    newPoint = np.zeros(3)
    plane.ProjectPoint( currPoint, newPoint )
    projPts.InsertNextPoint( newPoint )

projPd = vtk.vtkPolyData()
projPd.SetPoints( projPts )
projPd.GetPointData().AddArray(oldIds)

d2d = vtk.vtkDelaunay2D()
d2d.SetInputData( projPd )
d2d.Update()

mesh = d2d.GetOutput()
polys = vtk.vtkCellArray()
idList = vtk.vtkIdList()
oldIds = mesh.GetPointData().GetArray('OriginalIds')
while( mesh.GetPolys().GetNextCell( idList ) ):
    polys.InsertNextCell( 3 )
    for i in range( idList.GetNumberOfIds() ):
        oldId = oldIds.GetValue( idList.GetId( i ) )
        polys.InsertCellPoint( oldId )

pdo = vtk.vtkPolyData()
pdo.SetPoints( reader.GetOutput().GetPoints() )
pdo.SetPolys( polys )

pdw = vtk.vtkPolyDataWriter()
pdw.SetInputData( pdo )
pdw.SetFileName('Mesh.vtk')
pdw.Update()
pdw.Write()
