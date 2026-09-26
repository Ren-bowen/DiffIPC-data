from pathlib import Path
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

base = Path(__file__).resolve().parent
source = base.parent / 'result-target'
frames = base / 'frames'
frames.mkdir(exist_ok=True)
meshes=[]
lo=np.full(3,np.inf); hi=-lo
for i in range(41):
    reader=vtk.vtkXMLUnstructuredGridReader(); reader.SetFileName(str(source/f'step_{i}_surf.vtu')); reader.Update()
    grid=reader.GetOutput()
    grid.GetPointData().SetActiveVectors('displacement')
    warp=vtk.vtkWarpVector(); warp.SetInputData(grid); warp.SetScaleFactor(1); warp.Update()
    mesh=vtk.vtkUnstructuredGrid(); mesh.DeepCopy(warp.GetOutput()); meshes.append(mesh)
    pts=vtk_to_numpy(mesh.GetPoints().GetData()); ids=vtk_to_numpy(mesh.GetPointData().GetArray('body_ids')).ravel()
    moving=pts[ids>1]; lo=np.minimum(lo,moving.min(axis=0)); hi=np.maximum(hi,moving.max(axis=0))
print('bunny bounds',lo,hi,flush=True)
ren=vtk.vtkRenderer(); ren.SetBackground(0.94,0.96,0.98)
win=vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(1280,720); win.SetMultiSamples(8)
lut=vtk.vtkLookupTable(); lut.SetNumberOfTableValues(3); lut.SetTableRange(1,3); lut.Build()
for i,c in enumerate([(0.72,0.77,0.81,1),(0.18,0.52,0.86,1),(1.0,0.68,0.15,1)]): lut.SetTableValue(i,*c)
mapper=vtk.vtkDataSetMapper(); mapper.SetInputData(meshes[0]); mapper.SetScalarModeToUsePointFieldData(); mapper.SelectColorArray('body_ids'); mapper.SetLookupTable(lut); mapper.SetScalarRange(1,3)
actor=vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetAmbient(0.3); actor.GetProperty().SetDiffuse(0.7); ren.AddActor(actor)
center=(lo+hi)/2
cam=ren.GetActiveCamera(); cam.SetFocalPoint(*center); cam.SetPosition(*(center+np.array([0.55,0.8,1.7])*8)); cam.SetViewUp(0,1,0); cam.ParallelProjectionOn(); cam.SetParallelScale(max((hi[0]-lo[0])*0.36,(hi[1]-lo[1])*0.8,1.35))
label=vtk.vtkTextActor(); label.SetPosition(35,665); label.GetTextProperty().SetFontSize(25); label.GetTextProperty().SetColor(1,1,1); ren.AddActor2D(label)
sub=vtk.vtkTextActor(); sub.SetInput('Saved target simulation  |  0.25x speed'); sub.SetPosition(35,30); sub.GetTextProperty().SetFontSize(18); sub.GetTextProperty().SetColor(0.92,0.94,0.96); ren.AddActor2D(sub)
for i,mesh in enumerate(meshes):
    mapper.SetInputData(mesh); label.SetInput(f'Fig. 1  Bunnies     t = {i*0.05:.2f} s'); ren.ResetCameraClippingRange(); win.Render()
    capture=vtk.vtkWindowToImageFilter(); capture.SetInput(win); capture.ReadFrontBufferOff(); capture.Update()
    writer=vtk.vtkPNGWriter(); writer.SetFileName(str(frames/f'frame_{i:03d}.png')); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
    print('frame',i,flush=True)
