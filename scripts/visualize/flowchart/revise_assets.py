"""Source-backed assets for the compact editable workflow revision."""
from pathlib import Path
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.ndimage import map_coordinates, distance_transform_edt, binary_fill_holes
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_surf
from brainspace.plotting.base import Plotter
import vtk

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'compact'; OUT.mkdir(exist_ok=True)
ASSETS=OUT/'assets'; ASSETS.mkdir(exist_ok=True)
def read(p): return json.loads(Path(p).read_text())
def load(p): return np.load(p,allow_pickle=False)
base=ROOT/'outputs_hcp/inputs/hippunfold/dense_corobl_batch/sub-100610/hippunfold/sub-100610'
native=read_surface(str(base/'surf/sub-100610_hemi-L_space-T1w_den-512_label-hipp_midthickness.surf.gii'))
hip=read_surface(str(ROOT/'external/hippomaps/hippomaps/resources/canonical_surfs/tpl-avg_space-canonical_den-512_label-hipp_midthickness.surf.gii'))
hip.Points[:,0]*=-1
# Mirrored canonical and native meshes have opposite winding but identical IDs.
np.testing.assert_array_equal(np.sort(hip.GetCells2D(),axis=1),np.sort(native.GetCells2D(),axis=1))
fc=load(ROOT/'outputs_hcp/inputs/kong2022/shared_inputs/pfm/sub-100610/fc/kong2022/2mm/hemi_L/hipp_vertex_to_network_fc.npy')
ap=native.Points[:,1]

def render(name,surface,values,cmap,limits,size,zoom,view=None):
    surface.append_array(np.asarray(values,dtype=np.float32),name='v',at='point',overwrite=True)
    p=plot_surf({'s':surface},[['s']],array_name='v',view=view,cmap=cmap,color_range=limits,
        color_bar=None,nan_color=(.86,.87,.88,1),background=(1,1,1),size=size,zoom=zoom,
        screenshot=True,filename=str(ASSETS/name),transparent_bg=False,interactive=False,
        suppress_warnings=True,return_plotter=True,
        mapper__interpolateScalarsBeforeMapping=not isinstance(cmap,ListedColormap))
    ren=list(p.renderers.values())[0][-1].VTKObject
    actors=ren.GetActors(); actors.InitTraversal(); actor=actors.GetNextActor(); matrix=actor.GetMatrix()
    p.screenshot(str(ASSETS/name),transparent_bg=False)
    xy=[]
    for point in surface.Points:
        world=matrix.MultiplyPoint([*point,1]); ren.SetWorldPoint(*world); ren.WorldToDisplay(); d=ren.GetDisplayPoint()
        xy.append([d[0],size[1]-d[1]])
    im=Image.open(ASSETS/name).convert('RGB'); yy,xx=np.where(np.any(np.asarray(im)<245,axis=2))
    crop=(int(xx.min()-12),int(yy.min()-12),int(xx.max()+13),int(yy.max()+13))
    assert crop[0]>=0 and crop[1]>=0 and crop[2]<=size[0] and crop[3]<=size[1]
    im.crop(crop).save(ASSETS/name)
    xy=(np.asarray(xy)-np.array(crop[:2]))/np.array([crop[2]-crop[0],crop[3]-crop[1]])
    # True visible surface points for selecting a callout in the requested blue patch.
    transform=vtk.vtkTransform(); transform.SetMatrix(matrix)
    tf=vtk.vtkTransformPolyDataFilter(); tf.SetInputData(surface.VTKObject); tf.SetTransform(transform); tf.Update()
    ids=vtk.vtkIdFilter() if hasattr(vtk,'vtkIdFilter') else vtk.vtkGenerateIds()
    ids.SetInputData(tf.GetOutput()); ids.SetPointIdsArrayName('source_id'); ids.PointIdsOn(); ids.CellIdsOff(); ids.Update()
    visible=vtk.vtkSelectVisiblePoints(); visible.SetInputConnection(ids.GetOutputPort()); visible.SetRenderer(ren); visible.SetTolerance(.002); visible.Update()
    a=visible.GetOutput().GetPointData().GetArray('source_id')
    visible_ids=np.array([int(a.GetTuple1(i)) for i in range(a.GetNumberOfTuples())],dtype=int)
    Plotter.close_all(); return xy,visible_ids

# Pick a visible real vertex in the body, away from the rendered silhouette.
hipxy,hip_visible=render('hipp_neutral.png',hip,np.zeros(512),ListedColormap(['#D5DADE']),(-.5,.5),(600,800),1.45)
neutral=np.asarray(Image.open(ASSETS/'hipp_neutral.png').convert('RGB'))
interior=distance_transform_edt(binary_fill_holes(np.any(neutral<250,axis=2)))
pixels=np.rint(hipxy*np.array([neutral.shape[1],neutral.shape[0]])).astype(int)
pixels[:,0]=np.clip(pixels[:,0],0,neutral.shape[1]-1)
pixels[:,1]=np.clip(pixels[:,1],0,neutral.shape[0]-1)
depth=interior[pixels[:,1],pixels[:,0]]
ap_fraction=(ap-ap.min())/np.ptp(ap)
candidates=hip_visible[(ap_fraction[hip_visible]>.35)&(ap_fraction[hip_visible]<.65)
    &(hipxy[hip_visible,1]>.4)&(hipxy[hip_visible,1]<.62)
    &np.isfinite(fc[hip_visible]).all(axis=1)&(depth[hip_visible]>10)]
assert candidates.size,'No visible interior hippocampal body vertex available'
vertex=int(candidates[np.argmax(depth[candidates])])
highlight=np.zeros(512); highlight[vertex]=1
assert np.count_nonzero(highlight)==1
hipxy,_=render('hipp_neutral.png',hip,highlight,ListedColormap(['#D5DADE','#BB861C']),(0,1),(600,800),1.45)
ctx=read_surface(str(ROOT/'data/hippunfold_input/sub-100610/anat/sub-100610_hemi-L_space-fsLR_den-32k_desc-MSMAll_inflated.surf.gii'))
labels=np.loadtxt(ROOT/'outputs_hcp/inputs/kong2022/cortex_references/routea_areal_mshbm/sub-100610/lh_labels.txt',dtype=int)
lut=nib.load(ROOT/'outputs_hcp/inputs/kong2022/cortex_references/routea_areal_mshbm/sub-100610/assets/sub-100610_atlas-kong2022_areal_mshbm.L.label.gii').labeltable.labels
network={}
for l in lut:
    if l.key: network[l.label.split('_')[2]]='#'+''.join(f'{round(c*255):02X}' for c in l.rgba[:3])
profile=np.arctanh(np.clip(fc[vertex],-.999999,.999999)); profile=(profile-profile.mean())/profile.std()
values=np.full(labels.shape,np.nan); mask=labels>0; values[mask]=profile[labels[mask]-1]
ctxxy,visible_ids=render('cortex_seed_fc.png',ctx,values,'RdBu_r',(-2.5,2.5),(1000,700),1.05,'lateral')
# User-selected visual target: slightly right of the former callout, inside blue.
eligible=visible_ids[(labels[visible_ids]>0)&(values[visible_ids]<-.35)]
# Check the displayed pixel too: interpolated/illuminated parcel edges can look
# white despite a negative underlying ROI value. Keep the circle visibly blue.
rendered=np.asarray(Image.open(ASSETS/'cortex_seed_fc.png').convert('RGB'),dtype=float)
pixels=np.rint(ctxxy[eligible]*np.array([rendered.shape[1],rendered.shape[0]])).astype(int)
pixels[:,0]=np.clip(pixels[:,0],0,rendered.shape[1]-1); pixels[:,1]=np.clip(pixels[:,1],0,rendered.shape[0]-1)
rgb=rendered[pixels[:,1],pixels[:,0]]
eligible=eligible[(rgb[:,2]-rgb[:,0]>25)&(rgb[:,2]-rgb[:,1]>5)]
assert eligible.size,'No visible blue ROI near requested cortical callout'
target_xy=np.array([.64,.36])
roi_vertex=int(eligible[np.argmin(np.linalg.norm(ctxxy[eligible]-target_xy,axis=1))])
assert np.linalg.norm(ctxxy[roi_vertex]-target_xy)<.1,'Requested blue patch unavailable'
roi=int(labels[roi_vertex]); roi_name=[l.label for l in lut if l.key==roi]
roi_xy=ctxxy[labels==roi]

# A true sagittal plane; maximize AP extent of the actual surface intersection.
planes=[]
for x in np.arange(-31.,-19.9,.5):
    plane=vtk.vtkPlane(); plane.SetOrigin(float(x),0,0); plane.SetNormal(1,0,0)
    cut=vtk.vtkCutter(); cut.SetInputData(native.VTKObject); cut.SetCutFunction(plane); cut.Update(); mesh=cut.GetOutput()
    points=np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
    if points.size: planes.append((float(np.ptp(points[:,1])),float(x)))
_,slice_x=max(planes)
plane=vtk.vtkPlane(); plane.SetOrigin(slice_x,0,0); plane.SetNormal(1,0,0)
cut=vtk.vtkCutter(); cut.SetInputData(native.VTKObject); cut.SetCutFunction(plane); cut.Update(); mesh=cut.GetOutput()
mri=nib.load(ROOT/'data/hippunfold_input/sub-100610/anat/sub-100610_T1w.nii.gz')
y=np.linspace(3,-49,650); z=np.linspace(-34,16,625); yy,zz=np.meshgrid(y,z)
world=np.stack([np.full_like(yy,slice_x),yy,zz,np.ones_like(yy)],axis=0).reshape(4,-1)
vox=(np.linalg.inv(mri.affine)@world)[:3]
img=map_coordinates(np.asarray(mri.dataobj),vox,order=1).reshape(yy.shape)
fig=plt.figure(figsize=(5.2,5)); ax=fig.add_axes([0,0,1,1])
ax.imshow(img,extent=(3,-49,-34,16),origin='lower',cmap='gray',vmin=0,vmax=np.percentile(img,99.3))
for i in range(mesh.GetNumberOfCells()):
    cell=mesh.GetCell(i); pts=np.array([mesh.GetPoint(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())])
    ax.plot(pts[:,1],pts[:,2],color='#F0B44B',lw=1.5)
ax.set_xlim(3,-49); ax.set_ylim(-34,16); ax.axis('off'); fig.savefig(ASSETS/'sagittal_AP.png',dpi=200); plt.close(fig)

tm=read(ROOT/'outputs_hcp/templates/hcp151_routea_full_k3_fisherz/template_manifest.json')
rows=[r for r in tm['combinations'] if r['hemi']=='L' and r['smoothing']=='2mm']; assert len(rows)==1
t=rows[0]; gm=read(t['source_files']['gradient_manifest']['path'])
g=load(gm['outputs']['hippomaps_gradients_full']); group=load(t['outputs']['labels']['path']); support=load(t['outputs']['support']['path']).astype(bool)
fig=plt.figure(figsize=(6,4.5)); ax=fig.add_subplot(projection='3d')
for i,c in enumerate(['#DC5CFF','#6396C2','#D85C6B'],1):
    m=support&(group==i); ax.scatter(*g[m,:3].T,s=25,color=c,linewidths=0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(22,-55); ax.set_proj_type('ortho'); ax.grid(False); ax.set_box_aspect((1,1,.8),zoom=1.05)
for a in [ax.xaxis,ax.yaxis,ax.zaxis]: a.pane.fill=False; a.pane.set_edgecolor((1,1,1,0)); a.line.set_color('#BBC0C6')
fig.subplots_adjust(.02,.07,.9,.96); fig.canvas.draw()
axis_points={}
for name,a in zip(['Gradient 1','Gradient 2','Gradient 3'],[ax.xaxis,ax.yaxis,ax.zaxis]):
    screen=a.line.get_transform().transform(a.line.get_xydata())
    axis_points[name]=(screen/np.array(fig.bbox.size)).tolist()
assert abs(axis_points['Gradient 3'][0][0]-axis_points['Gradient 3'][1][0])<1e-8
fig.savefig(ASSETS/'gradient_space.png',dpi=200); plt.close(fig)
# Remove internal plotting margins, preserving the data and orthographic axes.
im=Image.open(ASSETS/'gradient_space.png').convert('RGB'); width,height=im.size
yy,xx=np.where(np.any(np.asarray(im)<245,axis=2))
left,top=max(0,int(xx.min())-18),max(0,int(yy.min())-18)
right,bottom=min(width,int(xx.max())+19),min(height,int(yy.max())+19)
im.crop((left,top,right,bottom)).save(ASSETS/'gradient_space.png')
axis_points={name:[[(u*width-left)/(right-left),1-((1-v)*height-top)/(bottom-top)]
    for u,v in ends] for name,ends in axis_points.items()}

ind=read(ROOT/'outputs_stanford/work/hcp151_fixed_template_matching/subjects/sub-1016/matching/2mm/hemi-L/combo_manifest.json')
ifc=load(ind['outputs']['fisherz_fc']['path']); proto=load(t['outputs']['prototypes']['path'])
scores=load(ind['outputs']['scores']['path']); ilabel=load(ind['outputs']['labels']['path'])
assert ind['template_prototypes']['path']==t['outputs']['prototypes']['path']
np.testing.assert_allclose(scores,np.array([[np.corrcoef(a,b)[0,1] for b in proto] for a in ifc]),atol=2e-6,rtol=0)
np.testing.assert_array_equal(ilabel,scores.argmax(1)+1)
for name,data in [('individual_fc.png',ifc),('templates.png',proto)]:
    fig=plt.figure(figsize=(4,2 if name=='individual_fc.png' else .7)); ax=fig.add_axes([0,0,1,1])
    ax.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-.2,vmax=.2,interpolation='nearest'); ax.axis('off')
    fig.savefig(ASSETS/name,dpi=180); plt.close(fig)
meta=dict(vertex=vertex,vertex_ap_mm=float(ap[vertex]),ap_fraction=float((ap[vertex]-ap.min())/np.ptp(ap)),
    hip_xy=hipxy[vertex].tolist(),roi=roi,roi_name=roi_name[0],roi_xy=ctxxy[roi_vertex].tolist(),
    roi_vertex=roi_vertex,roi_display_value=float(values[roi_vertex]),
    roi_bounds=[roi_xy.min(0).tolist(),roi_xy.max(0).tolist()],network=network,sagittal_x_mm=slice_x,
    gradient_axes=axis_points)
(OUT/'asset_details.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
