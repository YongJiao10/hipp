"""Compact source-backed layout, shared by PNG/PDF and native editable PPTX."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Ellipse
from matplotlib.font_manager import FontProperties
from PIL import Image

HERE=Path(__file__).resolve().parent
OUT=HERE/'compact'
meta=json.loads((OUT/'asset_details.json').read_text())
OLD='../final/assets/'
NEW='assets/'
PANELS=[]
def panel(title,x,y,w,h):
    p=dict(title=title,x=x,y=y,w=w,h=h,elements=[]); PANELS.append(p); return p
def text(p,s,x,y,w,size=14,h=.3,color='#263440',bold=False,align='center'):
    p['elements'].append(dict(type='text',text=s,x=x,y=y,w=w,h=h,size=size,color=color,bold=bold,align=align))
def image(p,path,x,y,w,h):
    iw,ih=Image.open(OUT/path).size; k=min(w/iw,h/ih); dw,dh=iw*k,ih*k
    box=dict(type='image',path=path,x=x+(w-dw)/2,y=y+(h-dh)/2,w=dw,h=dh)
    p['elements'].append(box); return box
def shape(p,t,x,y,w,h,color='#89949E',width=1.3,fill=None):
    p['elements'].append(dict(type=t,x=x,y=y,w=w,h=h,color=color,width=width,fill=fill))
def arrow(p,x,y,x2,y2,color='#89949E',width=1.3):shape(p,'arrow',x,y,x2-x,y2-y,color,width)
def point(box,xy):return box['x']+xy[0]*box['w'],box['y']+xy[1]*box['h']
def key(p,label,color,x,y,w,size=11):
    shape(p,'ellipse',x,y+.09,.1,.1,color,0,color)
    text(p,label,x+.15,y,w-.15,size,.28,align='left')

p=panel('1  Hippocampal surface reconstruction',0,0,7,3.7)
text(p,'Sagittal T1w',.2,.53,2.7,14,bold=True)
text(p,'512-vertex surface',3.65,.53,3.05,14,bold=True)
for i,(n,c) in enumerate(zip(['Subiculum','CA1','CA2','CA3','CA4'],['#0000FF','#79D8EC','#00ECA8','#FF9900','#FF0000'])):
    key(p,n,c,3.6+(i%3)*1.02,.93+(i//3)*.28,1.02,10.5)
text(p,'Anterior',.22,.87,1.1,12);text(p,'Posterior',1.94,.87,1.1,12)
image(p,NEW+'sagittal_AP.png',.15,1.19,3,2.35)
image(p,OLD+'01_subfields.png',3.65,1.55,2.65,2.0)
arrow(p,3.2,2.3,3.75,2.3)

p=panel('2  Individual cortical parcellation',7,0,7,3.7)
text(p,'Kong2022 ArealMSHBM · 400 ROIs',.15,.53,6.65,14,bold=True)
abbr={'DefaultA':'Default A','DefaultB':'Default B','DefaultC':'Default C','Language':'Language',
 'ContA':'Control A','ContB':'Control B','ContC':'Control C','SalVenAttnA':'Sal/VAttn A',
 'SalVenAttnB':'Sal/VAttn B','DorsAttnA':'DorsAttn A','DorsAttnB':'DorsAttn B','Aud':'Auditory',
 'SomMotA':'SomMot A','SomMotB':'SomMot B','VisualA':'Visual A','VisualB':'Visual B','VisualC':'Visual C'}
for i,(name,c) in enumerate(meta['network'].items()):key(p,abbr[name],c,.18+(i%6)*1.12,.86+(i//6)*.23,1.12,10)
text(p,'Individual 1',.25,1.63,2.9,14);text(p,'Individual 2',3.5,1.63,2.9,14)
image(p,OLD+'02_individual_100610.png',.2,1.96,3,1.67)
image(p,OLD+'02_individual_102311.png',3.45,1.96,3,1.67)
text(p,'…',6.45,2.6,.4,24)

p=panel('3  Hippocampal-cortical FC',0,3.7,7,3.7)
text(p,'Vertex',.08,.53,1.35,14,bold=True)
text(p,'Cortical ROI',1.5,.53,2.15,14,bold=True)
text(p,'FC matrix · 400 ROIs',3.7,.53,3.12,14,bold=True)
text(p,'2 mm FWHM',.08,.88,1.5,12)
text(p,'512 vertices × 400 ROIs',3.72,.88,3.1,12)
hb=image(p,NEW+'hipp_neutral.png',.08,1.35,1.32,2.1)
cb=image(p,NEW+'cortex_seed_fc.png',1.42,.94,2.18,1.25)
mb=image(p,OLD+'03_fc_matrix.png',3.95,1.42,2.8,1.8)
# Fill the desired box while retaining all real matrix rows/columns.
mb.update(x=3.95,y=1.42,w=2.8,h=1.8)
hx,hy=point(hb,meta['hip_xy']); cx,cy=point(cb,meta['roi_xy'])
shape(p,'ellipse',cx-.11,cy-.1,.22,.2,'#226D99',1.8)
row=meta['vertex']; col=meta['roi']-1
ry=mb['y']+(row+.5)*mb['h']/512
cxmat=mb['x']+(col+.5)*mb['w']/400
shape(p,'rect',mb['x'],mb['y']+row*mb['h']/512,mb['w'],mb['h']/512,'#BB861C',1.2)
shape(p,'rect',mb['x']+col*mb['w']/400,mb['y'],mb['w']/400,mb['h'],'#226D99',1.2)
# Direct connections to the selected matrix row/column.
arrow(p,hx+.06,hy,3.95,ry,'#BB861C',1.4)
arrow(p,cx+.11,cy,cxmat,1.42,'#226D99',1.4)

p=panel('4  Connectivity gradients',7,3.7,7,3.7)
for i in range(3):
    text(p,f'Gradient {i+1}',.15+i*2.27,.57,2.12,15,bold=True)
    image(p,OLD+f'04_G{i+1}.png',.15+i*2.27,.96,2.12,2.57)

p=panel('5  Spectral clustering',0,7.4,7,3.4)
text(p,'Three subregions',4.6,.55,2.35,14,bold=True)
gb=image(p,NEW+'gradient_space.png',.1,.65,4.05,2.35)
for name,ends in meta['gradient_axes'].items():
    mid=[(ends[0][j]+ends[1][j])/2 for j in range(2)]
    gx=gb['x']+mid[0]*gb['w']; gy=gb['y']+(1-mid[1])*gb['h']
    if name=='Gradient 3':
        text(p,name,gx-.27,gy-.15,1.05,12)
        p['elements'][-1]['rotation']=270
    else:
        text(p,name,gx-.55,gy+.06,1.1,12)
        p['elements'][-1]['rotation']=18 if name=='Gradient 1' else 325
image(p,OLD+'05_group_regions.png',4.95,.9,1.7,2.4)
arrow(p,4.4,2.08,4.93,2.08)

p=panel('6  Connectivity templates',7,7.4,7,3.4)
text(p,'Mean cortical Fisher-z FC within each region',.15,.53,6.65,13,bold=True)
for i,(n,c) in enumerate(zip(['Anterior','Middle','Posterior'],['#DC5CFF','#6396C2','#D85C6B'])):
    x=.14+i*2.27
    text(p,n,x,.98,2.05,14,color=c,bold=True)
    image(p,OLD+f'06_system_{i+1}.png',x+.25,1.36,1.45,.8)
    arrow(p,x+1.04,2.2,x+1.04,2.39)
    image(p,OLD+f'06_template_{i+1}_lateral.png',x,2.48,1.1,.68)
    image(p,OLD+f'06_template_{i+1}_medial.png',x+1.13,2.48,1.1,.68)

p=panel('7  Individual template matching',0,10.8,14,3.9)
text(p,'Individual Fisher-z FC',.1,.58,2.3,13,bold=True)
text(p,'512 vertices × 400 ROIs',.12,.9,2.2,11.5)
image(p,NEW+'individual_fc.png',.18,1.32,2.12,1.03)
text(p,'HCP fixed templates',.1,2.44,2.3,13,bold=True)
text(p,'3 templates × 400 ROIs',.1,2.76,2.3,11.5)
image(p,NEW+'templates.png',.18,3.13,2.12,.45)
text(p,'Pearson correlation across 400 ROIs',2.5,.58,6.9,14,bold=True)
text(p,'Each vertex profile vs. each template',2.5,.92,6.9,13)
for i,(n,c) in enumerate(zip(['Anterior','Middle','Posterior'],['#DC5CFF','#6396C2','#D85C6B'])):
    text(p,n,3.5+i*1.78,1.34,1.6,14,color=c,bold=True)
    image(p,OLD+f'07_template_correlation_{i+1}.png',3.5+i*1.78,1.7,1.6,1.98)
shape(p,'line',2.38,1.83,.33,.57,'#89949E')
shape(p,'line',2.38,3.35,.33,-.95,'#89949E')
arrow(p,2.71,2.4,3.35,2.4)
text(p,'For each vertex',9.15,1.37,2.3,13,bold=True)
text(p,'Select the template\nwith highest correlation',9.05,1.88,2.6,12,h=.55)
arrow(p,9.16,2.65,11.4,2.65)
text(p,'Individual labels',11.48,.58,2.35,14,bold=True)
image(p,OLD+'07_individual_regions.png',11.7,1.12,1.93,2.48)

doc=dict(width=14,height=14.7,font='Arial',panels=PANELS)
(OUT/'layout.json').write_text(json.dumps(doc,indent=2))

# A metric-compatible preview is explicitly selected because local Arial lookup
# resolves to a broken WOFF. PPTX independently declares actual Arial.
fontpath='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
boldpath='/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
def draw(ax,p):
    ax.set_xlim(0,p['w']);ax.set_ylim(p['h'],0);ax.axis('off')
    ax.text(.15,.27,p['title'],fontproperties=FontProperties(fname=boldpath,size=21),va='center',color='#19232D')
    for e in p['elements']:
        x,y,w,h=e['x'],e['y'],e['w'],e['h']
        if e['type']=='image':ax.imshow(Image.open(OUT/e['path']),extent=(x,x+w,y+h,y),aspect='auto')
        elif e['type']=='text':
            tx=x if e['align']=='left' else x+w if e['align']=='right' else x+w/2
            ax.text(tx,y+h/2,e['text'],ha=e['align'],va='center',color=e['color'],rotation=-e.get('rotation',0),fontproperties=FontProperties(fname=boldpath if e['bold'] else fontpath,size=e['size']))
        elif e['type']=='arrow':ax.annotate('',xy=(x+w,y+h),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',color=e['color'],lw=e['width'],mutation_scale=10))
        elif e['type']=='line':ax.plot([x,x+w],[y,y+h],color=e['color'],lw=e['width'])
        elif e['type']=='rect':ax.add_patch(Rectangle((x,y),w,h,fill=e['fill'] is not None,facecolor=e['fill'] or 'none',edgecolor=e['color'],lw=e['width']))
        elif e['type']=='ellipse':ax.add_patch(Ellipse((x+w/2,y+h/2),w,h,fill=e['fill'] is not None,facecolor=e['fill'] or 'none',edgecolor=e['color'],lw=e['width']))
fig=plt.figure(figsize=(14,14.7),facecolor='white')
for p in PANELS:draw(fig.add_axes([p['x']/14,1-(p['y']+p['h'])/14.7,p['w']/14,p['h']/14.7]),p)
fig.savefig(OUT/'workflow_preview.png',dpi=160);fig.savefig(OUT/'workflow_preview.pdf');plt.close(fig)
for i,p in enumerate(PANELS,1):
    fig=plt.figure(figsize=(p['w'],p['h']),facecolor='white');draw(fig.add_axes([0,0,1,1]),p)
    fig.savefig(OUT/f'panel_{i}.png',dpi=180);plt.close(fig)
print('Compact layout and previews built. Preview font: Liberation Sans; PPTX font: Arial.')
