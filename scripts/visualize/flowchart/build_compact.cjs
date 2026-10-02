const pptxgen=require('pptxgenjs');
const fs=require('fs'),path=require('path');
const root=path.join(__dirname,'compact');
const doc=JSON.parse(fs.readFileSync(path.join(root,'layout.json'),'utf8'));
const pptx=new pptxgen();
pptx.defineLayout({name:'COMPACT',width:doc.width,height:doc.height});pptx.layout='COMPACT';
pptx.theme={headFontFace:'Arial',bodyFontFace:'Arial',lang:'en-US'};
pptx.author='Hippocampal workflow';pptx.title='Compact editable hippocampal workflow';
function add(p,s,ox,oy) {
 s.addText(p.title,{x:ox+.15,y:oy+.08,w:p.w-.3,h:.4,fontSize:21,fontFace:'Arial',bold:true,color:'19232D',margin:0});
 for(const e of p.elements){
  const x=ox+e.x,y=oy+e.y,w=e.w,h=e.h;
  if(e.type==='image')s.addImage({path:path.resolve(root,e.path),x,y,w,h});
  else if(e.type==='text')s.addText(e.text,{x,y,w,h,fontFace:'Arial',fontSize:e.size,color:e.color.replace('#',''),bold:e.bold,align:e.align,margin:0,vertAnchor:'ctr',rotate:e.rotation||0});
  else if(e.type==='arrow'||e.type==='line')s.addShape(pptx.ShapeType.line,{x:Math.min(x,x+w),y:Math.min(y,y+h),w:Math.abs(w),h:Math.abs(h),flipH:w<0,flipV:h<0,line:{color:e.color.replace('#',''),width:e.width,beginArrowType:'none',endArrowType:e.type==='arrow'?'triangle':'none'}});
  else s.addShape(e.type==='ellipse'?pptx.ShapeType.ellipse:pptx.ShapeType.rect,{x,y,w,h,line:{color:e.color.replace('#',''),width:e.width},fill:e.fill?{color:e.fill.replace('#','')}:{color:'FFFFFF',transparency:100}});
 }
}
const combined=pptx.addSlide();combined.background={color:'FFFFFF'};
doc.panels.forEach(p=>add(p,combined,p.x,p.y));
const meta=JSON.parse(fs.readFileSync(path.join(root,'asset_details.json'),'utf8'));
combined.addNotes(`Arial throughout. Panel 3: sub-100610 vertex ${meta.vertex} (zero-based; row ${meta.vertex+1}), cortical ROI ${meta.roi} (${meta.roi_name}); original matrix order. Exactly one real hippocampal mesh vertex has a yellow point scalar; no overlaid sphere or circle. Blue circle centered on a visible negative-valued vertex of the selected cortical parcel. Direct arrows and row/column boxes are editable. Panel 5 uses cropped orthographic projection with screen-vertical Gradient 3. Panel 7 uses sub-1016 Fisher-z FC (512×400), fixed HCP151 Fisher-z templates (3×400), Pearson correlation across 400 ROIs and highest-correlation assignment per vertex. Kember shares the Pearson/argmax decision rule, not the upstream feature space, transformation or clustering. See README and kember_matching_check.md for sources and limits.`);
pptx.writeFile({fileName:path.join(root,'workflow_editable.pptx')});
