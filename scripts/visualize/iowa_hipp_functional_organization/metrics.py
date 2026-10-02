"""Numerical summaries with explicit missingness; see numerical_notes.md."""
import heapq
import numpy as np
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from scipy.special import gammaln

SYSTEMS = ('A', 'B', 'P')
NETWORKS = ('DMN', 'SMN', 'VAN', 'DAN', 'CON', 'LAN', 'VN')


def significance(p):
    if p is None or not np.isfinite(p):
        return 'NA'
    if not 0 <= p <= 1:
        raise ValueError('p must be in [0, 1]')
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '+' if p < .1 else ''


def welch(patient, hc):
    x, y = (np.asarray(v, float) for v in (patient, hc))
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    out = dict(mean_patient=float(x.mean()) if len(x) else np.nan,
               sd_patient=float(x.std(ddof=1)) if len(x)>1 else np.nan, n_patient=len(x),
               mean_hc=float(y.mean()) if len(y) else np.nan,
               sd_hc=float(y.std(ddof=1)) if len(y)>1 else np.nan, n_hc=len(y),
               difference=np.nan, ci_low=np.nan, ci_high=np.nan, g=np.nan,p=np.nan,df=np.nan,reason='')
    out['difference'] = out['mean_patient']-out['mean_hc']
    if min(len(x),len(y)) < 2:
        out['reason']='fewer than two finite participants in a group'; return out
    vx,vy=x.var(ddof=1),y.var(ddof=1)
    se2=vx/len(x)+vy/len(y)
    if se2<=0:
        out['reason']='zero standard error'; return out
    df=se2**2/((vx/len(x))**2/(len(x)-1)+(vy/len(y))**2/(len(y)-1))
    se=np.sqrt(se2); delta=out['difference']; half=stats.t.ppf(.975,df)*se
    pooled=np.sqrt(((len(x)-1)*vx+(len(y)-1)*vy)/(len(x)+len(y)-2))
    nu=len(x)+len(y)-2
    correction=np.exp(gammaln(nu/2)-.5*np.log(nu/2)-gammaln((nu-1)/2))
    out.update(ci_low=delta-half,ci_high=delta+half,p=2*stats.t.sf(abs(delta/se),df),df=df,g=correction*delta/pooled)
    return out


def _inputs(labels, area):
    labels=np.asarray(labels); area=np.asarray(area,float)
    if labels.ndim!=1 or labels.shape!=area.shape or not np.isin(labels,[0,1,2,3]).all():
        raise ValueError('labels must be one-dimensional 0/1/2/3 with matching area')
    if not np.isfinite(area).all() or (area<0).any() or area.sum()<=0:
        raise ValueError('area must be finite nonnegative with positive whole-side area')
    return labels,area


def _valid_mask(valid, n):
    if valid is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(valid)
    if mask.shape != (n,) or not np.isin(mask, (0, 1, False, True)).all():
        raise ValueError("valid mask must be one-dimensional binary with one value per vertex")
    return mask.astype(bool)



def _row(system, metric, value, reason='', **extra):
    return dict(system=system,metric=metric,value=float(value),reason=reason,**extra)


def _weighted(values, area, selected, system, metric):
    valid=selected & np.isfinite(values) & (area>0)
    va=float(area[valid].sum()); total=float(area[selected].sum())
    return _row(system,metric,np.average(values[valid],weights=area[valid]) if va else np.nan,
                '' if va else 'no finite positive-area vertices',valid_area=va,total_area=total)


def identity(scores, labels, area, valid=None):
    labels,area=_inputs(labels,area); scores=np.asarray(scores,float)
    mask=_valid_mask(valid,len(labels))
    if scores.shape!=(len(labels),3): raise ValueError('scores must be vertices x 3')
    good=mask & np.isfinite(scores).all(axis=1)
    ordered=np.sort(scores,axis=1)
    vals={'smax':ordered[:,2], 'margin':ordered[:,2]-ordered[:,1]}
    rows=[]
    selections=[('whole',np.ones(len(labels),bool))]
    selections.extend((s,mask & (labels==i)) for i,s in enumerate(SYSTEMS,1))
    for name,selected in selections:
        for metric,v in vals.items(): rows.append(_weighted(np.where(good,v,np.nan),area,selected,name,metric))
    return rows


def fixed_hcp_identity(scores, hcp_labels, hcp_support, area, valid=None):
    """Summarize HCP-template scores within fixed, external HCP territories."""
    labels,area=_inputs(hcp_labels,area); scores=np.asarray(scores,float)
    support=_valid_mask(hcp_support,len(labels))
    if not np.array_equal(labels != 0, support):
        raise ValueError('HCP labels/support must agree exactly on the fixed territory')
    if scores.shape!=(len(labels),3): raise ValueError('scores must be vertices x 3')
    good=_valid_mask(valid,len(labels)) & np.isfinite(scores).all(axis=1)
    rows=[]
    for j,s in enumerate(SYSTEMS):
        own=scores[:,j]
        advantage=own-np.max(scores[:,[k for k in range(3) if k!=j]],axis=1)
        selected=support & (labels==j+1)
        for metric,value in [('fixed_hcp_similarity',own),('fixed_hcp_advantage',advantage)]:
            rows.append(_weighted(np.where(good,value,np.nan),area,selected,s,metric))
    return rows


def _bins(axis,n):
    axis=np.asarray(axis,float)
    if axis.shape!=(n,) or ((axis[np.isfinite(axis)]<0)|(axis[np.isfinite(axis)]>1)).any():
        raise ValueError('axis must match vertices and lie within [0,1]')
    return axis,[(np.isfinite(axis)&(axis>=i/20)&(axis<(i+1)/20 if i<19 else axis<=1)) for i in range(20)]


def spatial(labels, area, axis, valid=None):
    labels,area=_inputs(labels,area)
    mask=_valid_mask(valid,len(labels))
    axis,bins=_bins(axis,len(labels)); rows=[]; curves=[]
    full_area=float(area.sum()); valid_area=float(area[mask].sum())
    for code,s in [(1,'A'),(2,'B'),(3,'P'),(0,'unassigned')]:
        selected=mask & (labels==code); amount=float(area[selected].sum())
        extras=dict(valid_area=valid_area,total_area=full_area)
        rows.extend([
            _row(s,'area_fraction',amount/full_area,**extras),
            _row(s,'area_mm2',amount,**extras),
            _row(s,'vertex_count',selected.sum(),**extras),
            _weighted(axis,area,selected,s,'centroid'),
        ])
        for b,inside in enumerate(bins):
            denominator=float(area[inside].sum())
            curves.append(dict(
                system=s,metric='area_fraction',bin=b,axis=(b+.5)/20,
                value=float(area[inside&selected].sum()/denominator) if denominator else np.nan,
                reason='' if denominator else 'empty coordinate bin',
                valid_area=float(area[inside&mask].sum()),total_area=denominator,
            ))
    return rows,curves


def boundary_positions(labels, axis, faces, valid=None):
    """Return every valid assigned-label boundary edge's AP midpoint."""
    labels = np.asarray(labels)
    n = len(labels)
    if labels.shape != (n,) or not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError('boundary labels must be one-dimensional 0/1/2/3')
    mask = _valid_mask(valid, n)
    axis, _ = _bins(axis, n)
    faces = np.asarray(faces)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or (faces < 0).any()
        or (faces >= n).any()
    ):
        raise ValueError('boundary faces must be in-range triangles')
    edges = set()
    for triangle in faces:
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edges.add(tuple(sorted((int(first), int(second)))))
    rows = []
    for first, second in sorted(edges):
        if not (mask[first] and mask[second] and np.isfinite(axis[first]) and np.isfinite(axis[second])):
            continue
        codes = (int(labels[first]), int(labels[second]))
        if 0 in codes or codes[0] == codes[1]:
            continue
        low, high = sorted(codes)
        rows.append(
            dict(
                system_pair=f'{SYSTEMS[low - 1]}_{SYSTEMS[high - 1]}',
                position=float((axis[first] + axis[second]) / 2.0),
                reason='',
            )
        )
    return rows


def template_axis(scores, area, axis, valid=None):
    scores=np.asarray(scores,float); area=np.asarray(area,float)
    _inputs(np.zeros(len(area)),area)
    mask=_valid_mask(valid,len(area))
    if scores.shape!=(len(area),3): raise ValueError('scores must be vertices x 3')
    axis,bins=_bins(axis,len(area)); rows=[]; curves=[]
    for s,v in zip(('rA','rB','rP','rP_minus_rA'),[*scores.T,scores[:,2]-scores[:,0]]):
        masked_values=np.where(mask,v,np.nan)
        good=mask & np.isfinite(v)&np.isfinite(axis)&(area>0)
        extras=dict(valid_area=float(area[good].sum()),total_area=float(area.sum()))
        slope=np.nan; reason='fewer than two finite coordinate values'
        if good.sum()>=2:
            x=axis[good]; y=v[good]; w=area[good]; dx=x-np.average(x,weights=w)
            if np.sum(w*dx*dx)>0:
                slope=np.sum(w*dx*(y-np.average(y,weights=w)))/np.sum(w*dx*dx);reason=''
            else: reason='constant longitudinal coordinate'
        rows.append(_row(s,'slope',slope,reason,**extras))
        rows.append(_row(s,'robust_range',np.ptp(np.percentile(v[good],[5,95])) if good.any() else np.nan,'' if good.any() else 'no finite vertices',**extras))
        for b,inside in enumerate(bins):
            row=_weighted(masked_values,area,inside,s,'mean'); row.update(bin=b,axis=(b+.5)/20); curves.append(row)
    return rows,curves


def _corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.size<3 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all(): return np.nan
    x=x-x.mean(); y=y-y.mean(); denominator=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.clip(np.dot(x,y)/denominator,-1,1)) if denominator>0 else np.nan


def regional_fc(ts, roi, labels, valid, systems=(1,2,3)):
    ts=np.asarray(ts,float); roi=np.asarray(roi,float); labels=np.asarray(labels);valid=np.asarray(valid,bool)
    if ts.ndim!=2 or roi.shape!=(400,ts.shape[1]) or labels.shape!=(len(ts),) or valid.shape!=labels.shape:
        raise ValueError('regional FC requires vertices x time and 400 ROI x matching time')
    z=np.full((len(systems),400),np.nan);reasons=[]
    for k,code in enumerate(systems):
        use=valid&(labels==code)
        if not use.any(): reasons.append('empty valid system');continue
        if not np.isfinite(ts[use]).all(): reasons.append('nonfinite regional time series');continue
        mean=ts[use].mean(axis=0)
        r=np.array([_corr(mean,target) for target in roi])
        good=np.isfinite(r)&(np.abs(r)<1)
        z[k,good]=np.arctanh(r[good])
        reasons.append('' if good.all() else 'noncomputable ROI correlations (constant, nonfinite, or perfect)')
    return z,reasons


def network_fc(z, networks):
    z=np.asarray(z,float); networks=np.asarray(networks)
    if z.ndim!=2 or z.shape[1]!=400 or networks.shape!=(400,) or set(networks)!=set(NETWORKS):
        raise ValueError('exact 400-ROI registry must contain all and only DMN SMN VAN DAN CON LAN VN')
    out=np.full((len(z),7),np.nan)
    for j,name in enumerate(NETWORKS):
        subset=z[:,networks==name]; good=np.isfinite(subset).all(axis=1)
        out[good,j]=subset[good].mean(axis=1)
    return out,list(NETWORKS)


def aligned_gradient_metrics(raw, reference, axis, valid):
    raw=np.asarray(raw,float); ref=np.asarray(reference,float); valid=np.asarray(valid,bool);axis=np.asarray(axis,float)
    if raw.ndim!=2 or raw.shape[1]!=5 or ref.shape!=raw.shape or valid.shape!=(len(raw),) or axis.shape!=valid.shape:
        raise ValueError('alignment requires matching vertices x 5 gradients, axis and valid mask')
    common=valid&np.isfinite(raw).all(axis=1)&np.isfinite(ref).all(axis=1)
    aligned=np.full(raw.shape,np.nan);rows=[]
    reason=''
    if common.sum()<5: reason='fewer than five common finite vertices'
    elif np.linalg.matrix_rank(raw[common])<5 or np.linalg.matrix_rank(ref[common])<5: reason='rank-deficient five-gradient alignment'
    else:
        rotation,_=orthogonal_procrustes(raw[common],ref[common]); use=valid&np.isfinite(raw).all(axis=1)
        aligned[use]=raw[use]@rotation
    for j,s in enumerate(('G1','G2','G3')):
        good=np.isfinite(aligned[:,j]); paired=good&np.isfinite(axis)
        values={'robust_range':np.ptp(np.percentile(aligned[good,j],[5,95])) if good.any() else np.nan,
                'axis_correlation':_corr(aligned[paired,j],axis[paired]),'reference_correlation':_corr(aligned[common,j],ref[common,j])}
        for metric,value in values.items():rows.append(_row(s,metric,value,reason or ('' if np.isfinite(value) else 'insufficient or constant finite values')))
    return aligned,rows


def _cohen(x,y):
    x=np.asarray(x);y=np.asarray(y);x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    if min(len(x),len(y))<2:return np.nan
    sd=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/(len(x)+len(y)-2))
    return float((x.mean()-y.mean())/sd) if sd>0 else np.nan


def kember(ts, labels, valid, points, faces, radius=4):
    ts=np.asarray(ts,float); labels=np.asarray(labels);valid=np.asarray(valid,bool);points=np.asarray(points,float);faces=np.asarray(faces,int)
    n=len(labels)
    if ts.ndim!=2 or len(ts)!=n or valid.shape!=(n,) or points.shape!=(n,3) or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(points).all() or (faces<0).any() or (faces>=n).any() or radius<=0:
        raise ValueError('invalid Kember surface, time-series, mask or radius')
    good=valid&np.isin(labels,[1,2,3])&np.isfinite(ts).all(axis=1)&(np.std(ts,axis=1)>0)
    edges=set()
    for face in faces:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):edges.add(tuple(sorted((int(a),int(b)))))
    graph=[[] for _ in range(n)]
    for a,b in edges:
        distance=float(np.linalg.norm(points[a]-points[b]));graph[a].append((b,distance));graph[b].append((a,distance))
    def neighborhood(seed):
        distances={seed:0.}; queue=[(0.,seed)]
        while queue:
            d,u=heapq.heappop(queue)
            if d!=distances[u]:continue
            for v,length in graph[u]:
                nd=d+length
                if nd<=radius and nd<distances.get(v,np.inf):distances[v]=nd;heapq.heappush(queue,(nd,v))
        return set(distances)
    correlations={}
    for code in (1,2,3):
        own=good&(labels==code)
        if own.any():
            mean=ts[own].mean(axis=0);correlations[code]=np.array([_corr(v,mean) if ok else np.nan for v,ok in zip(ts,good)])
    boundary={c:[] for c in (1,2,3)}
    for a,b in sorted(edges):
        if not(good[a] and good[b]) or labels[a]==labels[b]:continue
        nodes=np.array(sorted(neighborhood(a)|neighborhood(b)))
        left=nodes[good[nodes]&(labels[nodes]==labels[a])];right=nodes[good[nodes]&(labels[nodes]==labels[b])]
        if min(len(left),len(right))<2:continue
        # Local discriminability uses each side's mean as its target, symmetrically.
        lm=ts[left].mean(axis=0);rm=ts[right].mean(axis=0)
        dl=_cohen([_corr(ts[v],lm) for v in left],[_corr(ts[v],lm) for v in right])
        dr=_cohen([_corr(ts[v],rm) for v in right],[_corr(ts[v],rm) for v in left])
        if np.isfinite(dl) and np.isfinite(dr):
            value=(dl+dr)/2;boundary[int(labels[a])].append(value);boundary[int(labels[b])].append(value)
    rows=[]
    for code,s in enumerate(SYSTEMS,1):
        values=correlations.get(code,np.full(n,np.nan));seg=_cohen(values[good&(labels==code)],values[good&(labels!=code)])
        rows.append(_row(s,'segregation',seg,'' if np.isfinite(seg) else 'insufficient vertices or zero pooled SD',n_valid_boundaries=len(boundary[code])))
        bs=boundary[code];rows.append(_row(s,'boundary_sharpness',np.mean(bs) if bs else np.nan,'' if bs else 'no valid boundaries with >=2 vertices per side within radius',n_valid_boundaries=len(bs)))
    return rows
