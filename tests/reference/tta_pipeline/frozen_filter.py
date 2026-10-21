"""Local support for proposed changes, with a per-transition matched margin control."""
import numpy as np
from scipy.ndimage import binary_opening,generate_binary_structure


def apply(source,proposed,connectivity=1):
    s,r=np.asarray(source),np.asarray(proposed)
    if s.shape!=r.shape or s.ndim!=3 or any(n==0 for n in s.shape):
        raise ValueError('Expected equally shaped nonempty 3D labels')
    if not np.isin(s,range(4)).all() or not np.isin(r,range(4)).all() or connectivity not in (1,3):
        raise ValueError('Expected labels 0..3 and connectivity 1 or 3')
    positions=np.where(s!=r)
    if not positions[0].size:return s.astype(np.uint8,copy=True)
    # Opening has dependency radius two. Halo preserves full-volume results on every changed voxel.
    box=tuple(slice(max(0,int(x.min())-2),min(s.shape[i],int(x.max())+3)) for i,x in enumerate(positions))
    region=r[box]
    padded=np.pad(region,2,mode='edge')
    keep=np.zeros(region.shape,bool)
    for k in range(4):
        opened=binary_opening(padded==k,generate_binary_structure(3,connectivity))[2:-2,2:-2,2:-2]
        assert not (opened&(region!=k)).any()
        keep|=opened
    result=s.astype(np.uint8,copy=True)
    result[box]=np.where(keep,region,s[box])
    return result


def matched_margin(source,proposed,reference,margin):
    s,r,a,v=(np.asarray(x) for x in (source,proposed,reference,margin))
    assert s.shape==r.shape==a.shape==v.shape and np.isfinite(v).all()
    assert not ((a!=s)&(a!=r)).any()
    result=np.array(s,dtype=np.uint8,order='C',copy=True)
    for old in range(4):
        for new in range(4):
            if old==new:continue
            candidates=(s==old)&(r==new)
            n=int((candidates&(a==new)).sum())
            idx=np.flatnonzero(candidates)
            order=np.argsort(v.ravel()[idx],kind='stable')
            result.ravel()[idx[order[:n]]]=new
    return result
