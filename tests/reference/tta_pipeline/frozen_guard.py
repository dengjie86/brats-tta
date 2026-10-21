import numpy as np


def guard(filtered,proposed):
    a,r=np.asarray(filtered),np.asarray(proposed)
    if a.shape!=r.shape or a.ndim!=3 or not np.isin(a,range(4)).all() or not np.isin(r,range(4)).all():
        raise ValueError('Expected aligned 3D labels 0..3')
    result=a.copy();required=set(map(int,np.unique(r)));restored=[]
    for _ in range(4):
        missing=required-set(map(int,np.unique(result)))
        if not missing:break
        mask=np.isin(r,list(missing));result[mask]=r[mask];restored.extend(sorted(missing))
    assert required<=set(map(int,np.unique(result)))
    assert len(restored)==len(set(restored))<=4
    return result,restored


def independent_guard(filtered,proposed):
    a,r=np.asarray(filtered),np.asarray(proposed)
    table=np.bincount((a.astype(np.int64)*4+r).ravel(),minlength=16).reshape(4,4)
    required=table.sum(0)>0;valid=[]
    for code in range(16):
        selected=[k for k in range(4) if code&(1<<k)]
        counts=np.zeros(4,np.int64)
        for old in range(4):
            for target in range(4):counts[target if target in selected else old]+=table[old,target]
        if np.all(counts[required]>0):valid.append(selected)
    best=min(map(len,valid));minimal=[v for v in valid if len(v)==best]
    assert len(minimal)==1
    selected=minimal[0];result=a.copy();mask=np.isin(r,selected);result[mask]=r[mask]
    return result,selected
