import numpy as np
from brats_tta.native.context import teacher, VIEWS
from brats_tta.native.coordinates import crop, restore
def independent_teacher(source,probs):
    labels=np.argmax(np.stack(probs),axis=1)
    accepted=np.all(labels==labels[0],axis=0)
    for p,label in zip(probs,labels):
        top=p.max(0)
        other=p.copy();other[label,np.arange(len(label))]=-np.inf
        runnerup=other.max(0)
        accepted &= (top>=np.float32(.8)) & ((top-runnerup)>=np.float32(.2))
    return np.where(accepted,labels[0],source).astype(np.uint8),accepted
def rules(source,old,p0,e,index,independent=False):
    pair=[]
    for c in range(4):
        probs=[e[v][c] for v in VIEWS]
        value=independent_teacher(source,probs) if independent else teacher(source,probs,np.ones(len(source),bool))
        pair.append(value[0])
    pair=np.stack(pair);change=pair!=source[None]
    if independent:
        counts=np.stack([np.sum(change & (pair==k),axis=0) for k in range(4)])
    else:
        counts=(np.eye(4,dtype=np.int8)[pair]*change[:,:,None]).sum(0).T
    target=counts.argmax(0).astype(np.uint8)
    unique=(counts>0).sum(0)==1
    eligible=unique & (old==source)
    one=eligible & (counts.max(0)>=1);two=eligible & (counts.max(0)>=2)
    conf=[];ii=np.arange(len(source))
    for c in range(4):
        v=np.minimum(e[VIEWS[0]][c,target,ii],e[VIEWS[1]][c,target,ii])
        conf.append(np.where(change[c] & (pair[c]==target),v,-np.inf))
    confidence=np.max(conf,axis=0)
    margin=p0[source,ii]-p0[target,ii]
    masks=dict(union1=one,union2=two,confidence_matched=np.zeros_like(one),margin_matched=np.zeros_like(one))
    for s in range(4):
        for t in range(4):
            if s==t:continue
            domain=(source==s)&(target==t)
            candidates=np.flatnonzero(one&domain);n=int((two&domain).sum())
            for arm,score in [('confidence_matched',-confidence),('margin_matched',margin)]:
                order=np.argsort(score[candidates],kind='stable') if independent else np.lexsort((index[candidates],score[candidates]))
                masks[arm][candidates[order[:n]]]=True
                assert int((masks[arm]&domain).sum())==n
    assert not (two&~one).any()
    result={arm:np.where(mask,target,old).astype(np.uint8) for arm,mask in masks.items()}
    for value in result.values():np.testing.assert_array_equal(value[old!=source],old[old!=source])
    return result
def proposal_probabilities(source,labels,e,independent=False):
    pair=[]
    for c in range(4):
        ps=[e[v][c] for v in VIEWS]
        t=independent_teacher(source,ps)[0] if independent else teacher(source,ps,np.ones(len(source),bool))[0]
        pair.append(t)
    support=(np.stack(pair)==labels[None]) & (labels!=source)[None]
    counts=support.sum(0)
    assert (counts>0).all()
    if independent:
        # A separate per-column mean, avoiding the main masked reduction.
        values=np.stack([e[v][c].astype(np.float64) for c in range(4) for v in VIEWS])
        result=np.stack([values[np.repeat(support[:,j],2),:,j].mean(axis=0) for j in range(len(source))],axis=1) if len(source) else np.empty((4,0),np.float64)
    else:
        result=np.zeros((4,len(source)),np.float64)
        for c in range(4):
            for v in VIEWS:result+=e[v][c].astype(np.float64)*support[c][None]
        result/=2*counts[None]
    np.testing.assert_allclose(result.sum(0),1,atol=2e-6,rtol=0)
    np.testing.assert_array_equal(result.argmax(0),labels)
    return result

def extend(d,e,idx,source,old,labels,independent=False):
    extra=labels!=old.ravel()[idx]
    ni=idx[extra]
    np.testing.assert_array_equal(old.ravel()[ni],source.ravel()[ni])
    subset={v:e[v][:,:,extra] for v in VIEWS}
    q=proposal_probabilities(source.ravel()[ni],labels[extra],subset,independent)
    new=dict(d,qteacher=d['qteacher'].copy(),accepted=d['accepted'].copy())
    new['qteacher'].reshape(4,-1)[:,ni]=q
    new['accepted'].ravel()[ni]=True
    teacher=old.copy();teacher.ravel()[ni]=labels[extra]
    new['teacher']=restore(teacher,d['geo']['shape'],d['geo']['bbox'],d['geo']['fw'])
    old_edit=old!=source
    np.testing.assert_array_equal(teacher[old_edit],old[old_edit])
    np.testing.assert_array_equal(new['qteacher'][:,old_edit],d['qteacher'][:,old_edit])
    np.testing.assert_array_equal(new['accepted'][old_edit],d['accepted'][old_edit])
    unchanged=np.ones(old.size,bool);unchanged[ni]=False
    np.testing.assert_array_equal(new['qteacher'].reshape(4,-1)[:,unchanged],d['qteacher'].reshape(4,-1)[:,unchanged])
    np.testing.assert_array_equal(new['accepted'].ravel()[unchanged],d['accepted'].ravel()[unchanged])
    return new,ni,q
def proposal(d,e,independent=False):
    idx=e['index'];source=d['p'].argmax(0).astype(np.uint8)
    old=crop(d['teacher'],d['geo']['bbox'],d['geo']['fw'])
    labels=rules(source.ravel()[idx],old.ravel()[idx],d['p'].reshape(4,-1)[:,idx],e,idx,independent)['union2']
    return extend(d,e,idx,source,old,labels,independent)
