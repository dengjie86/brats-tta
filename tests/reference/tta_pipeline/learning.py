"""One-graph replay of Senior probability fusion, with frozen BN statistics."""
import itertools
import numpy as np
import torch
from brats_tta.native.efficient import soft_extent_adjoint
from .senior import predict_volume

FLOOR = 1e-30

class Layout:
    def __init__(self, image, patch=(128,128,128)):
        self.original = image.shape[1:]
        self.image = np.pad(image,[(0,0)]+[(0,max(0,p-n)) for p,n in zip(patch,self.original)])
        self.shape = self.image.shape[1:]
        starts=[]
        for n,p in zip(self.shape,patch):
            a=list(range(0,n-p+1,max(1,int(p*.5))))
            if a[-1]!=n-p:a.append(n-p)
            starts.append(a)
        self.slices=[tuple(slice(s,s+p) for s,p in zip(pos,patch)) for pos in itertools.product(*starts)]
        axes=[np.exp(-.5*((np.arange(p)-(p-1)/2)/(p/8))**2) for p in patch]
        w=axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:]
        self.weight=np.maximum(w/w.max(),1e-7).astype(np.float32)
        self.count=np.zeros(self.shape,np.float32)
        for sl in self.slices:self.count[sl]+=self.weight

    def replay(self, network, probability_gradient, dtype):
        full=np.zeros((4,*self.shape),np.float32)
        full[(slice(None),)+tuple(slice(0,n) for n in self.original)]=probability_gradient
        device=next(network.parameters()).device
        for sl in self.slices:
            x=torch.from_numpy(np.ascontiguousarray(self.image[(slice(None),*sl)])).unsqueeze(0).to(device)
            g=full[(slice(None),*sl)]*(self.weight/self.count[sl])[None]
            with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda',cache_enabled=False):
                logits=network(x)
            probability=logits.float().softmax(1)
            probability.backward(torch.from_numpy(np.ascontiguousarray(g)).unsqueeze(0).to(device))
            del logits,probability,x,g
        return len(self.slices)

def select_affine(network):
    network.eval().requires_grad_(False)
    for module in network.modules():
        if isinstance(module,torch.nn.BatchNorm3d):
            assert module.track_running_stats and module.running_mean is not None
            module.weight.requires_grad_(True);module.bias.requires_grad_(True)
    return {n:p for n,p in network.named_parameters() if p.requires_grad}

def adjoint(probability, reference_log, delta, edits, scale):
    p=torch.from_numpy(probability)
    z=p.clamp_min(FLOOR).log()
    grad,info=soft_extent_adjoint(z,reference_log,delta,edits,scale)
    # The logits for this model are log(fused patch probabilities). The extra
    # Jacobian is essential: the original native linear-logit replay is invalid.
    grad.div_(p.clamp_min(FLOOR));grad.masked_fill_(p<FLOOR,0)
    assert torch.isfinite(grad).all()
    return grad.numpy(),info

def adapt(network,image,source,delta,edits,dtype,callback):
    selected=select_affine(network)
    assert len(selected)==36 and sum(p.numel() for p in selected.values())==5888
    history=[];forwards=0;backwards=0
    if not edits.any():
        return source.copy(),selected,history,dict(learning_forward_patches=0,backward_patches=0)
    assert (~edits).any()
    reference=torch.from_numpy(source).clamp_min(FLOOR).log()
    delta=torch.from_numpy(delta);edits=torch.from_numpy(edits)
    layout=Layout(image);optimizer=torch.optim.Adam(selected.values(),lr=.001)
    current=source
    for step in range(1,9):
        callback('learning_adjoint',step)
        scale=1024.; retries=0
        while True:
            optimizer.zero_grad(set_to_none=True)
            gradient,info=adjoint(current,reference,delta,edits,scale)
            callback('learning_replay',step)
            calls=layout.replay(network,gradient,dtype);del gradient
            forwards+=calls;backwards+=calls
            finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected.values())
            if finite:break
            retries+=1;scale*=.5
            assert retries<=12,'Repeated nonfinite affine gradient; optimizer not stepped'
        norm=0.
        for p in selected.values():
            p.grad.div_(scale);norm+=float(p.grad.double().square().sum())
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in selected.values())
        history.append(dict(step=step,scale=scale,retries=retries,gradient_l2=norm**.5,**info))
        callback('learning_forward',step)
        current,n=predict_volume(network,image,dtype);forwards+=n
    return current,selected,history,dict(learning_forward_patches=forwards,backward_patches=backwards)
