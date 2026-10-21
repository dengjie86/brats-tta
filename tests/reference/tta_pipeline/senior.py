import itertools
import numpy as np
import torch
PATCH = (128, 128, 128)

def predict_volume(network, image, dtype, adapter=None, callback=None, patch=PATCH):
    original = image.shape[1:]
    pads = [(0, max(0,p-n)) for p,n in zip(patch,original)]
    if any(p[1] for p in pads):
        image = np.pad(image, [(0,0),*pads])
    shape = image.shape[1:]
    starts = []
    for n,p in zip(shape,patch):
        values = list(range(0,n-p+1,max(1,int(p*.5))))
        if values[-1] != n-p:
            values.append(n-p)
        starts.append(values)
    positions = list(itertools.product(*starts))
    axes = [np.exp(-.5*((np.arange(p)-(p-1)/2)/(p/8))**2) for p in patch]
    weight = axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:]
    weight = np.maximum(weight/weight.max(),1e-7).astype(np.float32)
    probs_sum = np.zeros((4,*shape),np.float32)
    count = np.zeros(shape,np.float32)
    device = next(network.parameters()).device
    for index,pos in enumerate(positions):
        slices = tuple(slice(s,s+p) for s,p in zip(pos,patch))
        x = torch.from_numpy(np.ascontiguousarray(image[(slice(None),*slices)])).unsqueeze(0).to(device)
        if adapter is None:
            with torch.no_grad(), torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda',cache_enabled=False):
                logits = network(x)
        else:
            logits = adapter.predict(x)
        with torch.no_grad():
            assert torch.isfinite(logits).all(), 'Nonfinite logits'
            probs = logits.float().softmax(1)[0].cpu().numpy()
        probs_sum[(slice(None),*slices)] += probs*weight
        count[slices] += weight
        del x, logits, probs
        if callback:
            callback(index+1,len(positions))
    assert np.all(count > 0)
    crop = tuple(slice(0,n) for n in original)
    return (probs_sum/count[None])[(slice(None),*crop)], len(positions)
