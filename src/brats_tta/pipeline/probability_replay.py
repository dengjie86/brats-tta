import itertools
import numpy as np
import torch
from brats_tta.native.efficient import soft_extent_adjoint

FLOOR = 1e-30


class Layout:
    def __init__(self, image, patch=(128, 128, 128)):
        self.original = image.shape[1:]
        self.image = np.pad(image, [(0, 0)] + [(0, max(0, p - n)) for p, n in zip(patch, self.original)])
        self.shape = self.image.shape[1:]
        starts = []
        for n, p in zip(self.shape, patch):
            a = list(range(0, n - p + 1, max(1, int(p * 0.5))))
            if a[-1] != n - p:
                a.append(n - p)
            starts.append(a)
        self.slices = [
            tuple(slice(s, s + p) for s, p in zip(pos, patch)) for pos in itertools.product(*starts)
        ]
        axes = [np.exp(-0.5 * ((np.arange(p) - (p - 1) / 2) / (p / 8)) ** 2) for p in patch]
        w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
        self.weight = np.maximum(w / w.max(), 1e-7).astype(np.float32)
        self.count = np.zeros(self.shape, np.float32)
        for sl in self.slices:
            self.count[sl] += self.weight

    def replay(self, network, probability_gradient, dtype):
        full = np.zeros((4, *self.shape), np.float32)
        full[(slice(None),) + tuple(slice(0, n) for n in self.original)] = probability_gradient
        device = next(network.parameters()).device
        for sl in self.slices:
            x = torch.from_numpy(np.ascontiguousarray(self.image[(slice(None), *sl)])).unsqueeze(0).to(device)
            g = full[(slice(None), *sl)] * (self.weight / self.count[sl])[None]
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda", cache_enabled=False):
                logits = network(x)
            probability = logits.float().softmax(1)
            probability.backward(torch.from_numpy(np.ascontiguousarray(g)).unsqueeze(0).to(device))
            del logits, probability, x, g
        return len(self.slices)


def adjoint(probability, reference_log, delta, edits, scale):
    p = torch.from_numpy(probability)
    z = p.clamp_min(FLOOR).log()
    grad, info = soft_extent_adjoint(z, reference_log, delta, edits, scale)
    # The logits for this model are log(fused patch probabilities). The extra
    # Jacobian is essential: the original native linear-logit replay is invalid.
    grad.div_(p.clamp_min(FLOOR))
    grad.masked_fill_(p < FLOOR, 0)
    assert torch.isfinite(grad).all()
    return grad.numpy(), info
