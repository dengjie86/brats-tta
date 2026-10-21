"""Protocol-specific optimization. Native logit fusion and Senior probability fusion stay distinct."""

import numpy as np
import torch
from scipy.special import expit, softmax

from brats_tta.native.adaptation import select_affine as select_in_affine
from brats_tta.native.context import MODALITIES
from brats_tta.native.efficient import LearningLayout, forward_views, replay_learning, soft_extent_adjoint
from brats_tta.native.replay import native_layout

from . import projection
from .probability_fusion import predict_volume
from .probability_replay import FLOOR, Layout, adjoint


def finite_gradients(selected):
    return all(p.grad is not None and torch.isfinite(p.grad).all() for p in selected.values())


class NativeBackend:
    modalities = MODALITIES

    def __init__(self, predictor, learning_views=1):
        if learning_views not in (1, 8):
            raise ValueError("Native learning views must be one or eight.")
        self.predictor = predictor
        self.network = predictor.network
        self.learning_views = learning_views

    def predict(self, image):
        logits = forward_views(self.predictor, image, 8)
        return softmax(logits.float().cpu().numpy(), axis=0).astype(np.float32), logits

    def extent_target(self, probability, reference):
        return float(expit(projection.logit_gap(reference.cpu().numpy())).mean())

    def learn(self, image, probability, source_logits, delta, edits, progress):
        selected, _ = select_in_affine(self.network)
        if not edits.any():
            return probability.copy(), selected, []
        if edits.all():
            raise ValueError("soft_extent requires unedited anchor voxels.")
        layout = LearningLayout(native_layout(self.predictor, image), self.learning_views)
        reference = source_logits if self.learning_views == 8 else forward_views(self.predictor, image, 1)
        device = reference.device
        delta, edits = torch.from_numpy(delta).to(device), torch.from_numpy(edits).to(device)
        optimizer = torch.optim.Adam(list(selected.values()), lr=0.001)
        current, scale, history = reference, 1024.0, []
        for step in range(1, 9):
            progress("learning", step)
            skipped = []
            for _ in range(12):
                optimizer.zero_grad(set_to_none=True)
                gradient, info = soft_extent_adjoint(current, reference, delta, edits, scale)
                calls = replay_learning(self.network, layout, gradient, amp=device.type == "cuda")
                del gradient
                if finite_gradients(selected):
                    break
                skipped.append(scale)
                scale /= 2
            else:
                raise RuntimeError("Nonfinite native affine gradient after retries.")
            for parameter in selected.values():
                parameter.grad.div_(scale)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in selected.values()):
                raise RuntimeError("Nonfinite native affine parameter.")
            history.append(dict(step=step, loss=info, views=calls, scale=scale, retries=skipped))
            if self.learning_views == 8 or step < 8:
                current = forward_views(self.predictor, image, self.learning_views)
        if self.learning_views == 1:
            current = forward_views(self.predictor, image, 8)
        student = softmax(current.float().cpu().numpy(), axis=0).astype(np.float32)
        return student, selected, history


class SeniorBackend:
    modalities = ("t1c", "t2w", "t2f", "t1n")

    def __init__(self, network, dtype=torch.bfloat16, patch=(128, 128, 128)):
        self.network, self.dtype, self.patch = network, dtype, patch

    def predict(self, image):
        probability, _ = predict_volume(self.network, image, self.dtype, patch=self.patch)
        return probability, None

    def extent_target(self, probability, reference):
        return float(expit(projection.probability_gap(probability)).mean())

    def learn(self, image, probability, source_logits, delta, edits, progress):
        self.network.eval().requires_grad_(False)
        for module in self.network.modules():
            if isinstance(module, torch.nn.BatchNorm3d):
                if not module.affine or not module.track_running_stats or module.running_mean is None:
                    raise ValueError("Senior adaptation requires affine BN with source running statistics.")
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
        selected = {name: p for name, p in self.network.named_parameters() if p.requires_grad}
        if not selected:
            raise ValueError("No affine BN parameters selected.")
        if not edits.any():
            return probability.copy(), selected, []
        if edits.all():
            raise ValueError("soft_extent requires unedited anchor voxels.")
        reference = torch.from_numpy(probability).clamp_min(FLOOR).log()
        delta, edits = torch.from_numpy(delta), torch.from_numpy(edits)
        layout = Layout(image, self.patch)
        optimizer = torch.optim.Adam(selected.values(), lr=0.001)
        current, history = probability, []
        for step in range(1, 9):
            progress("learning_adjoint", step)
            scale, retries = 1024.0, 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                gradient, info = adjoint(current, reference, delta, edits, scale)
                progress("learning_replay", step)
                layout.replay(self.network, gradient, self.dtype)
                del gradient
                if finite_gradients(selected):
                    break
                retries += 1
                scale *= 0.5
                if retries > 12:
                    raise RuntimeError("Nonfinite Senior affine gradient after retries.")
            norm = 0.0
            for parameter in selected.values():
                parameter.grad.div_(scale)
                norm += float(parameter.grad.double().square().sum())
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in selected.values()):
                raise RuntimeError("Nonfinite Senior affine parameter.")
            history.append(dict(step=step, scale=scale, retries=retries, gradient_l2=norm**0.5, **info))
            progress("learning_forward", step)
            current, _ = self.predict(image)
        return current, selected, history
