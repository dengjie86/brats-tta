"""Restore an episodic source model on both success and failure."""

import torch


class SourceState:
    def __init__(self, network):
        self.network = network
        self.values = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        self.parameters = [
            (p, p.requires_grad, None if p.grad is None else p.grad.detach().clone())
            for p in network.parameters()
        ]
        self.training = [(m, m.training) for m in network.modules()]

    def __enter__(self):
        self.network.eval().requires_grad_(False)
        return self

    def check(self, selected=()):
        identities = {id(p) for p in selected}
        aliases = {
            name for name, p in self.network.named_parameters(remove_duplicate=False) if id(p) in identities
        }
        for name, value in self.network.state_dict().items():
            if name not in aliases and not torch.equal(value.detach().cpu(), self.values[name]):
                raise RuntimeError(f"Unexpected source state change: {name}")

    def __exit__(self, exc_type, exc, traceback):
        self.network.load_state_dict(self.values, strict=True)
        for parameter, required, gradient in self.parameters:
            parameter.requires_grad_(required)
            parameter.grad = gradient
        for module, training in self.training:
            module.training = training
        return False
