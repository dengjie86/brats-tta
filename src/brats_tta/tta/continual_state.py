"""Commit metrics and continual optimizer state together at patient boundaries."""
import io
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch

from brats_tta.utils.atomic_io import atomic_write_bytes


@contextmanager
def source_affine_parameters(adapter):
    """Temporarily expose source affine without resetting continual Adam state.

    Valid for this frozen-convolution, untracked-IN model. Always restore the
    current patient-stream state, including when baseline inference raises.
    """
    current = [p.detach().clone() for p in adapter.parameters]
    try:
        with torch.no_grad():
            for p, source in zip(adapter.parameters, adapter._source_parameters):
                p.copy_(source)
        yield adapter.model
    finally:
        with torch.no_grad():
            for p, saved in zip(adapter.parameters, current):
                p.copy_(saved)


def save_state(path, adapter, settings, records):
    state = dict(settings=settings, records=records,
        names=adapter.parameter_names,
        affine=[p.detach().cpu().clone() for p in adapter.parameters],
        optimizer=deepcopy(adapter.optimizer.state_dict()),
        scaler=adapter.scaler.state_dict(), rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    buffer = io.BytesIO()
    torch.save(state, buffer)
    atomic_write_bytes(Path(path), buffer.getvalue())


def load_state(path, adapter, settings):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['settings'] != settings or state['names'] != adapter.parameter_names:
        raise ValueError('Continual checkpoint protocol or affine names mismatch')
    if len(state['affine']) != len(adapter.parameters):
        raise ValueError('Continual checkpoint parameter count mismatch')
    with torch.no_grad():
        for p, saved in zip(adapter.parameters, state['affine']):
            if p.shape != saved.shape:
                raise ValueError('Continual checkpoint parameter shape mismatch')
            p.copy_(saved)
    adapter.optimizer.load_state_dict(state['optimizer'])
    adapter.scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['rng'])
    if state['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['records']
