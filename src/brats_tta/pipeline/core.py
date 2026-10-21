"""Backend-independent orchestration; no GT or experiment-specific file lookup."""

import numpy as np

from . import projection
from .decoder import decode_no_come
from .state import SourceState
from .teacher import observe


def run_case(backend, image, geometry, progress=None):
    """Source -> ten observations -> union2 -> eight updates -> no_come -> restore.

    image is preprocessed in the backend's own channel/grid convention.
    Returned labels use the original image grid; affine values are copies.
    """
    if image.ndim != 4 or image.shape[0] != 4 or not np.isfinite(image).all():
        raise ValueError("Expected finite four-channel preprocessed MRI.")
    progress = progress or (lambda phase, step: None)
    with SourceState(backend.network) as source_state:
        progress("source", None)
        probability, reference = backend.predict(image)
        target = backend.extent_target(probability, reference)
        data, delta, edits, info, additions = observe(backend, image, probability, geometry, progress)
        source_state.check()
        student, selected, history = backend.learn(image, probability, reference, delta, edits, progress)
        source_state.check(selected.values())
        affine = {name: value.detach().cpu().numpy().copy() for name, value in selected.items()}
        progress("decode", None)
        fit = projection.fit(projection.probability_gap(student), target)
        prediction = decode_no_come(data, student, fit["bias"])
        result = dict(
            prediction=prediction,
            source=data["source"],
            teacher=data["teacher"],
            affine=affine,
            history=history,
            context=info,
            projection=fit,
            additions=additions,
            updates=len(history),
        )
    result["source_restored"] = True
    return result
