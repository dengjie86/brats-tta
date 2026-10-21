"""Full union2 teacher construction, including its original FP32/FP64 boundaries."""

import numpy as np

from brats_tta.native.adaptation import delta_logits
from brats_tta.native.context import VIEWS, make_contexts, teacher
from brats_tta.native.coordinates import crop, restore

from .teacher_rules import proposal


def assemble_teacher(probability, base, protected, geometry, *, evidence=None, additions=None):
    """Build teacher/target from fresh evidence or explicitly supplied cached additions."""
    if (evidence is None) == (additions is None):
        raise ValueError("Supply exactly one of evidence or additions.")
    p = probability
    source = p.argmax(0).astype(np.uint8)
    old, accepted = teacher(source, base, protected)
    data = dict(
        p=p,
        source=restore(source, **geometry),
        teacher=restore(old, **geometry),
        accepted=accepted,
        geo=geometry,
        qteacher=(base[0].astype(np.float64) + base[1].astype(np.float64)) * 0.5,
    )
    if evidence is not None:
        data, index, extra = proposal(data, evidence)
    else:
        index, extra = additions
        index = np.asarray(index)
        extra = np.asarray(extra)
        if (
            index.ndim != 1
            or not np.issubdtype(index.dtype, np.integer)
            or extra.shape != (4, len(index))
            or len(np.unique(index)) != len(index)
            or (index < 0).any()
            or (index >= source.size).any()
        ):
            raise ValueError("Malformed teacher additions.")
        if not protected.ravel()[index].all() or (old.ravel()[index] != source.ravel()[index]).any():
            raise ValueError("Additions must be protected and preserve original teacher edits.")
        if (
            not np.isfinite(extra).all()
            or (extra < 0).any()
            or not np.allclose(extra.sum(0), 1, atol=2e-6, rtol=0)
        ):
            raise ValueError("Invalid addition probabilities.")
        labels = old.copy()
        labels.ravel()[index] = extra.argmax(0)
        data["qteacher"].reshape(4, -1)[:, index] = extra
        data["accepted"].ravel()[index] = True
        data["teacher"] = restore(labels, **geometry)
    observations = [value.copy() for value in base]
    for value in observations:
        value.reshape(4, -1)[:, index] = extra.astype(np.float32)
    labels, accepted = teacher(source, observations, protected)
    np.testing.assert_array_equal(labels, crop(data["teacher"], geometry["bbox"], geometry["fw"]))
    np.testing.assert_array_equal(accepted, data["accepted"])
    edits = labels != source
    delta = delta_logits((observations[0] + observations[1]) * 0.5, p)
    delta[:, ~edits] = 0
    info = dict(
        base_edits=int((old != source).sum()),
        added_union2_voxels=len(index),
        edited_voxels=int(edits.sum()),
        accepted_voxels=int(accepted.sum()),
    )
    return data, delta, edits, info, (index, extra)


def observe(backend, image, probability, geometry, progress):
    source = probability.argmax(0).astype(np.uint8)
    contexts, protected, info = make_contexts(image, source)
    index = np.flatnonzero(protected)
    evidence = dict(index=index)
    base = []
    for name in VIEWS:
        context = contexts[name]
        progress("teacher_" + name, None)
        base.append(backend.predict(context)[0])
        complementary = []
        for channel, modality in enumerate(backend.modalities):
            progress("teacher_" + name + "_keep_" + modality, None)
            modified = context.copy()
            modified[channel] = image[channel]
            p, _ = backend.predict(modified)
            complementary.append(p.reshape(4, -1)[:, index])
            del modified, p
        evidence[name] = np.stack(complementary)
    result = assemble_teacher(probability, base, protected, geometry, evidence=evidence)
    result[3].update(info, context_volume_predictions=10)
    return result
