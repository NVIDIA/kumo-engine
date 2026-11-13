# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native fixed-graph response experiments with explicit checkpoint binding."""

import hashlib
import tempfile
from pathlib import Path

from kumo_core.explain.perturbation import (
    RecordRemovalPlan,
    record_removal_responses,
)


class NativeRecordRemovalModel:
    """Restore a trusted checkpoint into an initialized native node module.

    The module must have the original architecture and fitted encoders. This
    low-level adapter does not reconstruct them from an Engine model path.
    Checkpoint loading is explicit; ordinary evidence inspection never loads
    executable model state. Use only checkpoints from a trusted producer.
    """
    def __init__(self, module, *, checkpoint: str | Path,
                 checkpoint_sha256: str):
        import torch

        path = Path(checkpoint)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != checkpoint_sha256:
            raise ValueError('Checkpoint integrity mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        module.load_state_dict(state['state_dict'], strict=True)
        self.module = module.cpu().eval()
        self.checkpoint_sha256 = checkpoint_sha256

    def responses(self, sample: dict, record: dict, *, plan: RecordRemovalPlan,
                  expected_checkpoint_sha256: str):
        if expected_checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError('Evidence belongs to a different checkpoint')
        return record_removal_responses(self.module, sample,
                                        prediction=record['prediction'],
                                        prediction_id=record['prediction_id'],
                                        plan=plan)


__all__ = ['RecordRemovalPlan', 'NativeRecordRemovalModel']


def responses_from_trainer(trainer, sample: dict, record: dict,
                           provenance: dict, plan: RecordRemovalPlan):
    """Reconstruct the exact saved Engine model without source data access."""
    from kumo_core.artifacts import stage_model_bundle
    from kumo_core.explain.local_sensitivity import load_scoring_module

    # Include relationships absent from this sample: omission must not turn a
    # parent record into an eligible leaf.
    edges = provenance['model_reconstruction']['graph_runtime_metadata'][
        'edge_types']
    authoritative = {(edge['src'], edge['rel'], edge['dst']) for edge in edges}
    declared = set(plan.foreign_keys)
    declared.update(
        (dst, 'rev_' + fk, src) for src, fk, dst in plan.foreign_keys)
    if declared != authoritative:
        raise ValueError(
            'Removal plan must declare the complete retained relation schema')
    training = trainer._fitted('perturb')
    with tempfile.TemporaryDirectory(prefix='record-removal-model-') as temp:
        model_dir = stage_model_bundle(training.artifact_paths,
                                       Path(temp) / 'model')
        module = load_scoring_module(str(model_dir), provenance)
        yield from record_removal_responses(
            module, sample, prediction=record['prediction'],
            prediction_id=record['prediction_id'], plan=plan)
