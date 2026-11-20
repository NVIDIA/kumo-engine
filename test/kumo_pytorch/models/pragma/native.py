# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Kumo input consumption around the reviewed PRAGMA token model."""

import hashlib
import inspect
from pathlib import Path

import torch

from . import _kumo
from .batch import PackedBatch
from .pragmatiq import ModelConfig, PragmaModel, PragmaOutput

REFERENCE_SHA256 = "1300110a6c704484c3b3108ac1dcc4ca68f1f390f0d4edfc89350e41bf1cae10"
UNAVAILABLE_TIMESTAMP = torch.iinfo(torch.int64).min


def _check_reference():
    # The reviewed core never reads raw event_ts/event_source. Preserve that
    # guard while its native interface uses Kumo's elapsed/calendar features.
    source = Path(inspect.getfile(PragmaModel))
    if hashlib.sha256(source.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("PRAGMA reference changed; review raw timestamp/source requirements before native execution")


class NativePragmaModel(PragmaModel):
    """Consume one native Kumo batch using fitted field-token vocabulary IDs.

    The inherited packed token interface and parameter names remain unchanged.
    Packing, reserved-ID remapping and timing features are model operations in
    ``forward_kumo``. Kumo owns fitting, masking, sampling and event ordering.
    """

    def __init__(self, config: ModelConfig, vocabulary, *, profile_input=None,
                 history_input="history", elapsed_from="cutoff", calendar=("hour", "weekday", "day")):
        if type(self) is not NativePragmaModel:
            raise TypeError("unavailable-metadata mode supports exactly the reviewed NativePragmaModel")
        _check_reference()
        vocabulary = tuple(vocabulary)
        if vocabulary[:5] != _kumo.SPECIALS or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("expected a unique fitted Kumo vocabulary with PAD,UNK,MASK,USR,EVT first")
        if config.vocab_size != len(vocabulary):
            raise ValueError("model vocabulary size must match the fitted Kumo vocabulary")
        if elapsed_from not in ("cutoff", "latest_event"):
            raise ValueError("elapsed_from must be cutoff or latest_event")
        if tuple(calendar) != ("hour", "weekday", "day"):
            raise ValueError("PRAGMA requires calendar order hour,weekday,day")
        super().__init__(config)
        self.kumo_vocabulary = vocabulary
        self.vocabulary = (vocabulary[0], vocabulary[2], vocabulary[1], *vocabulary[3:])
        self.profile_input, self.history_input = profile_input, history_input
        self.elapsed_from = elapsed_from

    def validate_kumo_batch(self, batch, *, target_name=None, cpu=True) -> None:
        """Check inputs and targets; ``cpu=False`` also accepts device-resident batches."""
        _check_reference()
        _kumo.inspect(batch, self, check_values=True, cpu=cpu)
        _kumo.inspect_targets(batch, self, target_name, cpu=cpu)

    def select_kumo_targets(self, batch, *, target_name="masked_values"):
        """Return selected token indices and remapped labels after CPU validation.

        Outputs stay on the batch device, independently of the model device.
        Request padding and the -100 ignore label are excluded. This selects
        already-created Kumo targets and never draws new masking decisions.
        """
        return _kumo.selected_targets(batch, self.history_input, target_name)

    def _profile_fields(self, profile, size, device):
        if profile is None:
            return {name: torch.empty(0, dtype=torch.long, device=device)
                    for name in ("prof_key_ids", "prof_value_ids", "prof_positions",
                                 "item_of_prof_token", "user_of_prof_item")} | {
                "prof_time_log": torch.empty(0, dtype=torch.float32, device=device),
                "cu_seqlens_profile_item": torch.zeros(1, dtype=torch.int32, device=device),
                "cu_seqlens_profile": torch.zeros(size + 1, dtype=torch.int32, device=device),
            }
        valid, positions = profile["valid"], profile["position_ids"]
        starts = valid & (positions == 0)
        flattened_starts = starts[valid]
        start_indices = flattened_starts.nonzero(as_tuple=False).flatten()
        boundaries = torch.cat((start_indices, start_indices.new_tensor([flattened_starts.numel()])))
        lengths = boundaries[1:] - boundaries[:-1]
        counts = starts.sum(1)
        return {
            "prof_key_ids": _kumo.remap(profile["field_ids"][valid]),
            "prof_value_ids": _kumo.remap(profile["value_ids"][valid]),
            "prof_positions": positions[valid],
            "prof_time_log": 8 * torch.log1p(profile["elapsed_seconds"][starts].float() / 8),
            "cu_seqlens_profile_item": _kumo.offsets(lengths),
            "cu_seqlens_profile": _kumo.offsets(counts),
            "item_of_prof_token": flattened_starts.long().cumsum(0) - 1,
            "user_of_prof_item": torch.repeat_interleave(torch.arange(size, device=device), counts,
                                                         output_size=start_indices.numel()),
        }

    def forward_kumo(self, batch, *, validate_inputs=True) -> PragmaOutput:
        """Build internal packed fields and encode the unchanged native batch.

        CPU ``validate_kumo_batch`` should precede device transfer. Set
        ``validate_inputs=False`` afterward to omit repeated value checks.
        The inherited segmented attention still reads offsets on the host.
        """
        if validate_inputs:
            _check_reference()
        history, profile, size = _kumo.inspect(batch, self, check_values=validate_inputs,
                                               device=self.embedding_weight.device)
        keys, sequence = history["field_ids"], history["sequence"]
        device = keys.device
        boundaries = history["field_offsets"]
        event_count = boundaries.numel() - 1
        counts = sequence["valid"].sum(1)
        entities = batch.metadata["entity_ids"]
        entities = entities.tolist() if isinstance(entities, torch.Tensor) else entities
        packed = PackedBatch(
            key_ids=_kumo.remap(keys), value_ids=_kumo.remap(history["value_ids"]),
            positions=history["position_ids"], cu_seqlens_event=boundaries.to(torch.int32),
            event_of_token=_kumo.token_events(history),
            event_ts=torch.full((event_count,), UNAVAILABLE_TIMESTAMP, dtype=torch.long, device=device),
            event_time_log=8 * torch.log1p(sequence["elapsed_seconds"][sequence["valid"]].float() / 8),
            event_hour=history["calendar"][:, 0].long(), event_dow=history["calendar"][:, 1].long(),
            event_dom=history["calendar"][:, 2].long(),
            event_source=torch.full((event_count,), -1, dtype=torch.long, device=device),
            cu_seqlens_history=_kumo.offsets(counts), user_of_event=history["owner"],
            user_ids=[str(value) for value in entities], n_events_per_user=counts,
            **self._profile_fields(profile, size, device),
        )
        return PragmaModel.forward(self, packed)
