# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a 2.0.0a1 recipe config into the current keys (`tfm migrate-config`).

2.0.0a1 is the SDK the GTC Berlin lab ships. Every key it had either moved into
a sub-section here, with the same meaning and default, or was removed. A
removed key is dropped only when its value could not have changed the run;
otherwise migration refuses, because the current kit cannot reproduce it.

`docs/migration/2.0.0a1-to-2.0.0a2.md` lists the same moves for readers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from tfm.config import RecipeConfig
from tfm.contracts.errors import SchemaError

KeyPath = tuple[str, ...]

MOVES: dict[KeyPath, KeyPath] = {
    ("data", "pretrain_window_events"): ("data", "pretrain_windows", "events"),
    ("data", "pretrain_max_tokens"): ("data", "pretrain_windows", "max_tokens"),
    ("training", "optimizer", "muon_learning_rate"): (
        "training",
        "optimizer",
        "muon",
        "learning_rate",
    ),
    ("training", "optimizer", "muon_momentum"): ("training", "optimizer", "muon", "momentum"),
    ("training", "optimizer", "muon_adjust_lr"): ("training", "optimizer", "muon", "adjust_lr"),
    ("evaluation", "history_window_probe"): ("evaluation", "history_window", "enabled"),
    ("evaluation", "history_window_max_rows"): ("evaluation", "history_window", "max_rows"),
    ("evaluation", "history_window_train_negatives_per_positive"): (
        "evaluation",
        "history_window",
        "train_negatives_per_positive",
    ),
    ("fraud_task", "train_head"): ("fraud_task", "supervised_head", "enabled"),
    ("fraud_task", "head_loss_weight"): ("fraud_task", "supervised_head", "loss_weight"),
    ("fraud_task", "head_probe_only"): ("fraud_task", "supervised_head", "probe_only"),
}
"""2.0.0a1 key -> its key now. Defaults are unchanged, so an absent key stays absent."""

_JOINT_FUSION_DEFAULTS = {"cross_layers": 2, "plr_frequencies": 8}
_DECODER_MAX_SEQ_LEN_DEFAULT = 256


@dataclass
class Migration:
    """A migrated config and what was changed to get there."""

    config: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def validated(self) -> RecipeConfig:
        """The migrated config, loaded; raises if it is not a valid config now."""
        return RecipeConfig.model_validate(self.config)


def _dotted(path: KeyPath) -> str:
    return ".".join(path)


def _get(raw: dict[str, Any], path: KeyPath) -> tuple[bool, Any]:
    node: Any = raw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _pop(raw: dict[str, Any], path: KeyPath) -> Any:
    node = raw
    for key in path[:-1]:
        node = node[key]
    return node.pop(path[-1])


def _set(raw: dict[str, Any], path: KeyPath, value: Any) -> None:
    node = raw
    for depth, key in enumerate(path[:-1], start=1):
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise SchemaError(f"{_dotted(path[:depth])} must be a mapping.")
        node = child
    node[path[-1]] = value


def migrate_config(raw: dict[str, Any]) -> Migration:
    """Rewrite a 2.0.0a1 config's keys; `raw` is not modified.

    Raises `SchemaError` for a 2.0.0a1 setting the current kit cannot run, or
    for a config that already sets both a 2.0.0a1 key and its replacement.
    """
    config = copy.deepcopy(raw)
    notes: list[str] = []

    present, stride = _get(config, ("data", "pretrain_window_stride"))
    if present:
        _, events = _get(config, ("data", "pretrain_window_events"))
        if stride is not None and stride != events:
            raise SchemaError(
                f"data.pretrain_window_stride {stride} overlaps windows of {events} events; "
                "pretraining windows no longer overlap, so this run cannot be migrated."
            )
        _pop(config, ("data", "pretrain_window_stride"))
        notes.append("dropped data.pretrain_window_stride: windows never overlap now")

    has_events, events = _get(config, ("data", "pretrain_window_events"))
    has_row, row = _get(config, ("data", "pretrain_max_tokens"))
    if has_events and events is None:
        _pop(config, ("data", "pretrain_window_events"))
        has_events = False
    if has_row and (row is None or not has_events):
        _pop(config, ("data", "pretrain_max_tokens"))
        if row is not None:
            notes.append(
                "dropped data.pretrain_max_tokens: it applied only with "
                "data.pretrain_window_events, which is unset"
            )

    present, packed = _get(config, ("training", "packing_seq_length"))
    if present:
        _, context = _get(config, ("model", "composition", "granularity", "decoder", "max_seq_len"))
        context = _DECODER_MAX_SEQ_LEN_DEFAULT if context is None else context
        if packed is not None and packed != context:
            raise SchemaError(
                f"training.packing_seq_length {packed} differs from decoder.max_seq_len "
                f"{context}; packed rows are now the decoder's context, so set "
                "decoder.max_seq_len instead."
            )
        _pop(config, ("training", "packing_seq_length"))
        notes.append("dropped training.packing_seq_length: packed rows are decoder.max_seq_len")

    for key, default in _JOINT_FUSION_DEFAULTS.items():
        path = ("fraud_task", "finetune", key)
        present, value = _get(config, path)
        if not present:
            continue
        if value != default:
            raise SchemaError(
                f"{_dotted(path)} {value} is not {default}; joint fusion now always uses "
                f"{key}={default}, so this run cannot be migrated."
            )
        _pop(config, path)
        notes.append(f"dropped {_dotted(path)}: {default} is the only setting now")

    present, backend = _get(config, ("training", "backend"))
    if present and isinstance(backend, str) and backend.strip().lower() == "nemo":
        _set(config, ("training", "backend"), "fsdp2")
        notes.append("training.backend nemo -> fsdp2 (nemo was fsdp2 with a NeMo model builder)")

    for old, new in MOVES.items():
        present, value = _get(config, old)
        if not present:
            continue
        if _get(config, new)[0]:
            raise SchemaError(f"Both {_dotted(old)} and {_dotted(new)} are set; keep one.")
        _pop(config, old)
        _set(config, new, value)
        notes.append(f"{_dotted(old)} -> {_dotted(new)}")
    return Migration(config=config, notes=notes)


__all__ = ["MOVES", "Migration", "migrate_config"]
