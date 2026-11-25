# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-owned components exercised through the public extension points."""

import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from torch import nn

from tfm.api.graph import GraphArchitecture, GraphBuildContext
from tfm.compositions.profile_event_decoder import ProfileEventDecoder
from tfm.config import RecipeConfig
from tfm.contracts.protocols import AttentionSpec
from tfm.data.adapters.registry import Adapter
from tfm.data.adapters.result import AdapterResult
from tfm.data.synthetic.generator import generate_synthetic_events
from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
from tfm.layers.gnn import GNNEncoderModel
from tfm.layers.profile_event_decoder import decode_history, encode_events
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE


class UserTokenizer(KeyValueTimeTokenizer):
    """Inherits persistence, while supplying a custom named time array."""

    def collate(self, sequences, *, max_length):
        batch = super().collate(sequences, max_length=max_length)
        batch.events["custom_clock"] = (batch.events["hour"] + 3) * batch.event_mask
        return batch


class UserGraphWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.inner = encoder
        self.spec = encoder.spec

    def forward(self, batch):
        return self.inner(batch)


class MeanEvents(nn.Module):
    def forward(self, tokens, token_mask, event_mask):
        weights = token_mask.unsqueeze(-1).to(tokens.dtype)
        summaries = (tokens * weights).sum(-2) / weights.sum(-2).clamp_min(1)
        return summaries * event_mask.unsqueeze(-1), tokens


class MeanProfile(nn.Module):
    def forward(self, tokens, mask, time_log):
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)


class UserCalendar(nn.Module):
    def forward(self, batch):
        return batch.event_array("custom_clock").float().unsqueeze(-1) / 24


class CumulativeHistory(nn.Module):
    def forward(self, slots, mask, times):
        weights = mask.unsqueeze(-1).to(slots.dtype)
        return (slots * weights).cumsum(1) / weights.cumsum(1).clamp_min(1)


class UserHistory(nn.Module):
    def __init__(self, spec, *, position):
        super().__init__()
        if not isinstance(position, NoPE):
            raise ValueError("This custom cumulative history supports NoPE only")
        self.position_spec = position
        self.embed = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.event_encoder = MeanEvents()
        self.profile_encoder = MeanProfile()
        self.calendar = UserCalendar()
        self.decoder = CumulativeHistory()

    def embed_tokens(self, keys, values, positions, /):
        return self.embed(keys) + self.embed(values)

    def forward(self, history, *, context=None):
        assert history.elapsed_log is not None
        summaries, _ = encode_events(self, history)
        return decode_history(
            self, history, summaries, history.event_mask, history.elapsed_log, context=context
        )


@dataclass(frozen=True)
class UserHistorySpec:
    hidden_size: int
    vocab_size: int

    def build(self, attention: AttentionSpec, *, position: NoPE | TimeRoPE) -> nn.Module:
        return UserHistory(self, position=position)


class UserPool(nn.Module):
    def forward(self, states, mask):
        weights = mask.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(1) / weights.sum(1).clamp_min(1)


@dataclass(frozen=True)
class UserReadout:
    name: str = "user_mean"

    def build(self) -> nn.Module:
        return UserPool()


def history_factory(
    spec: ProfileEventDecoder, attention: AttentionSpec, *, position: NoPE | TimeRoPE
) -> nn.Module:
    return UserHistory(spec, position=position)


def head_factory(width: int) -> nn.Module:
    return nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))


def graph_factory(config: RecipeConfig, context: GraphBuildContext) -> nn.Module:
    from tfm.heads.binary import Binary

    return GNNEncoderModel(
        context.graph, objective=Binary(), head=head_factory(context.graph.hidden_size)
    )


ARCHITECTURE = GraphArchitecture(graph_factory, "graph")


def adapter(config, *, seed, flat_fields, fixture_dir):
    events = generate_synthetic_events(
        seed=seed,
        n_entities=config.n_entities,
        events_per_entity=config.events_per_entity,
        step_hours=config.event_step_hours,
    )
    return AdapterResult(events=tuple(events), metadata={"adapter": "user"})


def runner(command, *, config, config_path, run_dir, device, reuse_checkpoint=False):
    assert isinstance(config, RecipeConfig)
    result = run_dir or Path("outputs/user")
    result.mkdir(parents=True, exist_ok=True)
    (result / "selected.txt").write_text(config.recipe_family)
    return result


ADAPTER = Adapter(adapter)


def gross_amount_adapter(config, *, seed, flat_fields, fixture_dir):
    loaded = adapter(config, seed=seed, flat_fields=flat_fields, fixture_dir=fixture_dir)
    return AdapterResult(
        events=tuple(
            replace(
                event,
                numeric={
                    **{k: v for k, v in event.numeric.items() if k != "amount"},
                    "gross_amount": event.numeric["amount"],
                },
            )
            for event in loaded.events
        )
    )


def late_history_adapter(config, *, seed, flat_fields, fixture_dir):
    loaded = adapter(config, seed=seed, flat_fields=flat_fields, fixture_dir=fixture_dir)
    # Every owner starts after the earliest SSL codec cutoff, while forecast
    # windows and the downstream cutoff retain observed events.
    cutoff = datetime(2024, 2, 5, tzinfo=UTC)
    return AdapterResult(
        events=tuple(event for event in loaded.events if event.event_timestamp > cutoff)
    )


GROSS_AMOUNT_ADAPTER = Adapter(gross_amount_adapter)
LATE_HISTORY_ADAPTER = Adapter(late_history_adapter)


def mixed_amount_adapter(config, *, seed, flat_fields, fixture_dir):
    loaded = adapter(config, seed=seed, flat_fields=flat_fields, fixture_dir=fixture_dir)
    events = []
    for i, event in enumerate(loaded.events):
        numeric = {key: value for key, value in event.numeric.items() if key != "amount"}
        if i % 5 == 0:
            numeric[config.amount_field] = event.numeric["amount"]
        elif i % 5 in (1, 2, 3):
            numeric[config.amount_field] = (math.nan, math.inf, -math.inf)[i % 5 - 1]
        events.append(replace(event, numeric=numeric))
    return AdapterResult(events=tuple(events))


MIXED_AMOUNT_ADAPTER = Adapter(mixed_amount_adapter)
