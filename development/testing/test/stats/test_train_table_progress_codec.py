# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoRPC codec round-trip regression tests for ``TrainTableGenerationProgress``.

The original Python-3.14 / pydantic-2 bug surfaced not only when polling
progress from a job-dir JSON, but also whenever a pydantic-v2 model carrying a
bare ``pd.Timestamp`` field is (de)serialized over Temporal's
``json/pydanticclass`` codec (``temporalio_commons.dataclass_converter``). On
the wire pydantic serializes ``pd.Timestamp`` to an ISO *string*; on the way
back pydantic v2 strict-validates the string against the (bare) ``pd.Timestamp``
field and raises ``is_instance_of`` -- which is what crashes a worker with
"kumo-auto subprocess exited non-zero".

These tests drive the *exact* AutoRPC encode/decode path so a regression in the
``PandasTimestamp`` annotation on the nested timeframe dataclasses is caught.
They fail pre-fix (bare ``pd.Timestamp``) and pass post-fix.
"""
import pandas as pd

from io_utils.json import from_json, to_json
from kumo.stats.train_table_stats import (
    TemporalLinkTimeFrameStats,
    TemporalNodeTimeFrameStats,
    TrainTableGenerationProgress,
)
from temporalio_commons.dataclass_converter import (
    DataclassJsonPayloadConverter,
)


def _node_progress() -> TrainTableGenerationProgress:
    tf = TemporalNodeTimeFrameStats(
        min_date='2020-01-01',
        max_date='2020-02-01',
        anchor_date='2020-01-15',
        target_distribution={'a': 1},
        index=0,
    )
    return TrainTableGenerationProgress(
        num_expected_iterations=1,
        num_finished_iterations=1,
        iteration_data=[tf],
    )


def _link_progress() -> TrainTableGenerationProgress:
    tf = TemporalLinkTimeFrameStats(
        min_date='2020-01-01',
        max_date='2020-02-01',
        anchor_date='2020-01-15',
        num_train_rows=10,
        num_entities=5,
        num_targets=5,
        num_links=12,
        index=0,
    )
    return TrainTableGenerationProgress(
        num_expected_iterations=1,
        num_finished_iterations=1,
        iteration_data=[tf],
    )


def test_progress_autorpc_codec_roundtrip_node() -> None:
    conv = DataclassJsonPayloadConverter()
    payload = conv.to_payload(_node_progress())
    assert payload is not None
    assert payload.metadata['encoding'] == b'json/pydanticclass'

    back = conv.from_payload(payload, TrainTableGenerationProgress)
    assert isinstance(back, TrainTableGenerationProgress)
    assert back.iteration_data is not None
    tf = back.iteration_data[0]
    assert isinstance(tf, TemporalNodeTimeFrameStats)
    assert isinstance(tf.min_date, pd.Timestamp)
    assert tf.min_date == pd.Timestamp('2020-01-01')
    assert isinstance(tf.anchor_date, pd.Timestamp)


def test_progress_autorpc_codec_roundtrip_link() -> None:
    conv = DataclassJsonPayloadConverter()
    payload = conv.to_payload(_link_progress())
    assert payload is not None

    back = conv.from_payload(payload, TrainTableGenerationProgress)
    assert back.iteration_data is not None
    tf = back.iteration_data[0]
    assert isinstance(tf, TemporalLinkTimeFrameStats)
    assert isinstance(tf.min_date, pd.Timestamp)
    assert tf.num_links == 12


def test_progress_to_from_json_roundtrip() -> None:
    # The lower-level ``io_utils.json`` path that ``from_payload`` ultimately
    # calls. Serialized dates are ISO strings; deserialization must coerce them
    # back to ``pd.Timestamp`` rather than raising ``is_instance_of``.
    progress = _node_progress()
    encoded = to_json(progress)
    assert '2020-01-01' in encoded

    back = from_json(encoded, TrainTableGenerationProgress)
    assert back.iteration_data is not None
    assert isinstance(back.iteration_data[0].min_date, pd.Timestamp)
    assert back.iteration_data[0].max_date == pd.Timestamp('2020-02-01')
