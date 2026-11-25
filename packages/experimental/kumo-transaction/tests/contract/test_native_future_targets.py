# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-cached future labels match the separate reference oracle, not capped inputs."""

from dataclasses import replace
from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest
import torch

from tfm.config import load_recipe_config
from tfm.contracts.graph import GraphAnchor, utc_us
from tfm.data.adapters.synthetic_relational import declaration, source_files
from tfm.data.graph import FutureTargetTransform, GraphSampler, future_subgraph_targets
from tfm.data.native_anchors import task_anchors
from tfm.data.native_source import NativeSourceFiles
from tfm.data.native_targets import NativeFutureTargets
from tfm.data.relational import RelationalSource, key_token


def prepared(tmp_path, *, fanout=1):
    config = load_recipe_config("recipes/relational_inputs/config.yaml")
    task = config.data.relational
    assert task is not None
    source = source_files(config.data, seed=config.seed, fixture_dir=tmp_path / "source")
    normalized = NativeSourceFiles(source, tmp_path / "normalized", block_size=2)
    path = task_anchors(normalized, task, tmp_path / "anchors.parquet")
    normalized.connection.execute(
        "CREATE TABLE anchors (sample_id TEXT PRIMARY KEY,root INTEGER,cutoff INTEGER,split TEXT)"
    )
    anchors = []
    for row in pq.read_table(path).to_pylist():
        code = normalized.code(task.root, (row["id"],))
        normalized.connection.execute(
            "INSERT INTO anchors VALUES (?,?,?,?)",
            (row["sample_id"], code, utc_us(row["cutoff"]) // 1_000_000, row["split"]),
        )
        anchors.append((row["split"], GraphAnchor(row["sample_id"], row["id"], row["cutoff"])))
    normalized.connection.commit()
    cache = NativeFutureTargets(
        normalized,
        root=task.root,
        depth=2,
        fanout=fanout,
        horizon_us=2 * 86400_000_000,
        partition_end_us=utc_us(datetime(2024, 1, 21, tzinfo=UTC)),
        observed_through_us=utc_us(datetime(2024, 1, 31, tzinfo=UTC)),
    )
    cache.prepare()
    return config, normalized, cache, anchors


def test_native_future_targets_and_streamed_transform_match_reference(tmp_path, monkeypatch):
    config, normalized, cache, anchors = prepared(tmp_path)
    try:
        assert config.data.relational is not None
        source = RelationalSource(declaration(config.data), config.data.relational)
        sampler = GraphSampler(source.graph(), depth=6, fanout=1)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Future labels must not sample historical model inputs at a future cutoff"
            )

        monkeypatch.setattr(GraphSampler, "sample", forbidden)
        train_reference = None
        for split in ("train", "validation"):
            selected = tuple(anchor for name, anchor in anchors if name == split)
            expected = future_subgraph_targets(
                sampler,
                tuple(
                    replace(anchor, entity_id=key_token((anchor.entity_id,))) for anchor in selected
                ),
                config.data.relational.root,
                horizon_us=cache.horizon_us,
                depth=2,
                partition_end_us=cache.partition_us,
            )
            actual = cache.raw(selected)
            assert actual.columns == expected.columns
            torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
            torch.testing.assert_close(actual.valid, expected.valid, rtol=0, atol=0)
            torch.testing.assert_close(actual.available_us, expected.available_us, rtol=0, atol=0)
            if split == "train":
                train_reference = expected
            else:
                assert (
                    max(
                        actual.values[:, i].max()
                        for i, c in enumerate(actual.columns)
                        if c.endswith("/count")
                    )
                    > 1
                )  # terminal rows are not capped to fanout=1
        assert train_reference is not None
        reference = FutureTargetTransform.fit(train_reference)
        fitted = cache.fit()
        assert fitted.selected == reference.selected
        torch.testing.assert_close(fitted.mean, reference.mean, rtol=0, atol=0)
        torch.testing.assert_close(fitted.scale, reference.scale, rtol=0, atol=0)
    finally:
        cache.close()
        normalized.close()


def test_future_cache_owns_tensors_and_binds_root_and_cutoff(tmp_path):
    _, normalized, cache, anchors = prepared(tmp_path)
    try:
        selected = tuple(anchor for split, anchor in anchors if split == "train")
        first = cache.raw(selected)
        expected = first.values.clone()
        first.values.add_(1)
        with pytest.raises(ValueError, match="mutat|changed"):
            first.assert_validated()
        torch.testing.assert_close(cache.raw(selected).values, expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="exact requested anchor"):
            cache.raw((replace(selected[0], entity_id="different"),))
        with pytest.raises(ValueError, match="exact requested anchor"):
            cache.raw((replace(selected[0], cutoff=datetime(2024, 1, 1, tzinfo=UTC)),))
    finally:
        cache.close()
        normalized.close()


def test_future_cache_reader_count_stays_bounded_across_epoch_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    _, normalized, cache, anchors = prepared(tmp_path)
    try:
        selected = tuple(anchor for split, anchor in anchors if split == "train")
        expected = cache.raw(selected)
        for _ in range(8):
            with ThreadPoolExecutor(max_workers=1) as worker:
                actual = worker.submit(cache.raw, selected).result()
            torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
            assert actual.values.data_ptr() != expected.values.data_ptr()
        assert len(cache._readers) == 1
        cache.close()
        with pytest.raises(ValueError, match="closed"):
            cache.raw(selected)
    finally:
        cache.close()
        normalized.close()


def test_future_reader_escapes_source_directory_uri(tmp_path):
    _, normalized, cache, anchors = prepared(tmp_path / "source ? with#markers")
    try:
        values = cache.raw(tuple(anchor for split, anchor in anchors if split == "train"))
        assert values.values.shape[0] > 0
        assert values.valid.any()
    finally:
        cache.close()
        normalized.close()
