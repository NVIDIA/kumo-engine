# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tfm.config import RecipeConfig
from tfm.contracts.graph import SequenceSupervisedBatch
from tfm.data import graph_recipe
from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture, load_pragmatiq_dataset
from tfm.data.collate import sequence_label
from tfm.data.sequence_builders import build_entity_sequences
from tfm.runtime import common


@pytest.mark.parametrize("field", ["hidden_size", "num_heads"])
@pytest.mark.parametrize("value", [0, -1])
def test_graph_config_rejects_nonpositive_decoder_dimensions(field, value):
    with pytest.raises(ValueError, match="hidden_size and num_heads must be positive"):
        RecipeConfig.model_validate(
            {
                "recipe_family": "graph",
                "model": {"composition": {"granularity": {"decoder": {field: value}}}},
            }
        )


def book_config(tmp_path):
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=12, events_per_user=20)
    config = yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    config["data"].update(
        adapter="pragmatiq",
        path=str(root),
        n_entities=12,
        probe_cutoff="2024-01-19T12:00:00Z",
        observed_through="2024-01-20T12:00:00Z",
    )
    return root, RecipeConfig.model_validate(config)


@pytest.mark.parametrize("kind", ["missing", "empty", "unmatched", "one_owner_missing"])
def test_graph_book_requires_a_real_label_source_and_history_coverage(tmp_path, kind):
    root, config = book_config(tmp_path)
    path = root / "labels/fraud.parquet"
    table = pq.read_table(path)
    if kind == "missing":
        path.unlink()
    elif kind == "empty":
        pq.write_table(table.slice(0, 0), path)
    elif kind == "unmatched":
        table = table.set_column(
            table.schema.get_field_index("ts"),
            "ts",
            pa.array([datetime(2020, 1, 1)] * len(table), type=pa.timestamp("us")),
        )
        pq.write_table(table, path)
    else:
        pq.write_table(table.filter(pa.compute.not_equal(table["user_id"], "u0000")), path)
    with pytest.raises(ValueError, match="fraud.*source|fraud label coverage"):
        graph_recipe.prepare_graph_data(config)


def test_valid_and_all_negative_books_keep_existing_sequence_labels(tmp_path):
    root, config = book_config(tmp_path)
    for all_negative in (False, True):
        if all_negative:
            path = root / "labels/fraud.parquet"
            table = pq.read_table(path)
            table = table.set_column(
                table.schema.get_field_index("label"),
                "label",
                pa.array([0] * len(table), type=pa.int8()),
            )
            pq.write_table(table, path)
        prepared = graph_recipe.prepare_graph_data(config)
        book = load_pragmatiq_dataset(
            root, max_entities=12, stratify_by_fraud=False, sample_seed=config.seed
        )
        sequences = build_entity_sequences(
            book.events,
            datetime(2024, 1, 19, 12, tzinfo=UTC),
            max_events=config.data.sequence_length,
        )
        expected = {s.entity_id: sequence_label(s) for s in sequences}
        for batches in prepared.probes.values():
            for batch in batches:
                assert batch.labels.tolist() == [expected[a.entity_id] for a in batch.graph.anchors]
        if all_negative:
            assert set(expected.values()) == {0.0}


def test_unavailable_labels_do_not_silently_become_negative(tmp_path, monkeypatch):
    root, config = book_config(tmp_path)
    book = load_pragmatiq_dataset(root)
    events = tuple(
        replace(
            e,
            labels={
                k: replace(v, available_at=datetime(2025, 1, 1, tzinfo=UTC))
                for k, v in e.labels.items()
            },
        )
        for e in book.events
    )
    monkeypatch.setattr(graph_recipe, "_events", lambda config: (list(events), book.transfers))
    with pytest.raises(ValueError, match="available fraud label coverage"):
        graph_recipe.prepare_graph_data(config)


@pytest.mark.parametrize("case", ["no_checkout", "no_git", "git_missing"])
def test_source_provenance_can_be_unknown(tmp_path, monkeypatch, case):
    monkeypatch.setattr(common, "repo_root", lambda: tmp_path)
    if case != "no_checkout":
        (tmp_path / "src/tfm").mkdir(parents=True)
        (tmp_path / "pyproject.toml").write_text("")
    if case == "git_missing":

        def absent(*args, **kwargs):
            raise FileNotFoundError("git unavailable")

        monkeypatch.setattr(common.subprocess, "run", absent)
    assert common.source_provenance() == {"source_revision": None, "source_dirty": None}


def test_sequence_control_rows_clocks_and_batches_match_hybrid_in_both_modes():
    for variant in ("config.smoke.yaml", "config.pretrain_smoke.yaml"):
        config = RecipeConfig.model_validate(
            yaml.safe_load((Path("recipes/gnn_sequence") / variant).read_text())
        )
        data = graph_recipe.prepare_graph_data(config)
        for hybrid, control in zip(data.hybrid_train, data.sequence_supervised_train, strict=True):
            assert isinstance(control, SequenceSupervisedBatch)
            assert control.history is hybrid.history
            assert control.labels is hybrid.labels
            assert control.anchors == hybrid.history_anchors
            assert control.label_available_us is hybrid.label_available_us
            assert control.label_cutoff_us == hybrid.label_cutoff_us
        for hybrid, control in zip(
            data.probes["validation"], data.sequence_supervised_validation, strict=True
        ):
            assert control.history is hybrid.history and control.labels is hybrid.labels


def test_valid_book_app_only_retained_history_keeps_its_zero_label(tmp_path):
    _root, config = book_config(tmp_path)
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "probe_cutoff": "2024-01-18T12:00:00Z",
                    "sequence_length": 1,
                }
            )
        }
    )
    data = graph_recipe.prepare_graph_data(config)
    assert all(
        batch.labels.count_nonzero() == 0 for batches in data.probes.values() for batch in batches
    )


@pytest.mark.parametrize(
    "override", ["directory", "worktree", "common", "index", "objects", "config"]
)
def test_source_provenance_ignores_foreign_git_environment(tmp_path, monkeypatch, override):
    import os
    import subprocess

    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def git(root, *args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            env=clean_env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    source, foreign = tmp_path / "source", tmp_path / "foreign"
    for root in (source, foreign):
        (root / "src/tfm").mkdir(parents=True)
        (root / "pyproject.toml").write_text("")
        (root / "src/tfm/__init__.py").write_text("")
        (root / "marker").write_text(root.name)
        git(root, "init")
        git(root, "add", ".")
        git(
            root,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            root.name,
        )
    git(source, "config", "status.showUntrackedFiles", "no")
    expected_revision = git(source, "rev-parse", "HEAD")
    assert expected_revision != git(foreign, "rev-parse", "HEAD")
    (foreign / "untracked").write_text("foreign dirty state")
    monkeypatch.setattr(common, "repo_root", lambda: source)
    overrides = {
        "directory": {"GIT_DIR": str(foreign / ".git")},
        "worktree": {"GIT_DIR": str(foreign / ".git"), "GIT_WORK_TREE": str(foreign)},
        "common": {"GIT_COMMON_DIR": str(foreign / ".git")},
        "index": {"GIT_INDEX_FILE": str(foreign / ".git/index")},
        "objects": {"GIT_OBJECT_DIRECTORY": str(foreign / ".git/objects")},
        "config": {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.worktree",
            "GIT_CONFIG_VALUE_0": str(foreign),
        },
    }[override]
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    assert common.source_provenance() == {
        "source_revision": expected_revision,
        "source_dirty": False,
    }
    (source / "untracked").write_text("source dirty state")
    assert common.source_provenance() == {
        "source_revision": expected_revision,
        "source_dirty": True,
    }
    assert all(os.environ[key] == value for key, value in overrides.items())


@pytest.mark.parametrize("future", [False, True])
def test_observation_watermark_allows_quiet_tail(future):
    variant = "config.pretrain_smoke.yaml" if future else "config.smoke.yaml"
    payload = yaml.safe_load((Path("recipes/gnn_sequence") / variant).read_text())
    payload["data"]["observed_through"] = "2024-05-01T00:00:00Z"
    config = RecipeConfig.model_validate(payload)
    events, _ = graph_recipe._events(config)
    assert max(e.event_timestamp for e in events) < datetime(2024, 5, 1, tzinfo=UTC)
    prepared = graph_recipe.prepare_graph_data(config)
    assert prepared.split_manifest["source_observed_through"] == "2024-05-01T00:00:00+00:00"
    assert prepared.split_manifest["sampling"]["event_key_encoding"] == "json-array-v1"
    bad = config.model_copy(
        update={"data": config.data.model_copy(update={"probe_cutoff": "2024-05-02T00:00:00Z"})}
    )
    with pytest.raises(ValueError, match="Probe cutoff/source coverage"):
        graph_recipe.prepare_graph_data(bad)


def test_transaction_event_join_keys_do_not_collide():
    import json

    from tfm.contracts.schema import TransactionEvent
    from tfm.data.graph_transactions import transaction_graph

    at = datetime(2024, 1, 1, tzinfo=UTC)
    identities = [("a/b", "c"), ("a", "b/c"), ('a"', "b\\c"), ("α", "事件")]
    events = [
        TransactionEvent(owner, key, at, numeric={"amount": 1.0}) for owner, key in identities
    ]
    graph = transaction_graph(events, observed_through=at)
    keys = [row.row_id for row in graph.rows if row.table == "events"]
    assert len(set(keys)) == len(identities)
    assert {tuple(json.loads(key)) for key in keys} == set(identities)
    assert {link.target_id for link in graph.links} == set(keys)
    with pytest.raises(ValueError, match="duplicate graph primary key"):
        transaction_graph([*events, events[0]], observed_through=at)


def test_graph_recipe_rejects_empty_observations(monkeypatch):
    config = RecipeConfig.model_validate(
        yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    )
    monkeypatch.setattr(graph_recipe, "_events", lambda config: ([], ()))
    with pytest.raises(ValueError, match="Probe cutoff/source coverage"):
        graph_recipe.prepare_graph_data(config)
