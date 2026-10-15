# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for ``DiskgraphNestedTrainer``.

Covers the constructor contract and ``init()`` overrides. Heavier
integration coverage (real PL model build, real wheel iterator) lives
in the smoke test against a real task-dataset bundle.
"""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Mapping

import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod
import numpy as np
import pandas as pd
import pytest
import torch
from kumo_core.diskgraph.runtime_adapter import runtime_batch_to_hetero
from kumo_core.explain.subgraph_collector import SubgraphCollector
from kumo_core.train.diskgraph_nested_trainer import (
    DiskgraphNestedTrainer,
    _forecasting_out_timestamps,
    _rhs_canonical_node_ids,
    _rhs_primary_key_catalog,
    _rhs_primary_key_index_map,
    _seed_time_to_datetime64_ns,
)
from kumo_core.train.mapping import (
    NODE_INDEX_COLUMN,
    TTNG_RHS_PRIMARY_KEY_INDEX_NAME,
)
from kumo_core.training.config import FitMode
from kumo_core.types import TaskType
from kumoapi.model_plan import WeightMode
from kumoapi.train import TrainingStage


def _worker_shutdown_stub() -> dict[str, Any]:
    """The worker-pool barrier every ``DiskgraphLightningDataModule`` carries.

    ``predict`` and ``_predict_lp`` call both unconditionally, so a stub that
    omits them drifts from the surface it exists to imitate.
    """
    return {
        'defer_dataloader_worker_shutdown': lambda phase: nullcontext(),
        'shutdown_dataloader_workers': lambda phase: None,
    }


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Don't try to bring up the real kumo backend for these unit tests."""
    return False


class _FakeManifest:
    """Stand-in for ``DiskgraphTaskManifest``.

    Just enough surface for the subclass's ``init()`` to run end-to-end
    without touching disk. Mirrors the shape of the real loader.
    """

    metadata = {
        "train_table_name": "train_table",
        "entity_table_name": "my_customers",
        "seed_node_type": "my_customers",
        "task_type": "binary_classification",
        "train_table_columns": {
            "time_column": "TIMESTAMP",
        },
    }
    feature_schema = {
        "node_types": {
            "my_customers": {
                "feature_columns": [],
                "mapper_key": "my_customers",
                "time_column": "created_at",
            },
            "orders": {
                "feature_columns": [],
                "mapper_key": "orders",
                "time_column": None,
            },
        }
    }
    graph_runtime_metadata: dict[str, Any] = {
        "node_types": ["my_customers", "orders"],
        "edge_types": [{
            "src": "orders",
            "rel": "customer",
            "dst": "my_customers"
        }],
    }
    manifest_path = "/tmp/fake_bundle/manifest.json"
    runtime_manifest_path = "/tmp/fake_bundle/runtime_manifest.json"
    stage_seed_times: dict[str, tuple[int, int]] = {}
    manifest: dict[str, Any] = {}
    stats: dict[str, Any] = {}

    @property
    def train_table_columns(self) -> Mapping[str, Any]:
        return self.metadata["train_table_columns"]

    @property
    def train_table_name(self) -> str:
        return self.metadata["train_table_name"]

    @property
    def seed_node_type(self) -> str:
        return self.metadata["seed_node_type"]

    @property
    def target_node_type(self) -> str:
        # Mirrors ``DiskgraphTaskManifest.target_node_type``: the RHS entity
        # node type for LP, falling back to ``seed_node_type`` when the
        # bundle omits an explicit ``target_node_type``.
        name = self.metadata.get("target_node_type")
        return str(name) if name else self.seed_node_type

    @property
    def task_type(self) -> str:
        return self.metadata["task_type"]


class _StubModel:
    def __init__(self, *, fully_specified: bool = True,
                 channels: int = 64) -> None:
        self.is_fully_specified = fully_specified
        self.channels = channels
        self.metadata: Any = None
        self.validate_called = False

    def validate(self) -> None:
        self.validate_called = True


class _StubEncoder:
    def __init__(self, *, fully_specified: bool = True) -> None:
        self.is_fully_specified = fully_specified
        self.table_encoder_dict: dict[str, Any] = {}
        self.out_channels: int | None = None
        self.time_col_dict: dict[str, str] = {}
        self.exclude_cols_dict: dict[str, Any] = {}
        self.validate_called = False

    def validate(self) -> None:
        self.validate_called = True


class _StubHead:
    def __init__(self, *, fully_specified: bool = True,
                 out_channels: int | None = None) -> None:
        self.is_fully_specified = fully_specified
        self.out_channels = out_channels
        self.in_channels: int | None = None
        self.param_keys: list[str] = []
        self.validate_called = False

    def validate(self) -> None:
        self.validate_called = True


class _StubSequenceEncoder:
    def __init__(self, *, fully_specified: bool = False) -> None:
        self.is_fully_specified = fully_specified
        self.in_channels: int | None = None
        self.out_channels: int | None = None
        self.length: int | None = None


class _StubEncoders:
    """Stand-in for ``EncodersConfig`` — exposes ``.encoders`` dict
    keyed by table name, matching how the legacy code paths access it."""
    def __init__(self, encoders: dict[str, Any] | None = None) -> None:
        self.encoders = encoders or {}

    # Subscript access used by ``_build_mapper_bank_from_encoders``.
    def __contains__(self, key: str) -> bool:
        return key in self.encoders

    def __getitem__(self, key: str) -> Any:
        return self.encoders[key]


class _StubTrainer:
    """Minimal stand-in for the bits of ``Trainer`` state that
    ``DiskgraphNestedTrainer.init()`` reads.

    Each subcomponent (``model``, ``encoder``, ``head``) defaults to
    ``is_fully_specified=True`` so tests that don't care about the
    metadata-wiring step get a no-op. Tests that exercise
    ``_configure_model_triple_from_manifest`` flip the relevant flag
    to ``False`` and assert on the resulting attribute writes.
    """

    _UNSET = object()

    def __init__(
        self,
        *,
        bundle_dir: str = "/tmp/fake_bundle",
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        majority_sampling_ratio: float | None = None,
        weight_mode: WeightMode | None = None,
        fit_mode: FitMode = FitMode.fit,
        head_out_channels: int | None = None,
        model_fully_specified: bool = True,
        encoder_fully_specified: bool = True,
        head_fully_specified: bool = True,
        eval_loader_kwargs: Any = _UNSET,
        max_steps_per_epoch: int = 2000,
        min_steps_per_epoch: int = 1,
        max_val_steps: int | None = 2000,
        max_test_steps: int | None = 4000,
        sampler_replace: bool = False,
        sampler_disjoint: bool = True,
        sampler_adaptive_sampling: bool = False,
        random_state_seed: int | None = None,
        exclude_cols_dict: dict[str, list[str]] | None = None,
    ) -> None:
        self.diskgraph_task_dataset_bundle_dir = bundle_dir
        self._fully_initialized = False

        class _Task:
            def __init__(self, task_type: TaskType) -> None:
                self.task_type = task_type
                self.lag_timesteps = 3
                self.unix_step_size = 86400
                self.exclude_cols_dict = exclude_cols_dict or {}

        self.task = _Task(task_type)

        self.model = _StubModel(fully_specified=model_fully_specified)
        self.encoder = _StubEncoder(fully_specified=encoder_fully_specified)
        self.head = _StubHead(
            fully_specified=head_fully_specified,
            out_channels=head_out_channels,
        )
        self.past_encoder: Any = None
        self.year_over_year_encoder: Any = None
        self.encoders = _StubEncoders()

        class _LoaderKwargs:
            batch_size = 16
            num_workers = 0
            pin_memory = False
            # Mirrors the real ``LoaderKwargs``; the trainer forwards these
            # to the datamodule, so omitting them here makes the stub drift
            # from the surface it exists to imitate.
            persistent_workers = False
            prefetch_factor = None

        self.loader_kwargs = _LoaderKwargs()
        self.eval_loader_kwargs = ({}
                                   if eval_loader_kwargs is _StubTrainer._UNSET
                                   else eval_loader_kwargs)

        # ``sampler_kwargs`` is read by ``DiskgraphNestedTrainer.init``
        # only when a non-stub sampling translator is wired in; the
        # current path uses ``_DEFAULT_TTNG_SAMPLING_PLAN`` and ignores
        # this attribute. Kept on the stub so the trainer's surface
        # doesn't drift from the parent ``Trainer``.
        class _FakeNumNeighbors:
            default = None

            @staticmethod
            def get_values() -> Any:
                return [10, 10]

        class _SamplerKwargs:
            num_neighbors = _FakeNumNeighbors()
            temporal_strategy = 'last'
            replace = sampler_replace
            disjoint = sampler_disjoint
            adaptive_sampling = sampler_adaptive_sampling

        self.sampler_kwargs = _SamplerKwargs()
        self.majority_sampling_ratio = majority_sampling_ratio
        self.majority_sampling_weights = None
        self.weight_mode = weight_mode
        self.fit_mode = fit_mode
        self.num_targets = 1
        self.loss = None
        self.target_encoder = None
        self.max_steps_per_epoch = max_steps_per_epoch
        self.min_steps_per_epoch = min_steps_per_epoch
        self.max_val_steps = max_val_steps
        self.max_test_steps = max_test_steps
        self.random_state_seed = random_state_seed
        self.checkpoint_path: str | None = None
        self.diskgraph_runtime: Any = None

        self.pl_trainer_called = False
        self.pl_model_kwargs: dict[str, Any] | None = None
        self.datamodule: Any = None

    def create_pl_trainer(self) -> None:
        self.pl_trainer_called = True
        # ``init()`` appends the epoch-propagation callback onto
        # ``self.pl_trainer.callbacks`` immediately after this call.
        self.pl_trainer = SimpleNamespace(callbacks=[])

    def create_pl_model(self, **kwargs: Any) -> None:
        self.pl_model_kwargs = kwargs
        self.pl_model = torch.nn.Module()
        self.pl_model.use_precomputed_forecasting_sequences = False

    # Bind the real helper as a class attribute so ``self._configure_
    # model_triple_from_manifest(...)`` resolves through the descriptor
    # protocol when the bound test caller drives ``init()`` against a
    # stub instance.
    _configure_model_triple_from_manifest = (
        DiskgraphNestedTrainer._configure_model_triple_from_manifest)
    _build_ttng_sampling_plan = (
        DiskgraphNestedTrainer._build_ttng_sampling_plan)


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


def test_logs_are_nested_directly_below_automl_trial(tmp_path: Any) -> None:
    trainer = DiskgraphNestedTrainer.__new__(DiskgraphNestedTrainer)
    trainer.task = SimpleNamespace(name="task")
    trainer.root_dir = str(tmp_path / "task" / "automl")
    trainer.track_version = True
    trainer.version = 3
    trainer.fit_mode = FitMode.fit

    assert trainer.log_dir == str(tmp_path / "task" / "automl" / "version_3")


def test_constructor_requires_bundle_dir() -> None:
    """TT-NG: the subclass refuses to be constructed without a bundle."""
    # We can't easily construct the full Trainer surface here,
    # so just verify the kwarg pop + None check fires before super().
    with pytest.raises(ValueError, match="diskgraph_task_dataset_bundle_dir"):
        # Pass diskgraph_task_dataset_bundle_dir=None explicitly to trip
        # the guard. The parent constructor is never reached because the
        # check happens before super().__init__.
        DiskgraphNestedTrainer.__new__(DiskgraphNestedTrainer).__init__(
            diskgraph_task_dataset_bundle_dir=None)


def test_constructor_missing_bundle_dir_kwarg_raises() -> None:
    """TT-NG: omitting the kwarg also raises."""
    with pytest.raises(ValueError, match="diskgraph_task_dataset_bundle_dir"):
        DiskgraphNestedTrainer.__new__(DiskgraphNestedTrainer).__init__()


def test_set_explainers_keeps_rfm_format_for_ttng(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    collector = SubgraphCollector(
        task_type=TaskType.BINARY_CLASSIFICATION,
        root_dir=str(tmp_path),
        formats=['nested_json', 'output', 'rfm'],
        target_to_entity_paths=[],
    )

    def fake_parent_set_explainers(self: Any, *args: Any,
                                   **kwargs: Any) -> None:
        self.pl_model.explainers.append(collector)

    monkeypatch.setattr(dg_nested_mod.Trainer, "set_explainers",
                        fake_parent_set_explainers)

    trainer = DiskgraphNestedTrainer.__new__(DiskgraphNestedTrainer)
    trainer.pl_model = SimpleNamespace(explainers=[])
    DiskgraphNestedTrainer.set_explainers(
        trainer,
        TrainingStage.TEST,
        ['subgraph_collector'],
    )

    assert collector.formats == ['nested_json', 'output', 'rfm']


# ---------------------------------------------------------------------------
# init() behavior — driven against _StubTrainer to avoid the full
# Trainer.__init__ surface. We test init() by binding the unbound
# DiskgraphNestedTrainer.init to a stub instance.
# ---------------------------------------------------------------------------


def _run_init(
    stub: _StubTrainer,
    *,
    monkeypatch: pytest.MonkeyPatch,
    fake_manifest: _FakeManifest | None = None,
    task_manifest: _FakeManifest | None = None,
) -> _FakeManifest:
    """Drive the unbound ``init`` against a stub instance.

    Patches ``DiskgraphTaskManifest.load`` to return a fake manifest +
    ``_build_mapper_bank_from_encoders`` to return an empty bank +
    ``DiskgraphLightningDataModule`` to a sentinel. Returns the
    fake manifest so callers can assert on its values.
    """
    fake_manifest = fake_manifest or _FakeManifest()

    import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod

    monkeypatch.setattr(
        dg_nested_mod.DiskgraphTaskManifest,
        "load",
        staticmethod(lambda manifest_path: fake_manifest),
    )
    monkeypatch.setattr(
        dg_nested_mod,
        "_build_mapper_bank_from_encoders",
        lambda encoders, feature_schema: {},
    )

    class _FakeDLDM:
        def __init__(self, **kwargs: Any) -> None:
            stub.datamodule_kwargs = kwargs  # type: ignore[attr-defined]

        @staticmethod
        def shuffle_stream_callback() -> Any:
            return object()

        def rhs_predict_dataloader(self) -> list[Any]:
            rhs_pkeys = np.asarray(stub.rhs_pkeys)
            rhs_name = fake_manifest.target_node_type
            return [{
                rhs_name:
                SimpleNamespace(batch_size=len(rhs_pkeys), pkey=rhs_pkeys)
            }]

        @staticmethod
        def worker_pool_lifecycle_callback() -> Any:
            return object()

    monkeypatch.setattr(dg_nested_mod, "DiskgraphLightningDataModule",
                        _FakeDLDM)

    def _primary_key_batches(*_args, **_kwargs):
        rhs_node_ids = getattr(stub, 'rhs_node_ids', range(len(stub.rhs_pkeys)))
        return iter([(
            np.asarray(rhs_node_ids, dtype=np.uint64),
            np.asarray(stub.rhs_pkeys),
        )])

    monkeypatch.setattr(
        dg_nested_mod,
        'iter_primary_key_batches',
        _primary_key_batches,
    )

    # Call the unbound init() against our stub.
    DiskgraphNestedTrainer.init(
        stub,  # type: ignore[arg-type]
        task_manifest=task_manifest,  # type: ignore[arg-type]
    )
    return fake_manifest


def test_init_wires_majority_sampling_sample_mode(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifest = _FakeManifest()
    runtime_manifest_path = tmp_path / "runtime_manifest.json"
    runtime_manifest_path.write_text(
        json.dumps({
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 100,
                    "binary_label_index": {
                        "false_node_ids": "train_seeds/false_node_ids.bin",
                        "false_count": 90,
                        "true_node_ids": "train_seeds/true_node_ids.bin",
                        "true_count": 10,
                    },
                },
            },
        }))
    manifest.runtime_manifest_path = str(runtime_manifest_path)
    manifest.stats = {
        "target_distribution": {
            "percentage": {
                "False": 0.5,
                "True": 0.5,
            },
        },
    }
    stub = _StubTrainer(
        majority_sampling_ratio=1.0,
        weight_mode=WeightMode.SAMPLE,
        max_steps_per_epoch=4,
        min_steps_per_epoch=1,
    )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert stub.majority_sampling_weights is None
    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["weighted_seed_sampling"] == {
        "use_sample_weights": False,
        "false_weight": 1.0,
        "true_weight": 9.0,
    }


def test_init_preserves_sample_mode_for_refit(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path) -> None:
    manifest = _FakeManifest()
    runtime_manifest_path = tmp_path / "runtime_manifest.json"
    runtime_manifest_path.write_text(
        json.dumps({
            "stages": {
                "train": {
                    "binary_label_index": {
                        "false_count": 80,
                        "true_count": 20,
                    },
                },
                "val": {
                    "binary_label_index": {
                        "false_count": 9,
                        "true_count": 1,
                    },
                },
            },
        }))
    manifest.runtime_manifest_path = str(runtime_manifest_path)
    manifest.stats = {
        "target_distribution": {
            "percentage": {
                "False": 0.9,
                "True": 0.1,
            },
        },
    }
    stub = _StubTrainer(
        majority_sampling_ratio=1.0,
        weight_mode=WeightMode.SAMPLE,
        fit_mode=FitMode.refit_trainval,
    )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert stub.weight_mode == WeightMode.SAMPLE
    assert stub.majority_sampling_weights is None
    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["weighted_seed_sampling"] == {
        "use_sample_weights": False,
        "false_weight": 1.0,
        "true_weight": 4.0,
    }
    assert dm_kwargs["train_stages"] == ("train", "val")


def test_init_wires_weight_col_sample_mode(
        monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _FakeManifest()
    manifest.metadata = {
        **_FakeManifest.metadata,
        "train_table_columns": {
            "time_column": "TIMESTAMP",
            "weight_column": "WEIGHT",
        },
    }
    stub = _StubTrainer(weight_mode=WeightMode.SAMPLE, )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert stub.weight_mode == WeightMode.SAMPLE
    assert stub.majority_sampling_weights is None
    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["weighted_seed_sampling"] == {
        "use_sample_weights": True,
    }


def test_init_wires_weight_col_and_majority_sampling_sample_mode(
        monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifest = _FakeManifest()
    manifest.metadata = {
        **_FakeManifest.metadata,
        "train_table_columns": {
            "time_column": "TIMESTAMP",
            "weight_column": "WEIGHT",
        },
    }
    runtime_manifest_path = tmp_path / "runtime_manifest.json"
    runtime_manifest_path.write_text(
        json.dumps({
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 100,
                    "binary_label_index": {
                        "false_node_ids": "train_seeds/false.bin",
                        "false_count": 90,
                        "true_node_ids": "train_seeds/true.bin",
                        "true_count": 10,
                    },
                },
            },
        }))
    manifest.runtime_manifest_path = str(runtime_manifest_path)
    stub = _StubTrainer(
        majority_sampling_ratio=1.0,
        weight_mode=WeightMode.SAMPLE,
    )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert stub.majority_sampling_weights is None
    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["weighted_seed_sampling"] == {
        "use_sample_weights": True,
        "false_weight": 1.0,
        "true_weight": 9.0,
    }


def test_init_computes_majority_sampling_weights_from_manifest_stats(
        monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _FakeManifest()
    manifest.stats = {
        "target_distribution": {
            "percentage": {
                "False": 0.9,
                "True": 0.1,
            },
        },
    }
    stub = _StubTrainer(
        majority_sampling_ratio=1.0,
        weight_mode=WeightMode.WEIGHTED_LOSS,
    )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert stub.majority_sampling_weights == {False: 1, True: 9.0}


def test_init_skips_majority_sampling_weights_for_prediction(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        majority_sampling_ratio=1.0,
        weight_mode=WeightMode.WEIGHTED_LOSS,
    )
    stub._prediction_mode = True

    _run_init(stub, monkeypatch=monkeypatch)

    assert stub.majority_sampling_weights is None


def test_init_normalizes_multiclass_with_single_head_to_binary(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror Trainer.init's task_type normalization: MULTICLASS with
    out_channels=1 is treated as binary classification."""
    stub = _StubTrainer(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        head_out_channels=1,
    )
    _run_init(stub, monkeypatch=monkeypatch)
    assert stub.task.task_type == TaskType.BINARY_CLASSIFICATION


def test_init_happy_path_builds_datamodule_and_calls_pl_helpers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG happy path: init() loads manifest, builds DLDM, calls the
    parent's create_pl_trainer + create_pl_model with bundle-derived
    args (empty rhs_index_map, entity_table_name from metadata)."""
    stub = _StubTrainer(bundle_dir="/tmp/my_bundle")
    fake_manifest = _run_init(stub, monkeypatch=monkeypatch)

    # DLDM was constructed with the loaded manifest.
    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["manifest"] is fake_manifest
    assert dm_kwargs["mapper_bank"] == {}
    # ``DiskgraphNestedTrainer`` currently uses the hardcoded
    # ``_DEFAULT_TTNG_SAMPLING_PLAN`` (2-hop uniform fanout=10).
    # When the ``cfg.neighbor_sampling`` translation lands, update
    # this assertion to assert against the translated plan.
    import kumo_core.train.diskgraph_nested_trainer as diskgraph_nested_trainer
    assert (dm_kwargs["sampling_plan"] ==
            diskgraph_nested_trainer._DEFAULT_TTNG_SAMPLING_PLAN)
    assert dm_kwargs["train_batch_size"] == 16
    assert dm_kwargs["eval_batch_size"] == 16
    assert dm_kwargs["max_steps_per_epoch"] == 2000
    assert dm_kwargs["min_steps_per_epoch"] == 1
    assert dm_kwargs["max_val_steps"] == 2000
    assert dm_kwargs["max_test_steps"] == 4000
    assert dm_kwargs["train_stages"] == ("train", )
    assert dm_kwargs["task_type"] == "binary_classification"

    # PL trainer + model were built.
    assert stub.pl_trainer_called
    assert len(stub.pl_trainer.callbacks) == 2
    assert stub.pl_model_kwargs is not None
    assert stub.pl_model_kwargs["entity_table_names"] == ["my_customers"]
    assert stub.pl_model_kwargs["train_table_name"] == "train_table"

    # rhs_index_map is empty for v1 (LP support deferred).
    rhs_index_map = stub.pl_model_kwargs["rhs_index_map"]
    assert isinstance(rhs_index_map, pd.DataFrame)
    assert len(rhs_index_map) == 0


def test_init_uses_a_handed_over_manifest(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(bundle_dir="/tmp/my_bundle")
    handed = _FakeManifest()

    loaded = _run_init(stub, monkeypatch=monkeypatch, task_manifest=handed)

    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["manifest"] is handed is not loaded


def test_init_passes_eval_limits_and_runtime_options(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        max_val_steps=7,
        max_test_steps=9,
        sampler_replace=True,
        sampler_disjoint=False,
        sampler_adaptive_sampling=True,
    )

    _run_init(stub, monkeypatch=monkeypatch)

    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["max_val_steps"] == 7
    assert dm_kwargs["max_test_steps"] == 9
    assert dm_kwargs["runtime_options"] == {
        "replace": True,
        "disjoint": False,
        "adaptive": True,
    }


def test_init_applies_diskgraph_auto_config_to_ttng_runtime(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG must forward the legacy policy to the wheel, not bypass it."""
    import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod

    expected = {
        "mode": "standalone",
        "num_workers": 8,
        "rayon_threads": 4,
    }
    monkeypatch.setattr(
        dg_nested_mod,
        "_ttng_diskgraph_runtime_config",
        lambda *args: expected,
    )
    stub = _StubTrainer()

    _run_init(stub, monkeypatch=monkeypatch)

    assert stub.loader_kwargs.num_workers == 8
    assert stub.datamodule_kwargs[
        "num_workers"] == 8  # type: ignore[attr-defined]
    assert stub.datamodule_kwargs["runtime_options"] == {  # type: ignore[attr-defined]
        "replace": False,
        "disjoint": True,
        "adaptive": False,
        "num_threads": 4,
        "fused": False,
    }


def test_ttng_auto_config_uses_compiled_plan_and_actual_batch_size(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The TT-NG estimate is plan-derived, not legacy's fixed 1024 batch."""
    import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod

    index = tmp_path / "index"
    index.mkdir()
    observed: dict[str, Any] = {}

    def fake_auto_config(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"mode": "standalone", "num_workers": 8, "rayon_threads": 4}

    runtime = SimpleNamespace(auto_config=fake_auto_config)
    plan = {
        "layers": [
            {
                "fanouts": {
                    "first": {
                        "fanout": 2
                    }
                },
                "default_fanout": None
            },
            {
                "fanouts": {
                    "second": {
                        "fanout": 5
                    }
                },
                "default_fanout": None
            },
        ]
    }

    config = dg_nested_mod._ttng_diskgraph_runtime_config(
        runtime, {"graph_index_path": str(index)}, plan, batch_size=2000)

    assert config == {
        "mode": "standalone",
        "num_workers": 8,
        "rayon_threads": 4
    }
    assert observed["index_path"] == str(index)
    assert observed["estimated_edges"] == 5_000


@pytest.mark.parametrize(("fit_mode", "expected"), [
    (FitMode.fit, ("train", )),
    (FitMode.refit_trainval, ("train", "val")),
    (FitMode.refit_full, ("train", "val", "test")),
])
def test_init_selects_train_stages_for_fit_mode(
    monkeypatch: pytest.MonkeyPatch,
    fit_mode: FitMode,
    expected: tuple[str, ...],
) -> None:
    stub = _StubTrainer(fit_mode=fit_mode)

    _run_init(stub, monkeypatch=monkeypatch)

    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["train_stages"] == expected


def test_init_passes_bundle_manifest_path_through(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: the manifest path passed to .load() is exactly
    ``{bundle_dir}/manifest.json``."""
    seen: dict[str, str] = {}

    import kumo_core.diskgraph.lightning_data as dg_data_mod
    import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod

    def _capture_load(manifest_path: str) -> _FakeManifest:
        seen["manifest_path"] = manifest_path
        return _FakeManifest()

    monkeypatch.setattr(
        dg_data_mod.DiskgraphTaskManifest,
        "load",
        staticmethod(_capture_load),
    )
    monkeypatch.setattr(
        dg_nested_mod,
        "_build_mapper_bank_from_encoders",
        lambda encoders, feature_schema: {},
    )

    class _NoopDLDM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        @staticmethod
        def shuffle_stream_callback() -> Any:
            return object()

    monkeypatch.setattr(dg_data_mod, "DiskgraphLightningDataModule", _NoopDLDM)

    stub = _StubTrainer(bundle_dir="/mnt/localssd/foo/bar")
    DiskgraphNestedTrainer.init(stub)  # type: ignore[arg-type]

    assert seen["manifest_path"] == "/mnt/localssd/foo/bar/manifest.json"


# ---------------------------------------------------------------------------
# load() — TT-NG variant that delegates to Trainer.load + rebinds class.
# ---------------------------------------------------------------------------


def test_load_rebinds_class_and_sets_bundle_dir(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: ``DiskgraphNestedTrainer.load`` delegates to
    ``Trainer.load`` to restore saved config fields, then rebinds the returned
    instance's ``__class__`` to the subclass and stamps on the bundle
    dir. This is the mechanism that makes ``trainer.init()`` resolve
    to the TT-NG override on a freshly loaded best trainer."""
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    # Stand-in for the legacy load result. Plain Trainer instance
    # without going through the heavy constructor.
    from kumo_core.train.trainer import Trainer
    fake_base = Trainer.__new__(Trainer)

    captured: dict[str, Any] = {}

    def _fake_nested_load(task: Any, root_dir: str, **kwargs: Any) -> Any:
        captured["task"] = task
        captured["root_dir"] = root_dir
        captured["kwargs"] = kwargs
        return fake_base

    monkeypatch.setattr(
        Trainer, "load",
        classmethod(lambda cls, task, root_dir, **kw: _fake_nested_load(
            task, root_dir, **kw)))

    out = DiskgraphNestedTrainer.load(
        task="fake_task_config",
        root_dir="/tmp/log_dir",
        version=3,
        diskgraph_task_dataset_bundle_dir="/mnt/localssd/foo/bundle",
        encoders={"e": 1},
    )

    # Class was rebound — same instance, different class.
    assert out is fake_base
    assert isinstance(out, DiskgraphNestedTrainer)
    assert type(out) is DiskgraphNestedTrainer
    # Bundle dir was stamped on.
    assert out.diskgraph_task_dataset_bundle_dir == "/mnt/localssd/foo/bundle"
    # Omitted stores use the None defaults in Trainer.load.
    assert "feature_store" not in captured["kwargs"]
    assert "graph_store" not in captured["kwargs"]
    # encoders + version passed through.
    assert captured["kwargs"]["encoders"] == {"e": 1}
    assert captured["kwargs"]["version"] == 3


def test_load_rejects_none_bundle_dir() -> None:
    with pytest.raises(AssertionError, match="requires a task-dataset bundle"):
        DiskgraphNestedTrainer.load(
            task="fake_task_config",
            root_dir="/tmp/log_dir",
            diskgraph_task_dataset_bundle_dir=None,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# _configure_model_triple_from_manifest — Gap B / Gap C: wire model.metadata
# + encoder.table_encoder_dict + head out_channels from the bundle, so the
# wheel iterator's HeteroData and the GNN model agree on node/edge types.
# ---------------------------------------------------------------------------


def test_configure_model_triple_stamps_metadata_when_not_fully_specified(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: when model/encoder/head come back unspecified from the
    AutoML space, init() stamps each from the bundle manifest. Mirrors
    the legacy ``Trainer.initialize_model`` + encoder/head spec block
    so the GNN forward pass sees aligned node/edge types."""
    stub = _StubTrainer(
        model_fully_specified=False,
        encoder_fully_specified=False,
        head_fully_specified=False,
    )
    _run_init(stub, monkeypatch=monkeypatch)

    # Model.metadata wired from manifest graph_runtime_metadata.
    assert stub.model.metadata == (
        ["my_customers", "orders"],
        [("orders", "customer", "my_customers")],
    )
    assert stub.model.validate_called

    # Encoder: table_encoder_dict covers ALL node types from the
    # manifest plus the seed train_table block.
    expected_keys = {"my_customers", "orders", "train_table"}
    assert set(stub.encoder.table_encoder_dict.keys()) == expected_keys
    assert stub.encoder.out_channels == 64  # _StubModel.channels default
    assert stub.encoder.time_col_dict == {"my_customers": "created_at"}
    assert stub.encoder.exclude_cols_dict == {}
    assert stub.encoder.validate_called

    # Head: v1 binary classification → out_channels=1, in from model.
    assert stub.head.in_channels == 64
    assert stub.head.out_channels == 1
    assert stub.head.param_keys == []
    assert stub.head.validate_called


def test_configure_model_triple_excludes_task_columns_like_legacy(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        encoder_fully_specified=False,
        exclude_cols_dict={
            "s3_f1ec953bf2538f440a5f4431536e34eb.customers_10k":
            ["fashion_news_frequency"],
        },
    )

    _run_init(stub, monkeypatch=monkeypatch)

    assert stub.encoder.exclude_cols_dict == {
        "customers_10k": ["fashion_news_frequency"],
    }


def test_configure_model_triple_sorts_node_metadata_like_legacy(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        model_fully_specified=False,
        encoder_fully_specified=True,
        head_fully_specified=True,
    )
    fake_manifest = _FakeManifest()
    fake_manifest.graph_runtime_metadata = {
        "node_types": ["orders", "my_customers"],
        "edge_types": [{
            "src": "orders",
            "rel": "customer",
            "dst": "my_customers"
        }],
    }

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=fake_manifest)

    assert stub.model.metadata == (
        ["my_customers", "orders"],
        [("orders", "customer", "my_customers")],
    )


def test_configure_model_triple_reads_time_columns_from_feature_schema(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        model_fully_specified=True,
        encoder_fully_specified=False,
        head_fully_specified=True,
    )
    fake_manifest = _FakeManifest()
    fake_manifest.feature_schema = {
        "node_types": {
            "my_customers": {
                "feature_columns": [],
                "mapper_key": "my_customers",
            },
            "orders": {
                "feature_columns": ["TIMESTAMP"],
                "mapper_key": "orders",
                "time_column": "TIMESTAMP",
            },
        }
    }
    fake_manifest.graph_runtime_metadata = {
        "node_types": ["my_customers", "orders"],
        "edge_types": [],
    }

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=fake_manifest)

    assert stub.encoder.time_col_dict == {"orders": "TIMESTAMP"}


def test_configure_model_triple_skips_schema_time_columns_without_train_time(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(encoder_fully_specified=False, )
    fake_manifest = _FakeManifest()
    fake_manifest.metadata = {
        **fake_manifest.metadata,
        "train_table_columns": {
            "time_column": None,
        },
    }
    fake_manifest.feature_schema = {
        "node_types": {
            "my_customers": {
                "feature_columns": [],
                "mapper_key": "my_customers",
            },
            "orders": {
                "feature_columns": ["TIMESTAMP"],
                "mapper_key": "orders",
                "time_column": "TIMESTAMP",
            },
        }
    }
    fake_manifest.graph_runtime_metadata = {
        "node_types": ["my_customers", "orders"],
        "edge_types": [],
    }

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=fake_manifest)

    assert stub.encoder.time_col_dict == {}


def test_configure_model_triple_skips_time_columns_for_static_lp(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        task_type=TaskType.STATIC_LINK_PREDICTION,
        encoder_fully_specified=False,
    )
    fake_manifest = _FakeManifest()
    fake_manifest.feature_schema = {
        "node_types": {
            "my_customers": {
                "feature_columns": [],
                "mapper_key": "my_customers",
                "time_column": "created_at",
            },
        }
    }
    fake_manifest.graph_runtime_metadata = {
        "node_types": ["my_customers"],
        "edge_types": [],
    }

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=fake_manifest)

    assert stub.encoder.time_col_dict == {}


def test_configure_model_triple_sizes_multiclass_head_from_num_targets(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        model_fully_specified=False,
        encoder_fully_specified=False,
        head_fully_specified=False,
    )
    stub.num_targets = 4
    _run_init(stub, monkeypatch=monkeypatch)

    assert stub.head.in_channels == 64
    assert stub.head.out_channels == 4
    assert stub.head.param_keys == []
    assert stub.head.validate_called


def test_configure_model_triple_stamps_forecasting_head_and_sequence_encoders(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer(
        task_type=TaskType.FORECASTING,
        model_fully_specified=False,
        encoder_fully_specified=False,
        head_fully_specified=False,
    )
    stub.past_encoder = _StubSequenceEncoder()
    stub.year_over_year_encoder = _StubSequenceEncoder()
    fake_manifest = _FakeManifest()
    fake_manifest.metadata = {
        **fake_manifest.metadata,
        "task_type": "forecasting",
    }
    fake_manifest.stats = {"num_entities": 123}

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=fake_manifest)

    assert stub.past_encoder.in_channels == 1
    assert stub.past_encoder.out_channels == 64
    assert stub.past_encoder.length == 3
    assert stub.year_over_year_encoder.in_channels is not None
    assert stub.year_over_year_encoder.out_channels == 64
    assert stub.head.num_entities == 123
    assert stub.head.past_channels == 64
    assert stub.head.past_year_channels == 64
    assert stub.head.unix_step_size == 86400
    assert stub.head.out_channels == 1
    assert stub.head.validate_called


def test_configure_model_triple_skips_when_model_fully_specified(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: ``is_fully_specified`` guard — if the AutoML space already
    handed back a fully-specified model, init() must NOT overwrite its
    metadata or call validate()."""
    stub = _StubTrainer(
        model_fully_specified=True,
        encoder_fully_specified=False,
        head_fully_specified=False,
    )
    _run_init(stub, monkeypatch=monkeypatch)

    # Model: untouched.
    assert stub.model.metadata is None
    assert stub.model.validate_called is False
    # Encoder + head still configured (they were unspecified).
    assert stub.encoder.validate_called
    assert stub.head.validate_called


def test_configure_model_triple_skips_when_encoder_fully_specified(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: encoder guard — pre-specified encoder is left alone."""
    stub = _StubTrainer(
        model_fully_specified=False,
        encoder_fully_specified=True,
        head_fully_specified=False,
    )
    _run_init(stub, monkeypatch=monkeypatch)

    assert stub.encoder.table_encoder_dict == {}
    assert stub.encoder.out_channels is None
    assert stub.encoder.validate_called is False
    # Model + head still configured.
    assert stub.model.validate_called
    assert stub.head.validate_called


def test_configure_model_triple_skips_when_head_fully_specified(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: head guard — pre-specified head is left alone, preserving
    whatever out_channels the AutoML space set."""
    stub = _StubTrainer(
        model_fully_specified=False,
        encoder_fully_specified=False,
        head_fully_specified=True,
        head_out_channels=42,  # Arbitrary pre-existing value.
    )
    _run_init(stub, monkeypatch=monkeypatch)

    # Head: out_channels unchanged, validate not called.
    assert stub.head.out_channels == 42
    assert stub.head.in_channels is None
    assert stub.head.validate_called is False
    # Model + encoder still configured.
    assert stub.model.validate_called
    assert stub.encoder.validate_called


def test_configure_model_triple_threads_encoders_into_table_encoder_dict(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: ``encoders.encoders`` entries are preserved in the
    table_encoder_dict; node types without learned features get an
    empty dict via setdefault. Mirrors the legacy behavior so the
    HeteroConv layers see entries for every graph node type."""
    stub = _StubTrainer(
        model_fully_specified=False,
        encoder_fully_specified=False,
        head_fully_specified=False,
    )
    # Pre-populate encoders.encoders with a learned-feature entry.
    stub.encoders = _StubEncoders(
        encoders={"my_customers": {
            "age": "fake_encoder"
        }})
    _run_init(stub, monkeypatch=monkeypatch)

    # The learned-feature entry survives; the bare node type ("orders")
    # and the seed train_table get an empty-dict default.
    assert stub.encoder.table_encoder_dict["my_customers"] == {
        "age": "fake_encoder"
    }
    assert stub.encoder.table_encoder_dict["orders"] == {}
    assert stub.encoder.table_encoder_dict["train_table"] == {}


# ---------------------------------------------------------------------------
# Gap D — eval_loader_kwargs can be None on the parent Trainer (no
# normalization). The subclass must not crash trying to call .get(...).
# ---------------------------------------------------------------------------


def test_init_handles_none_eval_loader_kwargs(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """TT-NG: when the parent Trainer was constructed with
    ``eval_loader_kwargs=None`` (the default + no normalization),
    init() falls back to the train batch size rather than raising
    AttributeError on ``None.get(...)``."""
    stub = _StubTrainer(eval_loader_kwargs=None)
    _run_init(stub, monkeypatch=monkeypatch)

    # eval_batch_size should fall back to train_batch_size (16, from
    # _StubTrainer.loader_kwargs).
    dm_kwargs = stub.datamodule_kwargs  # type: ignore[attr-defined]
    assert dm_kwargs["eval_batch_size"] == 16
    assert dm_kwargs["train_batch_size"] == 16


# ---------------------------------------------------------------------------
# Tests for _build_ttng_sampling_plan
# ---------------------------------------------------------------------------


class _FakeNumNeighbors:
    def __init__(self, values: Any, default: Any = None) -> None:
        self.values = values
        self.default = default

    def get_values(self) -> Any:
        return self.values


def test_build_ttng_sampling_plan_non_lp() -> None:
    """Standard node classification task gets uniform default fanouts."""
    # Stub trainer
    stub = _StubTrainer()
    stub.sampler_kwargs.num_neighbors = _FakeNumNeighbors([15, 10])
    stub.sampler_kwargs.temporal_strategy = "last"
    stub.task.task_type = TaskType.BINARY_CLASSIFICATION

    # Manifest
    tm = _FakeManifest()
    tm.graph_runtime_metadata = {
        "graph_index_path": "/fake/index",
    }

    # Call method
    plan = DiskgraphNestedTrainer._build_ttng_sampling_plan(
        stub, tm)  # type: ignore[arg-type]

    # Verify standard uniform fallback plan
    assert len(plan["layers"]) == 2
    assert plan["layers"][0]["default_fanout"] == {
        "fanout": 15,
        "mode": "temporal_most_recent",
        "absent_key_policy": "uniform_fallback",
    }
    assert plan["layers"][1]["default_fanout"] == {
        "fanout": 10,
        "mode": "temporal_most_recent",
        "absent_key_policy": "uniform_fallback",
    }
    assert plan["layers"][0]["fanouts"] == {}
    assert plan["layers"][1]["fanouts"] == {}


def test_build_ttng_sampling_plan_preserves_encoded_relation_fanout(
        monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubTrainer()
    stub.sampler_kwargs.num_neighbors = _FakeNumNeighbors(
        {
            "transactions_10k__KUMO_EDGE_KEY__customer_id__KUMO_EDGE_KEY__customers_10k":
            [420, 16],
        },
        default=[16, 16],
    )
    stub.sampler_kwargs.temporal_strategy = "last"
    stub.task.task_type = TaskType.MULTILABEL_CLASSIFICATION

    tm = _FakeManifest()
    tm.graph_runtime_metadata = {
        "graph_index_path": "/fake/index",
    }

    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer._load_index_metadata",
        lambda graph_runtime_metadata: {
            "edge_relations": [{
                "name":
                "transactions_10k_customer_id_to_customers_10k_customer_id",
                "src_type": "transactions_10k",
                "dst_type": "customers_10k",
            }, {
                "name":
                "transactions_10k_customer_id_to_customers_10k_customer_id_reverse",
                "src_type": "customers_10k",
                "dst_type": "transactions_10k",
            }]
        })

    plan = DiskgraphNestedTrainer._build_ttng_sampling_plan(
        stub, tm)  # type: ignore[arg-type]

    rel = "transactions_10k_customer_id_to_customers_10k_customer_id_reverse"
    assert plan["layers"][0]["fanouts"][rel] == {
        "fanout": 420,
        "mode": "temporal_most_recent",
        "absent_key_policy": "uniform_fallback",
    }
    assert plan["layers"][0]["default_fanout"]["fanout"] == 16


def test_build_ttng_sampling_plan_list_all_neighbors_sentinel() -> None:
    """A -1 ("all neighbors") request must not degenerate to an empty plan.

    Kumo encodes "sample all neighbors" as -1; the DG engine caps the fanout
    at the node's natural degree, so -1 is translated to a large fanout rather
    than dropped (which would previously yield ``default_fanout=None`` and make
    the sampler fetch zero neighbors -> seed-only features -> XAI crash).
    """
    stub = _StubTrainer()
    stub.sampler_kwargs.num_neighbors = _FakeNumNeighbors([-1, -1])
    stub.sampler_kwargs.temporal_strategy = "last"
    stub.task.task_type = TaskType.BINARY_CLASSIFICATION

    tm = _FakeManifest()
    tm.graph_runtime_metadata = {"graph_index_path": "/fake/index"}

    plan = DiskgraphNestedTrainer._build_ttng_sampling_plan(
        stub, tm)  # type: ignore[arg-type]

    assert len(plan["layers"]) == 2
    for layer in plan["layers"]:
        assert layer["default_fanout"] is not None
        assert layer["default_fanout"]["fanout"] == 1_000_000_000


def test_build_ttng_sampling_plan_dict_all_neighbors_and_zero() -> None:
    """Per-relation -1 -> large fanout; an explicit 0 still means "skip"."""
    stub = _StubTrainer()
    stub.sampler_kwargs.num_neighbors = _FakeNumNeighbors(
        {
            ("orders", "orders_to_customers", "customers"): [-1, 0],
        }, default=[-1, 0])
    stub.sampler_kwargs.temporal_strategy = "uniform"
    stub.task.task_type = TaskType.BINARY_CLASSIFICATION

    tm = _FakeManifest()
    tm.graph_runtime_metadata = {"graph_index_path": "/fake/index"}

    import kumo_core.train.diskgraph_nested_trainer as _dnt
    orig = _dnt._load_index_metadata
    _dnt._load_index_metadata = lambda graph_runtime_metadata: {
        "edge_relations": [{
            "name": "orders_to_customers",
            "src_type": "orders",
            "dst_type": "customers",
        }]
    }
    try:
        plan = DiskgraphNestedTrainer._build_ttng_sampling_plan(
            stub, tm)  # type: ignore[arg-type]
    finally:
        _dnt._load_index_metadata = orig

    # Hop 0: -1 -> large fanout on both the per-relation and default entries.
    assert plan["layers"][0]["fanouts"]["orders_to_customers"][
        "fanout"] == 1_000_000_000
    assert plan["layers"][0]["default_fanout"]["fanout"] == 1_000_000_000
    # Hop 1: explicit 0 is dropped (skip), not translated.
    assert "orders_to_customers" not in plan["layers"][1]["fanouts"]
    assert plan["layers"][1]["default_fanout"] is None


def test_init_attaches_runtime_feature_store_for_rhs_feature_mode(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod
    from kumo_core.training.config import LinkPredModuleType

    stub = _StubTrainer(task_type=TaskType.LINK_PREDICTION)
    stub.diskgraph_runtime = object()
    stub.module_kwargs = SimpleNamespace(
        module=LinkPredModuleType.SHALLOW_RHS,
        rhs_emb_mode="feature",
        rhs_sample_size=None,
    )
    manifest = _FakeManifest()
    manifest.metadata = {
        **manifest.metadata,
        "task_type": "temporal_link_prediction",
        "target_node_type": "articles",
    }

    captured: dict[str, Any] = {}

    def _fake_build_runtime_feature_store(**kwargs: Any) -> tuple[str, str]:
        captured.update(kwargs)
        return "feature-store", "graph-store"

    monkeypatch.setattr(
        dg_nested_mod,
        "_build_runtime_feature_store",
        _fake_build_runtime_feature_store,
    )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert captured["tm"] is manifest
    assert captured["runtime"] is stub.diskgraph_runtime
    assert captured["encoders"] is stub.encoders
    assert captured["entity_table_names"] == ["my_customers", "articles"]
    assert stub.feature_store == "feature-store"
    assert stub.graph_store == "graph-store"
    assert stub.datamodule._data == ("feature-store", "graph-store")


def test_init_does_not_attach_runtime_feature_store_for_rhs_lookup_mode(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import kumo_core.train.diskgraph_nested_trainer as dg_nested_mod
    from kumo_core.training.config import LinkPredModuleType

    stub = _StubTrainer(task_type=TaskType.LINK_PREDICTION)
    stub.module_kwargs = SimpleNamespace(
        module=LinkPredModuleType.SHALLOW_RHS,
        rhs_emb_mode="lookup",
        rhs_sample_size=None,
    )
    manifest = _FakeManifest()
    manifest.metadata = {
        **manifest.metadata,
        "task_type": "temporal_link_prediction",
        "target_node_type": "articles",
    }

    monkeypatch.setattr(
        dg_nested_mod,
        "_build_runtime_feature_store",
        lambda **_: pytest.fail("lookup mode should not build feature_store"),
    )

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert not hasattr(stub, "feature_store")
    assert not hasattr(stub, "graph_store")
    assert not hasattr(stub.datamodule, "_data")


def test_init_keys_rhs_lookup_checkpoint_by_raw_primary_key(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from kumo_core.training.config import LinkPredModuleType

    stub = _StubTrainer(task_type=TaskType.LINK_PREDICTION)
    stub.module_kwargs = SimpleNamespace(
        module=LinkPredModuleType.SHALLOW_RHS,
        rhs_emb_mode="lookup",
        rhs_sample_size=None,
    )
    stub.rhs_pkeys = ["item-c", "item-a", "item-b"]
    stub.rhs_node_ids = [0, 2, 4]
    manifest = _FakeManifest()
    manifest.metadata = {
        **manifest.metadata,
        "task_type": "temporal_link_prediction",
        "target_node_type": "articles",
    }
    manifest.graph_runtime_metadata = {
        **manifest.graph_runtime_metadata,
        'graph_index_path': '/fake/index',
    }
    monkeypatch.setattr(dg_nested_mod, '_get_num_nodes_from_index_metadata',
                        lambda *_args: 5)
    monkeypatch.setattr(dg_nested_mod, '_ttng_diskgraph_runtime_config',
                        lambda *_args: None)

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    index_map = stub.pl_model_kwargs["rhs_index_map"]
    assert list(index_map.index) == ["item-c", "item-a", "item-b"]
    assert index_map.index.name == TTNG_RHS_PRIMARY_KEY_INDEX_NAME
    assert list(index_map[NODE_INDEX_COLUMN]) == [0, 2, 4]
    assert stub.pl_model_kwargs["num_rhs_nodes"] == 5
    assert stub.pl_model.canonical_rhs_node_idx.tolist() == [0, 2, 4]
    assert not hasattr(stub.pl_model, 'canonical_rhs_node_mask')


def test_init_skips_runtime_filter_for_dense_canonical_catalog(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from kumo_core.training.config import LinkPredModuleType

    stub = _StubTrainer(task_type=TaskType.LINK_PREDICTION)
    stub.module_kwargs = SimpleNamespace(
        module=LinkPredModuleType.SHALLOW_RHS,
        rhs_emb_mode="lookup",
        rhs_sample_size=None,
    )
    stub.rhs_pkeys = ["item-a", "item-b", "item-c"]
    stub.rhs_node_ids = [0, 1, 2]
    manifest = _FakeManifest()
    manifest.metadata = {
        **manifest.metadata,
        "task_type": "temporal_link_prediction",
        "target_node_type": "articles",
    }
    manifest.graph_runtime_metadata = {
        **manifest.graph_runtime_metadata,
        'graph_index_path': '/fake/index',
    }
    monkeypatch.setattr(dg_nested_mod, '_get_num_nodes_from_index_metadata',
                        lambda *_args: 3)
    monkeypatch.setattr(dg_nested_mod, '_ttng_diskgraph_runtime_config',
                        lambda *_args: None)

    _run_init(stub, monkeypatch=monkeypatch, fake_manifest=manifest)

    assert not hasattr(stub.pl_model, 'canonical_rhs_node_idx')
    assert not hasattr(stub.pl_model, 'canonical_rhs_node_mask')


def test_rhs_primary_key_index_map_rejects_duplicate_catalog() -> None:
    with pytest.raises(RuntimeError, match="duplicates"):
        _rhs_primary_key_index_map(np.asarray(["dup", "dup"]),
                                   np.asarray([0, 1]), 2)


def test_rhs_primary_key_index_map_preserves_sparse_node_ids() -> None:
    catalog = np.asarray(["item-c", "item-a", "item-b"])

    index_map = _rhs_primary_key_index_map(catalog, np.asarray([0, 2, 4]), 5)

    assert list(index_map.index) == list(catalog)
    assert index_map.index.name == TTNG_RHS_PRIMARY_KEY_INDEX_NAME
    assert list(index_map[NODE_INDEX_COLUMN]) == [0, 2, 4]


def test_rhs_primary_key_index_map_rejects_unsorted_node_ids() -> None:
    with pytest.raises(RuntimeError, match="not ordered"):
        _rhs_primary_key_index_map(np.asarray(["a", "b", "c"]),
                                   np.asarray([0, 4, 2]), 5)


def test_rhs_primary_key_catalog_reads_bounded_batches_without_sampling(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def _batches(*args: Any, **kwargs: Any) -> Any:
        calls.append((*args, kwargs))
        yield np.asarray([0, 2], dtype=np.uint64), np.asarray([11, 12],
                                                              dtype=np.int64)
        yield np.asarray([4], dtype=np.uint64), np.asarray([13], dtype=np.int64)

    monkeypatch.setattr(dg_nested_mod, 'iter_primary_key_batches', _batches)

    node_ids, catalog = _rhs_primary_key_catalog('/index', 'items', 5,
                                                 batch_size=2)

    assert node_ids.tolist() == [0, 2, 4]
    assert node_ids.dtype == np.int64
    assert catalog.tolist() == [11, 12, 13]
    assert calls == [('/index', 'items', {
        'stop': 5,
        'batch_size': 2,
    })]


def test_rhs_primary_key_catalog_preserves_decimal_values(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def _batches(*_args: Any, **_kwargs: Any) -> Any:
        yield np.asarray([0, 2], dtype=np.uint64), np.asarray(
            [Decimal('1.00'), Decimal('-2.50')], dtype=object)
        yield np.asarray([4], dtype=np.uint64), np.asarray(
            [Decimal('12345678901234567890.1234')], dtype=object)

    monkeypatch.setattr(dg_nested_mod, 'iter_primary_key_batches', _batches)

    node_ids, catalog = _rhs_primary_key_catalog('/index', 'items', 5,
                                                 batch_size=2)
    index_map = _rhs_primary_key_index_map(catalog, node_ids, 5)

    assert node_ids.tolist() == [0, 2, 4]
    assert catalog.dtype == object
    assert catalog.tolist() == [
        Decimal('1.00'),
        Decimal('-2.50'),
        Decimal('12345678901234567890.1234'),
    ]
    assert list(index_map.index) == catalog.tolist()
    assert list(index_map[NODE_INDEX_COLUMN]) == [0, 2, 4]


def test_rhs_primary_key_index_map_rejects_scale_equivalent_decimals() -> None:
    catalog = np.asarray([Decimal('1.0'), Decimal('1.00')], dtype=object)

    with pytest.raises(RuntimeError, match='duplicates'):
        _rhs_primary_key_index_map(catalog, np.asarray([0, 1]), 2)


def test_rhs_canonical_node_ids_discards_primary_key_values(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def _batches(*_args: Any, **_kwargs: Any) -> Any:
        yield np.asarray([0, 2], dtype=np.uint64), np.asarray(['a', 'b'])
        yield np.asarray([4], dtype=np.uint64), np.asarray(['c'])

    monkeypatch.setattr(dg_nested_mod, 'iter_primary_key_batches', _batches)

    node_ids = _rhs_canonical_node_ids('/index', 'items', 5, batch_size=2)

    assert node_ids.dtype == np.int64
    assert node_ids.tolist() == [0, 2, 4]


@pytest.mark.parametrize(
    'batch,error',
    [
        (np.asarray(['a', None], dtype=object), 'contains nulls'),
        (np.asarray([1.5, 2.5]), 'unsupported.*dtype'),
        (np.asarray([object()], dtype=object), 'unsupported.*object dtype'),
    ],
)
def test_rhs_primary_key_catalog_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    batch: np.ndarray,
    error: str,
) -> None:
    monkeypatch.setattr(
        dg_nested_mod,
        'iter_primary_key_batches', lambda *_args, **_kwargs: iter([(np.arange(
            len(batch), dtype=np.uint64), batch)]))
    with pytest.raises(RuntimeError, match=error):
        _rhs_primary_key_catalog('/index', 'items', len(batch))


def test_build_ttng_sampling_plan_parity_with_sampler(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``_build_ttng_sampling_plan``'s output for a fixed graph/fanout
    against a golden plan.

    This plan was originally cross-checked against the (now-retired) legacy
    ``DiskgraphNeighborSampler._build_sampling_plan`` for the same inputs —
    the two were byte-for-byte identical, which is what pinned this exact
    literal as ``EXPECTED_PLAN`` rather than deriving it inline. That
    sampler and its callers are gone (TT-NG bypasses it in production; see
    ``kumo_core.train.diskgraph_nested_trainer``), so this is now a plain
    regression test against a frozen expected value instead of a live
    parity check between two implementations.
    """
    EXPECTED_PLAN = {
        "layers": [
            {
                "fanouts": {
                    "orders_customer_to_my_customers_pk_reverse": {
                        "fanout": 15,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    },
                    "orders_customer_to_my_customers_pk": {
                        "fanout": 5,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
                "default_fanout": None,
            },
            {
                "fanouts": {
                    "orders_customer_to_my_customers_pk_reverse": {
                        "fanout": 10,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    },
                    "orders_customer_to_my_customers_pk": {
                        "fanout": 5,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    },
                },
                "default_fanout": None,
            },
        ],
    }

    stub = _StubTrainer()
    stub.sampler_kwargs.num_neighbors = _FakeNumNeighbors({
        ("orders", "orders_customer_to_my_customers_pk", "my_customers"):
        [15, 10],
        ("my_customers", "orders_customer_to_my_customers_pk_reverse", "orders"):
        [5, 5],
    })
    stub.sampler_kwargs.temporal_strategy = "uniform"
    stub.task.task_type = TaskType.BINARY_CLASSIFICATION

    tm = _FakeManifest()
    tm.graph_runtime_metadata = {
        "graph_index_path": "/fake/index",
    }

    schema_mapping = {
        "node_types": ["orders", "my_customers"],
        "edge_relations": [
            {
                "name": "orders_customer_to_my_customers_pk",
                "src_type": "orders",
                "dst_type": "my_customers"
            },
            {
                "name": "orders_customer_to_my_customers_pk_reverse",
                "src_type": "my_customers",
                "dst_type": "orders"
            },
        ],
    }
    monkeypatch.setattr(
        "kumo_core.train.diskgraph_nested_trainer._load_index_metadata",
        lambda graph_runtime_metadata: schema_mapping)

    trainer_plan = DiskgraphNestedTrainer._build_ttng_sampling_plan(
        stub, tm)  # type: ignore[arg-type]

    assert trainer_plan == EXPECTED_PLAN


# ---------------------------------------------------------------------------
# _build_eval_df — TT-NG holdout DataFrame builder.
#
# Output column names come from the bundle metadata's
# train_table_columns. Driven against a stub so we don't stand up the
# full PL eval loop.
# ---------------------------------------------------------------------------


def _bc_target_encoder() -> Any:
    """Real BC target encoder (two classes False/True), enough to drive
    ``format_prediction`` without a fitted model."""
    from kumo_core.encoders.target.categorical import CategoricalTargetEncoder
    from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper
    from kumoapi.task import TaskType as _ApiTaskType

    tm = CategoricalTensorMapper(category_counts=[["False", 100], ["True", 50]])
    return CategoricalTargetEncoder(_ApiTaskType.BINARY_CLASSIFICATION, tm)


def _reg_target_encoder() -> Any:
    from kumo_core.encoders.target.numerical import NumericalTargetEncoder

    return NumericalTargetEncoder(offset=0.0, scale=1.0)


def _mlr_target_encoder() -> Any:
    from kumo_core.encoders.target.multicategorical import (
        MultiCategoricalTargetEncoder,
    )
    from kumo_core.encoders.tensor_mapper import MultiCategoricalTensorMapper

    return MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([["A", 4], ["B", 3], ["C", 2]], sep='|'))


class _EvalDFStub:
    """Minimal stand-in exposing exactly the surface ``_build_eval_df``
    reads: ``task.task_type``, ``datamodule.task_manifest.train_table_columns``
    (the authoritative output schema), ``encoders.target.col_name``, and
    ``target_encoder``."""
    def __init__(
        self,
        *,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        seed_id_column: str = "ENTITY",
        time_column: str | None = "TIMESTAMP",
        target_col_name: str = "TARGET",
        target_encoder: Any | None = None,
    ) -> None:
        class _Task:
            pass

        self.task = _Task()
        self.task.task_type = task_type

        # Mirror the bundle metadata.json train_table_columns block.
        ttc = {
            "seed_id_column": seed_id_column,
            "target_column": target_col_name,
            "time_column": time_column,
            "weight_column": None,
        }
        self.datamodule = SimpleNamespace(
            task_manifest=SimpleNamespace(train_table_columns=ttc),
            index_np_time_unit='s')

        # ``encoder`` must be non-None: _build_eval_df gates on
        # ``encoders.target.encoder is None`` (None => LP, return None). Node-pred
        # tests need it present so the builder proceeds (mirrors production, where
        # a node-pred target always carries a fitted encoder).
        self.encoders = SimpleNamespace(
            target=SimpleNamespace(col_name=target_col_name, encoder=object()))
        self.target_encoder = target_encoder or _bc_target_encoder()
        self.pl_model = SimpleNamespace(loss=object())

    _build_eval_df = DiskgraphNestedTrainer._build_eval_df
    _eval_top_k_from_outputs = DiskgraphNestedTrainer._eval_top_k_from_outputs


def _bc_outputs(n: int = 3, *, with_time: bool = True) -> dict[str, Any]:
    """Shape-faithful stand-in for the concatenated ``pl_model.outputs``
    dict a BC eval pass produces."""
    import torch

    pkey = np.array([f"cust_{i}" for i in range(n)], dtype=object)
    pred = torch.tensor([i % 2 == 0 for i in range(n)], dtype=torch.bool)
    # prob shape is pred.shape + (2,) for BC (see format_prediction
    # sanity checks): [P(False), P(True)] per row.
    prob = torch.tensor([[0.3, 0.7] if i % 2 == 0 else [0.6, 0.4]
                         for i in range(n)])
    target = torch.tensor([[i % 2] for i in range(n)], dtype=torch.long)
    out: dict[str, Any] = dict(pkey=pkey, pred=pred, prob=prob, target=target)
    # ``outputs['time']`` is adapter-normalized canonical *nanoseconds* (the
    # runtime adapter converts the engine's seed time, default seconds, to ns
    # at the batch boundary -- see #29499). Base value is 2020-06-24 00:00:00
    # UTC (1_592_956_800_000_000_000 ns), matching the real e2e run, plus one
    # day per row.
    out["time"] = (torch.full(
        (n, ), 1_592_956_800_000_000_000, dtype=torch.long) +
                   torch.arange(n, dtype=torch.long) * 86_400_000_000_000
                   if with_time else None)
    return out


def test_build_eval_df_bc_schema_matches_legacy() -> None:
    """BC: df carries fkey/time/target + the columns downstream
    (``_get_confusion``) reads — ``TARGET`` and ``True_PROB``."""
    stub = _EvalDFStub()
    df = DiskgraphNestedTrainer._build_eval_df(stub, _bc_outputs(n=4))

    assert df is not None
    assert len(df) == 4
    # fkey + time + target (from train_table_columns), plus the
    # format_prediction-appended prediction + probability columns.
    for col in [
            "ENTITY", "TIMESTAMP", "TARGET", "TARGET_PRED", "False_PROB",
            "True_PROB"
    ]:
        assert col in df.columns, f"missing column {col!r}: {list(df.columns)}"
    # Confusion-matrix consumer reads df['TARGET'] + df['True_PROB'].
    assert df["True_PROB"].between(0.0, 1.0).all()
    assert list(df["ENTITY"]) == ["cust_0", "cust_1", "cust_2", "cust_3"]
    # Seed timestamps are unix seconds -> must land in 2020, not 1970
    # (regression for the ns/seconds unit bug found in the e2e run).
    assert df["TIMESTAMP"].iloc[0] == pd.Timestamp("2020-06-24 00:00:00")


def test_eval_df_uses_adapter_normalized_seed_time() -> None:
    import pyarrow as pa
    import torch

    seed_seconds = [1_592_956_800, 1_593_043_200]
    batch = {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint64()),
            "tt_stage_node_ids": pa.array([0, 1], type=pa.uint64()),
            "seed_node_ids": pa.array([100, 101], type=pa.uint64()),
            "seed_timestamps": pa.array(seed_seconds, type=pa.int64()),
            "target": pa.array([1.0, 0.0], type=pa.float64()),
        },
        "graphs": {},
        "raw_features": {},
        "metadata": {},
    }

    hetero, _, _ = runtime_batch_to_hetero(
        batch,
        train_table_name="train_table",
        index_np_time_unit="s",
    )
    seed_time = hetero["train_table"].time
    assert seed_time.tolist() == [
        pd.Timestamp("2020-06-24").value,
        pd.Timestamp("2020-06-25").value,
    ]

    outputs = _bc_outputs(n=2, with_time=False)
    outputs["time"] = seed_time
    outputs["pkey"] = np.array(["cust_0", "cust_1"], dtype=object)
    outputs["pred"] = torch.tensor([True, False], dtype=torch.bool)
    outputs["prob"] = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    outputs["target"] = torch.tensor([[1], [0]], dtype=torch.long)

    df = DiskgraphNestedTrainer._build_eval_df(_EvalDFStub(), outputs)

    assert df is not None
    assert list(df["TIMESTAMP"]) == [
        pd.Timestamp("2020-06-24"),
        pd.Timestamp("2020-06-25"),
    ]


def test_build_eval_df_regression_formats_predictions() -> None:
    import torch

    stub = _EvalDFStub(
        task_type=TaskType.REGRESSION,
        target_encoder=_reg_target_encoder(),
    )
    outputs = {
        "pkey": np.array(["cust_0", "cust_1"], dtype=object),
        "time": torch.tensor([1_592_956_800, 1_593_043_200]),
        "target": torch.tensor([[1.0], [2.0]]),
        "pred": torch.tensor([[1.5], [2.5]]),
        "prob": None,
    }

    df = DiskgraphNestedTrainer._build_eval_df(stub, outputs)

    assert df is not None
    assert list(df["TARGET"]) == [1.0, 2.0]
    assert list(df["TARGET_PRED"]) == [1.5, 2.5]


def test_build_eval_df_forecasting_passes_future_timestamps() -> None:
    import torch

    ns_day = 86_400 * 1_000_000_000
    stub = _EvalDFStub(
        task_type=TaskType.FORECASTING,
        target_encoder=_reg_target_encoder(),
    )
    # get_future_unix actually CONSUMES the (ns) seed time and emits anchor +
    # {1, 2} days, so a mis-scaled / overflowed seed time surfaces here as wrong
    # TIMESTAMP values rather than being silently discarded.
    stub.pl_model = SimpleNamespace(
        loss=object(),
        get_future_unix=lambda time: (time.view(-1, 1) + torch.tensor(
            [ns_day, 2 * ns_day], dtype=torch.long).view(1, -1)),
    )
    outputs = {
        "pkey":
        np.array(["cust_0", "cust_1"], dtype=object),
        # adapter-normalized canonical ns (2020-06-24 / 2020-06-25)
        "time":
        torch.tensor([1_592_956_800_000_000_000, 1_593_043_200_000_000_000],
                     dtype=torch.long),
        "target":
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "pred":
        torch.tensor([[1.5, 2.5], [3.5, 4.5]]),
        "prob":
        None,
    }

    df = DiskgraphNestedTrainer._build_eval_df(stub, outputs)

    assert df is not None
    assert list(df["ENTITY"]) == ["cust_0", "cust_0", "cust_1", "cust_1"]
    assert list(df["TARGET_PRED"]) == [1.5, 2.5, 3.5, 4.5]
    assert list(df["TIMESTAMP"]) == [
        pd.Timestamp("2020-06-25"),
        pd.Timestamp("2020-06-26"),
        pd.Timestamp("2020-06-26"),
        pd.Timestamp("2020-06-27"),
    ]
    assert list(df["ANCHOR_TIMESTAMP"]) == [
        pd.Timestamp("2020-06-24"),
        pd.Timestamp("2020-06-24"),
        pd.Timestamp("2020-06-25"),
        pd.Timestamp("2020-06-25"),
    ]


def test_forecasting_out_timestamps_passes_ns_through() -> None:
    import torch

    class _ForecastingModel:
        def get_future_unix(self, seed_time: Any) -> Any:
            step_ns = 86_400 * 1_000_000_000
            offsets = torch.tensor([0, step_ns], dtype=torch.long)
            return seed_time.view(-1, 1) + offsets.view(1, -1)

    # ``time`` reaching the egress is already adapter-normalized canonical ns
    # (2020-06-24 / 2020-06-25); get_future_unix consumes ns directly. The old
    # ``time * unit_factor`` rescale would overflow int64 and corrupt the dates.
    out = _forecasting_out_timestamps(
        _ForecastingModel(),
        torch.tensor([1_592_956_800_000_000_000, 1_593_043_200_000_000_000],
                     dtype=torch.long),
    )

    assert list(pd.to_datetime(out.numpy())) == [
        pd.Timestamp("2020-06-24"),
        pd.Timestamp("2020-06-25"),
        pd.Timestamp("2020-06-25"),
        pd.Timestamp("2020-06-26"),
    ]


def test_build_eval_df_multilabel_numeric_classes_skips_target_decode() -> None:
    import torch
    from kumo_core.encoders.target.multicategorical import (
        MultiCategoricalTargetEncoder,
    )
    from kumo_core.encoders.tensor_mapper import MultiCategoricalTensorMapper

    target_encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([[2, 10], [1, 5]], sep='|'))
    stub = _EvalDFStub(
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        target_encoder=target_encoder,
    )
    stub.num_targets = 2
    outputs = {
        "pkey": np.array(["cust_0", "cust_1"], dtype=object),
        "time": torch.tensor([1_592_956_800, 1_593_043_200]),
        "target": torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        "pred": torch.tensor([[1, 0], [1, 1]], dtype=torch.long),
        "prob": torch.tensor([[0.9, 0.2], [0.6, 0.8]]),
    }

    df = DiskgraphNestedTrainer._build_eval_df(stub, outputs)

    assert df is not None
    assert "TARGET" not in df.columns
    assert list(df["ENTITY"]) == ["cust_0", "cust_0", "cust_1", "cust_1"]
    assert list(df["CLASS"]) == [2, 1, 1, 2]
    assert list(df["LABEL"]) == [True, False, True, False]


def test_build_eval_df_multilabel_ranking_matches_legacy_long_format() -> None:
    import torch

    stub = _EvalDFStub(
        task_type=TaskType.MULTILABEL_RANKING,
        target_encoder=_mlr_target_encoder(),
    )
    stub.num_targets = 3
    outputs = {
        "pkey":
        np.array(["cust_0", "cust_1"], dtype=object),
        "time":
        torch.tensor([1_592_956_800_000_000_000, 1_593_043_200_000_000_000],
                     dtype=torch.long),
        "target":
        torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.long),
        "pred":
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "prob":
        torch.tensor([[0.8, 0.7, 0.2], [0.1, 0.9, 0.3]]),
    }

    df = DiskgraphNestedTrainer._build_eval_df(stub, outputs)

    assert df is not None
    assert list(
        df.columns) == ["ENTITY", "TIMESTAMP", "CLASS", "SCORE", "LABEL"]
    assert list(df["ENTITY"]) == [
        "cust_0", "cust_0", "cust_0", "cust_1", "cust_1", "cust_1"
    ]
    assert list(df["CLASS"]) == ["A", "B", "C", "B", "C", "A"]
    assert list(df["LABEL"]) == [True, True, False, True, False, False]


def test_build_eval_df_multilabel_ranking_numeric_classes_skips_target_decode(
) -> None:
    import torch
    from kumo_core.encoders.target.multicategorical import (
        MultiCategoricalTargetEncoder,
    )
    from kumo_core.encoders.tensor_mapper import MultiCategoricalTensorMapper

    target_encoder = MultiCategoricalTargetEncoder(
        MultiCategoricalTensorMapper([[2, 10], [1, 5]], sep='|'))
    stub = _EvalDFStub(
        task_type=TaskType.MULTILABEL_RANKING,
        target_encoder=target_encoder,
    )
    stub.num_targets = 2
    outputs = {
        "pkey":
        np.array(["cust_0", "cust_1"], dtype=object),
        "time":
        torch.tensor([1_592_956_800_000_000_000, 1_593_043_200_000_000_000],
                     dtype=torch.long),
        "target":
        torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        "pred":
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "prob":
        torch.tensor([[0.8, 0.7], [0.9, 0.1]]),
    }

    df = DiskgraphNestedTrainer._build_eval_df(stub, outputs)

    assert df is not None
    assert "TARGET" not in df.columns
    assert list(df["CLASS"]) == [2, 1, 2, 1]
    assert list(df["LABEL"]) == [True, False, False, True]


def test_build_eval_df_no_time_column_in_schema() -> None:
    """When ``train_table_columns.time_column`` is null (static task with
    no time column), no TIMESTAMP column is emitted and the
    downstream-critical columns still exist — no crash."""
    stub = _EvalDFStub(time_column=None)
    df = DiskgraphNestedTrainer._build_eval_df(stub,
                                               _bc_outputs(with_time=False))

    assert df is not None
    assert "TIMESTAMP" not in df.columns
    for col in ["ENTITY", "TARGET", "TARGET_PRED", "True_PROB"]:
        assert col in df.columns


def test_build_eval_df_time_present_but_schema_has_no_time_col() -> None:
    """Regression for the e2e crash: the wheel emits a per-seed timestamp
    on every batch, but a static dataset's schema declares
    ``time_column: null``. The builder must drop the time column rather
    than assert — column names come from the schema, not the batch."""
    stub = _EvalDFStub(time_column=None)
    df = DiskgraphNestedTrainer._build_eval_df(stub,
                                               _bc_outputs(with_time=True))

    assert df is not None
    assert "TIMESTAMP" not in df.columns
    for col in ["ENTITY", "TARGET", "TARGET_PRED", "True_PROB"]:
        assert col in df.columns


def test_predict_honors_output_options_and_progress(tmp_path) -> None:
    import torch

    class _PLModel:
        eval_top_k = None
        loss = object()

        def __init__(self) -> None:
            self.registered_stage = None

        def has_embedding_hook(self) -> bool:
            return False

        def reset_embedding_hook(self) -> None:
            raise AssertionError("no hook should be reset")

        def register_embedding_hook(self, stage: Any) -> None:
            self.registered_stage = stage

        def get_embeddings(self) -> dict[str, torch.Tensor]:
            return {"customers": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    class _PLTrainer:
        def __init__(self) -> None:
            self.callbacks = []
            self.callback_counts = []

        def predict(self, **_: Any) -> list[dict[str, Any]]:
            self.callback_counts.append(len(self.callbacks))
            output = _bc_outputs(n=2)
            output['embedding'] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            return [output]

    datamodule = SimpleNamespace(
        _seed_node_type="customers",
        time_col_dict={"customers": "created_at"},
        predict_dataloader=lambda: [object(), object()],
        **_worker_shutdown_stub(),
    )
    init_prediction_modes: list[bool] = []
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        datamodule=datamodule,
        pl_model=_PLModel(),
        pl_trainer=_PLTrainer(),
        best_ckpt="/tmp/ckpt",
        target_encoder=_bc_target_encoder(),
        init=lambda **_: init_prediction_modes.append(stub._prediction_mode),
        _preprocess_top_k=lambda top_k: top_k,
    )
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("CUSTOMER_ID", )))

    out = DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        return_embeddings=True,
        job_start_time=pd.Timestamp("2026-06-04T12:00:00"),
        ignore_time_col=True,
        pred_time_col_name="ANCHOR_TIMESTAMP",
        progress_log_dir=str(tmp_path),
        progress_log_id=7,
        total_partitions=4,
        partition_id=2,
    )

    assert stub.pl_trainer.callback_counts == [2]
    assert init_prediction_modes == [True]
    assert (tmp_path / "prediction_progress_7.json").exists()
    assert (tmp_path / "out" / "predictions.parquet" /
            "part-00002-00000.parquet").exists()
    assert not hasattr(stub.pl_model, '_kumo_return_predict_embeddings')
    assert "ANCHOR_TIMESTAMP" not in out.df.columns
    assert "JOB_TIMESTAMP" in out.df.columns
    assert "EMBEDDING" in out.embedding.columns
    assert list(out.embedding["CUSTOMER_ID"]) == ["cust_0", "cust_1"]


@pytest.mark.parametrize(
    'task_type',
    [
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
        TaskType.FORECASTING,
    ],
)
def test_predict_non_lp_embeddings_only_omits_predictions(tmp_path,
                                                          task_type) -> None:
    import torch

    prediction = {
        'pkey':
        np.array(['cust_0', 'cust_1'], dtype=object),
        'time':
        torch.tensor(
            [1_592_956_800_000_000_000, 1_593_043_200_000_000_000],
            dtype=torch.long,
        ),
        # Embeddings-only output must not depend on decoded target predictions.
        'embedding':
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }

    class _PLTrainer:
        def __init__(self) -> None:
            self.callbacks = []

        def predict(self, **kwargs):
            assert kwargs['model']._kumo_return_predict_embeddings is True
            return [prediction]

    init_calls = []
    model = SimpleNamespace(eval_top_k=None, loss=object())
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=task_type),
        datamodule=SimpleNamespace(
            _seed_node_type='customers',
            time_col_dict={'customers': 'EVENT_TIME'},
            predict_dataloader=lambda: [object()],
            **_worker_shutdown_stub(),
        ),
        pl_model=model,
        pl_trainer=_PLTrainer(),
        best_ckpt='/tmp/ckpt',
        target_encoder=_bc_target_encoder(),
        init=lambda **_: init_calls.append(True),
        _preprocess_top_k=lambda _: pytest.fail(
            'embeddings-only must not preprocess prediction top_k'),
    )

    out = DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / 'out'),
        task=SimpleNamespace(train_table=SimpleNamespace(
            foreign_key_columns=('CUSTOMER_ID', ))),
        return_embeddings=True,
        return_predictions=False,
        top_k=0,
        job_start_time=pd.Timestamp('2026-06-04T12:00:00'),
        output_extra_fields={
            'EMBEDDINGS': (['ANCHOR_TIMESTAMP', 'JOB_TIMESTAMP']
                           if task_type == TaskType.FORECASTING else []),
        },
    )

    assert init_calls == [True]
    assert out.df.empty
    assert list(out.embedding['CUSTOMER_ID']) == ['cust_0', 'cust_1']
    np.testing.assert_array_equal(
        np.stack(out.embedding['EMBEDDING']),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    assert not (tmp_path / 'out' / 'predictions.parquet').exists()
    embeddings = pd.read_parquet(tmp_path / 'out' / 'embeddings.parquet')
    assert list(embeddings['CUSTOMER_ID']) == ['cust_0', 'cust_1']
    if task_type == TaskType.FORECASTING:
        assert list(embeddings['ANCHOR_TIMESTAMP']) == [
            pd.Timestamp('2020-06-24'),
            pd.Timestamp('2020-06-25'),
        ]
        assert list(embeddings['JOB_TIMESTAMP']) == [
            pd.Timestamp('2026-06-04T12:00:00'),
            pd.Timestamp('2026-06-04T12:00:00'),
        ]
        assert 'EVENT_TIME' not in embeddings
    assert not hasattr(model, '_kumo_return_predict_embeddings')


def test_predict_non_lp_embeddings_only_chunks_without_prediction_artifact(
        tmp_path) -> None:
    import torch

    outputs = [{
        'pkey': np.array([f'cust_{i}'], dtype=object),
        'time': torch.tensor([1_592_956_800_000_000_000 + i]),
        'embedding': torch.tensor([[float(i), 0.0]]),
    } for i in range(3)]

    class _PLTrainer:
        def __init__(self) -> None:
            self.callbacks = []

        def predict(self, **kwargs):
            for batch_idx, prediction in enumerate(outputs):
                for callback in list(self.callbacks):
                    if hasattr(callback, 'write_on_batch_end'):
                        callback.write_on_batch_end(self, kwargs['model'],
                                                    prediction, None, None,
                                                    batch_idx, 0)
            for callback in list(self.callbacks):
                if hasattr(callback, 'on_predict_end'):
                    callback.on_predict_end(self, kwargs['model'])

    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.REGRESSION),
        datamodule=SimpleNamespace(
            _seed_node_type='customers',
            time_col_dict={'customers': 'EVENT_TIME'},
            predict_dataloader=lambda: [object(), object(),
                                        object()],
            **_worker_shutdown_stub(),
        ),
        pl_model=SimpleNamespace(eval_top_k=None, loss=object()),
        pl_trainer=_PLTrainer(),
        best_ckpt='/tmp/ckpt',
        target_encoder=_reg_target_encoder(),
        init=lambda **_: None,
        _preprocess_top_k=lambda _: pytest.fail(
            'embeddings-only must not preprocess prediction top_k'),
    )

    DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / 'out'),
        task=SimpleNamespace(train_table=SimpleNamespace(
            foreign_key_columns=('CUSTOMER_ID', ))),
        return_embeddings=True,
        return_predictions=False,
        output_partition_ids=[2, 3],
        total_partitions=4,
    )

    assert not (tmp_path / 'out' / 'predictions.parquet').exists()
    part_2 = pd.read_parquet(tmp_path / 'out' / 'embeddings.parquet' /
                             'part-00002-00000.parquet')
    part_3 = pd.read_parquet(tmp_path / 'out' / 'embeddings.parquet' /
                             'part-00003-00000.parquet')
    assert list(part_2['CUSTOMER_ID']) == ['cust_0', 'cust_1']
    assert list(part_3['CUSTOMER_ID']) == ['cust_2']


def test_predict_rejects_no_outputs_before_init(tmp_path) -> None:
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.REGRESSION),
        init=lambda **_: pytest.fail('no-output request must fail before init'),
    )

    with pytest.raises(ValueError, match='must request'):
        DiskgraphNestedTrainer.predict(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / 'out'),
            task=SimpleNamespace(),
            return_embeddings=False,
            return_predictions=False,
        )


@pytest.mark.parametrize(
    'extra_fields',
    [
        [],
        ['ANCHOR_TIMESTAMP'],
        ['JOB_TIMESTAMP'],
        ['ANCHOR_TIMESTAMP', 'JOB_TIMESTAMP'],
    ],
)
def test_forecasting_predict_writes_horizon_and_requested_metadata(
        tmp_path, extra_fields) -> None:
    import torch
    from kumo_core.prediction_options import resolve_prediction_time_options

    day_ns = 86_400 * 1_000_000_000
    times = torch.tensor(
        [1_592_956_800_000_000_000, 1_593_043_200_000_000_000],
        dtype=torch.long,
    )
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model = SimpleNamespace(
        eval_top_k=None,
        loss=object(),
        get_future_unix=lambda time: time.view(-1, 1) + torch.tensor(
            [day_ns, 2 * day_ns]).view(1, -1),
    )
    trainer = SimpleNamespace(
        callbacks=[],
        predict=lambda **_:
        [{
            'pkey': np.array(['cust_0', 'cust_1'], dtype=object),
            'time': times,
            'pred': values,
            'prob': None,
            'embedding': values,
        }],
    )

    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.FORECASTING),
        datamodule=SimpleNamespace(
            _seed_node_type='customers',
            time_col_dict={'customers': 'EVENT_TIME'},
            predict_dataloader=lambda: [object()],
            **_worker_shutdown_stub(),
        ),
        pl_model=model,
        pl_trainer=trainer,
        best_ckpt='/tmp/ckpt',
        target_encoder=_reg_target_encoder(),
        init=lambda **_: None,
        _preprocess_top_k=lambda top_k: top_k,
    )
    output_dir = tmp_path / ('-'.join(extra_fields) or 'no-metadata')

    time_options = resolve_prediction_time_options(
        TaskType.FORECASTING, {'PREDICTIONS': extra_fields})
    DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(output_dir),
        task=SimpleNamespace(train_table=SimpleNamespace(
            foreign_key_columns=('CUSTOMER_ID', ))),
        return_embeddings=True,
        job_start_time=pd.Timestamp('2026-06-04T12:00:00'),
        ignore_time_col=time_options.ignore_time_col,
        pred_time_col_name=time_options.pred_time_col_name,
        output_extra_fields={
            'PREDICTIONS': extra_fields,
            'EMBEDDINGS': extra_fields,
        },
    )

    predictions = pd.read_parquet(output_dir / 'predictions.parquet')
    embeddings = pd.read_parquet(output_dir / 'embeddings.parquet')
    assert list(predictions['EVENT_TIME']) == [
        pd.Timestamp('2020-06-25'),
        pd.Timestamp('2020-06-26'),
        pd.Timestamp('2020-06-26'),
        pd.Timestamp('2020-06-27'),
    ]
    assert set(predictions) == {
        'CUSTOMER_ID',
        'EVENT_TIME',
        'TARGET_PRED',
        *extra_fields,
    }
    assert set(embeddings) == {
        'CUSTOMER_ID',
        'EMBEDDING',
        *extra_fields,
    }
    if 'ANCHOR_TIMESTAMP' in extra_fields:
        assert list(predictions['ANCHOR_TIMESTAMP']) == [
            pd.Timestamp('2020-06-24'),
            pd.Timestamp('2020-06-24'),
            pd.Timestamp('2020-06-25'),
            pd.Timestamp('2020-06-25'),
        ]
        assert list(embeddings['ANCHOR_TIMESTAMP']) == [
            pd.Timestamp('2020-06-24'),
            pd.Timestamp('2020-06-25'),
        ]
    if 'JOB_TIMESTAMP' in extra_fields:
        assert predictions['JOB_TIMESTAMP'].nunique() == 1
        assert embeddings['JOB_TIMESTAMP'].nunique() == 1


def test_predict_chunks_predictions_and_embeddings_without_returning_predictions(
        tmp_path) -> None:
    import torch
    loader_state = SimpleNamespace(_iterator=object())

    class _PLModel:
        eval_top_k = None
        loss = object()

    class _PLTrainer:
        def __init__(self) -> None:
            self.callbacks = []
            self.predict_kwargs = None
            self.chunk_writer = None

        def predict(self, **kwargs: Any) -> None:
            self.predict_kwargs = kwargs
            self.chunk_writer = next(
                callback for callback in self.callbacks
                if callback.__class__.__name__ == "_ChunkedPredictionWriter")
            predictions = []
            for i in range(3):
                prediction = _bc_outputs(n=1)
                prediction["embedding"] = torch.tensor([[float(i), 0.0]])
                predictions.append(prediction)
            for batch_idx, prediction in enumerate(predictions):
                for callback in list(self.callbacks):
                    if hasattr(callback, "write_on_batch_end"):
                        callback.write_on_batch_end(self, kwargs["model"],
                                                    prediction, None, None,
                                                    batch_idx, 0)
            for callback in list(self.callbacks):
                if hasattr(callback, "on_predict_end"):
                    callback.on_predict_end(self, kwargs["model"])
            assert loader_state._iterator is None
            shutdown_events.append(('lightning-teardown', ))
            return None

    shutdown_events = []

    @contextmanager
    def defer_shutdown(phase):
        shutdown_events.append(('defer-enter', phase))
        try:
            yield
        finally:
            shutdown_events.append(('defer-exit', phase))

    shutdown_complete = False

    def shutdown_workers(phase):
        nonlocal shutdown_complete
        if shutdown_complete:
            return
        shutdown_complete = True
        loader_state._iterator = None
        output_exists = (tmp_path / 'out' / 'predictions.parquet' /
                         'part-00003-00000.parquet').exists()
        shutdown_events.append(('shutdown', phase, output_exists))

    datamodule = SimpleNamespace(
        _seed_node_type="customers",
        time_col_dict={"customers": "created_at"},
        predict_dataloader=lambda: [object(), object(),
                                    object()],
        defer_dataloader_worker_shutdown=defer_shutdown,
        shutdown_dataloader_workers=shutdown_workers,
    )
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        datamodule=datamodule,
        pl_model=_PLModel(),
        pl_trainer=_PLTrainer(),
        best_ckpt="/tmp/ckpt",
        target_encoder=_bc_target_encoder(),
        init=lambda **_: None,
        _preprocess_top_k=lambda top_k: top_k,
    )
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("CUSTOMER_ID", )))

    out = DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        return_embeddings=True,
        ignore_time_col=True,
        total_partitions=4,
        partition_id=2,
        output_partition_ids=[2, 3],
    )

    assert stub.pl_trainer.predict_kwargs["return_predictions"] is False
    assert (tmp_path / "out" / "predictions.parquet" /
            "part-00002-00000.parquet").exists()
    assert (tmp_path / "out" / "predictions.parquet" /
            "part-00003-00000.parquet").exists()
    assert (tmp_path / "out" / "embeddings.parquet" /
            "part-00002-00000.parquet").exists()
    assert (tmp_path / "out" / "embeddings.parquet" /
            "part-00003-00000.parquet").exists()
    assert not hasattr(stub.pl_trainer.chunk_writer, "saved_outputs")
    assert stub.pl_trainer.chunk_writer.saved_output_count == 2
    assert len(out.df) == 1
    assert "EMBEDDING" in out.embedding.columns
    assert shutdown_events == [
        ('defer-enter', 'predict'),
        ('shutdown', 'predict', True),
        ('lightning-teardown', ),
        ('defer-exit', 'predict'),
    ]


def test_predict_streams_multiple_batches_into_single_partition(
        tmp_path) -> None:
    class _PLModel:
        eval_top_k = None
        loss = object()

    class _PLTrainer:
        def __init__(self) -> None:
            self.callbacks = []
            self.writer = None

        def predict(self, **kwargs: Any) -> None:
            self.writer = next(
                callback for callback in self.callbacks
                if callback.__class__.__name__ == '_ChunkedPredictionWriter')
            for batch_idx in range(3):
                prediction = _bc_outputs(n=1)
                prediction['pkey'] = np.asarray([f'cust_{batch_idx}'])
                pred = prediction.pop('pred')
                prob = prediction.pop('prob')
                prediction['logits'] = np.asarray([0.0])
                for callback in list(self.callbacks):
                    if hasattr(callback, 'write_on_batch_end'):
                        callback.write_on_batch_end(self, kwargs['model'],
                                                    prediction, None, None,
                                                    batch_idx, 0)
                # Lightning module hooks run after callback hooks and decode
                # the raw logits in place.
                prediction.pop('logits')
                prediction['pred'] = pred
                prediction['prob'] = prob
            for callback in list(self.callbacks):
                if hasattr(callback, 'on_predict_end'):
                    callback.on_predict_end(self, kwargs['model'])

    trainer = _PLTrainer()
    datamodule = SimpleNamespace(
        _seed_node_type='customers',
        time_col_dict={'customers': 'created_at'},
        predict_dataloader=lambda: [object(), object(),
                                    object()],
        **_worker_shutdown_stub(),
    )
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        datamodule=datamodule,
        pl_model=_PLModel(),
        pl_trainer=trainer,
        best_ckpt='/tmp/ckpt',
        target_encoder=_bc_target_encoder(),
        init=lambda **_: None,
        _preprocess_top_k=lambda top_k: top_k,
    )
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=('CUSTOMER_ID', )))

    DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / 'out'),
        task=task,
        ignore_time_col=True,
    )

    output = pd.read_parquet(tmp_path / 'out' / 'predictions.parquet')
    assert list(output['CUSTOMER_ID']) == ['cust_0', 'cust_1', 'cust_2']
    assert trainer.writer.batch_count == 3
    assert not hasattr(trainer.writer, 'step_outputs')


@pytest.mark.parametrize('return_embeddings', [False, True])
def test_predict_multilabel_ranking_matches_legacy_long_format(
        tmp_path, return_embeddings) -> None:
    import torch

    class _PLModel:
        eval_top_k = None
        loss = object()

    class _PLTrainer:
        def __init__(self) -> None:
            self.callbacks = []

        def predict(self, **_: Any) -> list[dict[str, Any]]:
            return [{
                "pkey":
                np.array(["cust_0", "cust_1"], dtype=object),
                "time":
                torch.tensor([
                    1_592_956_800_000_000_000,
                    1_593_043_200_000_000_000,
                ], dtype=torch.long),
                "target":
                None,
                "pred":
                torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "prob":
                torch.tensor([[0.8, 0.7, 0.2], [0.1, 0.9, 0.3]]),
                "embedding":
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            }]

    datamodule = SimpleNamespace(
        _seed_node_type="customers",
        time_col_dict={"customers": "created_at"},
        predict_dataloader=lambda: [object()],
        **_worker_shutdown_stub(),
    )
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.MULTILABEL_RANKING),
        datamodule=datamodule,
        pl_model=_PLModel(),
        pl_trainer=_PLTrainer(),
        best_ckpt="/tmp/ckpt",
        target_encoder=_mlr_target_encoder(),
        init=lambda **_: None,
        _preprocess_top_k=lambda top_k: top_k,
    )
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("CUSTOMER_ID", )))

    out = DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        top_k=2,
        return_embeddings=return_embeddings,
    )

    assert list(
        out.df.columns) == ["CUSTOMER_ID", "created_at", "CLASS", "SCORE"]
    assert list(
        out.df["CUSTOMER_ID"]) == ["cust_0", "cust_0", "cust_1", "cust_1"]
    assert list(out.df["CLASS"]) == ["A", "B", "B", "C"]
    assert list(out.df["SCORE"]) == pytest.approx([0.8, 0.7, 0.9, 0.3])
    assert (tmp_path / "out" / "predictions.parquet" /
            "part-00000-00000.parquet").exists()
    assert (tmp_path / "out" /
            "embeddings.parquet").exists() is return_embeddings


def _lp_stub(pl_model: Any, *, rhs_pkeys: list[Any]) -> Any:
    """Build a stub trainer for ``_predict_lp`` with a fake RHS dataloader."""
    rhs_name = "items"

    def rhs_predict_dataloader() -> list[Any]:
        batch = {
            rhs_name:
            SimpleNamespace(
                batch_size=len(rhs_pkeys),
                pkey=np.asarray(rhs_pkeys),
            )
        }
        return [batch]

    datamodule = SimpleNamespace(
        _seed_node_type="users",
        time_col_dict={"users": "created_at"},
        index_np_time_unit="s",
        predict_dataloader=lambda: [object()],
        rhs_predict_dataloader=rhs_predict_dataloader,
        task_manifest=SimpleNamespace(
            target_node_type=rhs_name,
            graph_runtime_metadata={'graph_index_path': '/unused'},
        ),
        **_worker_shutdown_stub(),
    )
    return SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.TEMPORAL_LINK_PREDICTION),
        datamodule=datamodule,
        pl_model=pl_model,
        pl_trainer=SimpleNamespace(callbacks=[]),
        best_ckpt="/tmp/ckpt",
        init=lambda **_: None,
        _rhs_primary_key_catalog=np.asarray(rhs_pkeys),
        _rhs_primary_key_node_ids=np.arange(len(rhs_pkeys), dtype=np.uint64),
    )


class _TwoTowerStubModel:
    """Stub mimicking the two-tower (LinkGNN) embedding-hook contract."""

    num_rhs_nodes = 3
    pred_top_k = None
    eval_top_k = None
    lhs_emb_name = "users"
    rhs_emb_name = "items"
    entity_table_names = ("users", "items")

    def __init__(self, step_outputs: list[dict[str, Any]]) -> None:
        import torch
        self._step_outputs = step_outputs
        self._hooked = False
        self._got = False
        self.mips_index = SimpleNamespace(
            _key=torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]))

    # Embedding hook contract:
    def _get_register_embedding_hook_fn(self) -> Any:
        return object()  # non-None => family supports embeddings

    def has_embedding_hook(self) -> bool:
        return self._hooked

    def reset_embedding_hook(self) -> None:
        self._hooked = False

    def register_embedding_hook(self, stage: Any) -> None:
        self._hooked = True

    def get_embeddings(self) -> dict[str, Any]:
        import torch
        self._got = True
        return {
            self.lhs_emb_name: torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            self.rhs_emb_name: torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0,
                                                                      5.0]]),
        }


class _ShallowRhsStubModel(_TwoTowerStubModel):
    def __init__(self, step_outputs: list[dict[str, Any]]) -> None:
        super().__init__(step_outputs)
        self.pending_rhs_index_map: pd.DataFrame | None = None

    def set_rhs_index_map(self, index_map: pd.DataFrame) -> None:
        self.pending_rhs_index_map = index_map


class _LPCallbackTrainer:
    def __init__(self, step_outputs: list[dict[str, Any]]) -> None:
        self.callbacks: list[Any] = []
        self.step_outputs = step_outputs
        self.predict_kwargs: dict[str, Any] | None = None

    def predict(self, **kwargs: Any) -> None:
        self.predict_kwargs = kwargs
        for batch_idx, prediction in enumerate(self.step_outputs):
            for callback in list(self.callbacks):
                if hasattr(callback, "write_on_batch_end"):
                    callback.write_on_batch_end(self, kwargs["model"],
                                                prediction, None, None,
                                                batch_idx, 0)
        for callback in list(self.callbacks):
            if hasattr(callback, "on_predict_end"):
                callback.on_predict_end(self, kwargs["model"])
        return None


def test_predict_lp_emits_embeddings_when_requested(tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),  # 2 LHS anchors
        "target": torch.tensor([[0, 1], [2, 0]]),  # (N, k=2) dense RHS ids
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
        "lhs_embedding": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.pl_trainer.predict = lambda **_: step_outputs
    shutdown_events = []

    @contextmanager
    def defer_shutdown(phase):
        shutdown_events.append(('defer-enter', phase))
        try:
            yield
        finally:
            shutdown_events.append(('defer-exit', phase))

    def shutdown_workers(phase):
        output_exists = (tmp_path / 'out' / 'predictions.parquet' /
                         'part-00000-00000.parquet').exists()
        shutdown_events.append(('shutdown', phase, output_exists))

    stub.datamodule.defer_dataloader_worker_shutdown = defer_shutdown
    stub.datamodule.shutdown_dataloader_workers = shutdown_workers
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    out = DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        return_embeddings=True,
    )

    # Predictions still written:
    assert (tmp_path / "out" / "predictions.parquet").exists()
    assert list(out.df["CLASS"]) == ["item_a", "item_b", "item_c", "item_a"]
    # Embeddings written + passed to PredOutput:
    assert model._got is False
    assert out.embedding is not None
    assert (tmp_path / "out" / "embeddings.parquet").exists()
    assert list(out.embedding.columns[:2]) == ["ID", "TYPE"]
    assert shutdown_events == [
        ('defer-enter', 'predict'),
        ('shutdown', 'predict', True),
        ('defer-exit', 'predict'),
    ]
    assert "EMBEDDING" in out.embedding.columns
    # The returned value is the final bounded model batch; the persisted stream
    # contains all LHS anchors followed by the partition's RHS catalog.
    embedding_df = pd.read_parquet(tmp_path / "out" / "embeddings.parquet")
    assert list(embedding_df["ID"].astype(str)) == [
        "10", "11", "item_a", "item_b", "item_c"
    ]
    assert list(
        embedding_df["TYPE"]) == ["users", "users", "items", "items", "items"]
    assert len(out.embedding["EMBEDDING"].iloc[0]) == 2


@pytest.mark.parametrize('top_k', [None, 0, 5])
@pytest.mark.parametrize('rhs_pkeys', [[], ['item_a', 'item_b']])
def test_predict_lp_embeddings_only_never_enters_mips(tmp_path, top_k,
                                                      rhs_pkeys) -> None:
    """Regression for a future-anchor, empty-candidate catalog."""
    import torch

    step_outputs = [{
        'pkey': np.array(['user_a'], dtype=object),
        'time': torch.tensor([1_700_000_000_000_000_000]),
        'lhs_embedding': torch.tensor([[1.0, 2.0]]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    model.num_rhs_nodes = len(rhs_pkeys)

    class _NoSearchMips:
        def __init__(self) -> None:
            self._key = torch.tensor([[3.0, 4.0]] * len(rhs_pkeys))

        def search(self, *_args, **_kwargs):
            pytest.fail('embeddings-only prediction must not call MIPS')

    model.mips_index = _NoSearchMips()
    stub = _lp_stub(model, rhs_pkeys=rhs_pkeys)

    class _EmbeddingOnlyTrainer(_LPCallbackTrainer):
        def predict(self, **kwargs):
            assert kwargs['model'].pred_top_k == 0
            return super().predict(**kwargs)

    stub.pl_trainer = _EmbeddingOnlyTrainer(step_outputs)
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=('USER_ID', 'ITEM_ID')))

    out = DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / 'out'),
        task=task,
        return_embeddings=True,
        return_predictions=False,
        top_k=top_k,
    )

    assert out.df.empty
    assert out.embedding is not None
    assert list(out.embedding.columns) == ['ID', 'TYPE', 'EMBEDDING']
    assert not (tmp_path / 'out' / 'predictions.parquet').exists()
    embedding_df = pd.read_parquet(tmp_path / 'out' / 'embeddings.parquet')
    assert list(embedding_df.columns) == ['ID', 'TYPE', 'EMBEDDING']
    assert list(embedding_df['ID']) == ['user_a', *rhs_pkeys]
    assert list(
        embedding_df['TYPE']) == ['users', *(['items'] * len(rhs_pkeys))]


def test_predict_lp_embeddings_only_uses_exact_temporal_rhs_catalog(
        tmp_path) -> None:
    import torch

    step_outputs = [{
        'pkey': np.array(['user_a'], dtype=object),
        'time': torch.tensor([1_700_000_000_000_000_000]),
        'lhs_embedding': torch.tensor([[1.0, 2.0]]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    model.num_rhs_nodes = 4
    # This is the non-prefix catalog selected by the temporal PRED-stage
    # filter before its MIPS index is built.
    model.valid_rhs_node_idx = torch.tensor([1, 3], dtype=torch.int64)
    model.mips_index = SimpleNamespace(
        _key=torch.tensor([[11.0, 12.0], [31.0, 32.0]]))
    stub = _lp_stub(
        model,
        rhs_pkeys=['expired_a', 'valid_b', 'expired_c', 'valid_d'],
    )
    stub.pl_trainer = _LPCallbackTrainer(step_outputs)
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=('USER_ID', 'ITEM_ID')))

    DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / 'out'),
        task=task,
        return_embeddings=True,
        return_predictions=False,
    )

    embeddings = pd.read_parquet(tmp_path / 'out' / 'embeddings.parquet')
    assert list(embeddings['ID']) == ['user_a', 'valid_b', 'valid_d']
    assert list(embeddings['TYPE']) == ['users', 'items', 'items']
    np.testing.assert_array_equal(
        np.stack(embeddings['EMBEDDING']),
        np.array([[1.0, 2.0], [11.0, 12.0], [31.0, 32.0]], dtype=np.float32),
    )


@pytest.mark.parametrize('rhs_keys', [
    ['item_a', 'item_b', 'item_c'],
    [2**60 + 1, 2**60 + 2, 2**60 + 3],
    [2**63 + 1, 2**63 + 2, 2**63 + 3],
])
@pytest.mark.parametrize('pad_last', [False, True])
@pytest.mark.parametrize('return_embeddings', [False, True])
@pytest.mark.parametrize(
    ('top_k', 'model_top_k', 'expected_top_k'),
    [(None, 7, 3), (2, None, 2), (5, None, 3)],
)
def test_predict_lp_prediction_outputs_preserve_top_k_behavior(
        tmp_path, return_embeddings, top_k, model_top_k, expected_top_k,
        rhs_keys, pad_last) -> None:
    import torch

    step_outputs = [{
        'pkey':
        torch.tensor([10]),
        'target':
        torch.arange(expected_top_k).reshape(1, -1),
        'score':
        torch.linspace(0.9, 0.7, expected_top_k).reshape(1, -1),
        'time':
        torch.tensor([1_700_000_000_000_000_000]),
        'lhs_embedding':
        torch.tensor([[1.0, 2.0]]),
    }]
    if pad_last:
        step_outputs[0]['target'][0, -1] = -1
    model = _TwoTowerStubModel(step_outputs)
    model.pred_top_k = model_top_k
    stub = _lp_stub(model, rhs_pkeys=rhs_keys)
    stub.pl_trainer = _LPCallbackTrainer(step_outputs)
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=('USER_ID', 'ITEM_ID')))

    DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / 'out'),
        task=task,
        return_embeddings=return_embeddings,
        return_predictions=True,
        top_k=top_k,
    )

    assert model.pred_top_k == expected_top_k
    assert (tmp_path / 'out' / 'predictions.parquet' /
            'part-00000-00000.parquet').exists()
    predictions = pd.read_parquet(tmp_path / 'out' / 'predictions.parquet')
    assert len(predictions) == expected_top_k
    assert predictions['CLASS'].isna().sum() == int(pad_last)
    real_count = expected_top_k - int(pad_last)
    assert predictions['CLASS'].dropna().tolist() == rhs_keys[:real_count]
    if isinstance(rhs_keys[0], int):
        expected_dtype = 'UInt64' if rhs_keys[0] >= 2**63 else 'Int64'
        assert str(predictions['CLASS'].dtype) == expected_dtype
    assert (tmp_path / 'out' /
            'embeddings.parquet').exists() is (return_embeddings)


@pytest.mark.parametrize('return_embeddings', [False, True])
def test_predict_lp_predictions_reject_nonpositive_top_k(
        tmp_path, return_embeddings) -> None:
    model = _TwoTowerStubModel([])
    stub = _lp_stub(model, rhs_pkeys=['item_a'])
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=('USER_ID', 'ITEM_ID')))

    with pytest.raises(ValueError, match='top_k must be greater than 0'):
        DiskgraphNestedTrainer._predict_lp(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / 'out'),
            task=task,
            return_embeddings=return_embeddings,
            return_predictions=True,
            top_k=0,
        )


def test_predict_lp_rejects_request_without_outputs(tmp_path) -> None:
    model = _TwoTowerStubModel([])
    stub = _lp_stub(model, rhs_pkeys=['item_a'])
    stub.init = lambda **_: pytest.fail('no-output request must fail before init')
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=('USER_ID', 'ITEM_ID')))

    with pytest.raises(ValueError, match='must request'):
        DiskgraphNestedTrainer._predict_lp(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / 'out'),
            task=task,
            return_embeddings=False,
            return_predictions=False,
        )


def test_predict_lp_no_embeddings_when_not_requested(tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0, 1], [2, 0]]),
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.pl_trainer.predict = lambda **_: step_outputs
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    out = DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        return_embeddings=False,
    )

    assert model._got is False
    assert out.embedding is None
    assert not (tmp_path / "out" / "embeddings.parquet").exists()


def test_predict_lp_queues_raw_rhs_catalog_for_checkpoint_reorder(
        tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10]),
        "target": torch.tensor([[0, 2]]),
        "score": torch.tensor([[0.9, 0.7]]),
        "time": torch.tensor([1700000000]),
    }]
    model = _ShallowRhsStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item-c", "item-a", "item-b"])
    stub.pl_trainer.predict = lambda **_: step_outputs
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
    )

    assert model.pending_rhs_index_map is not None
    assert list(
        model.pending_rhs_index_map.index) == ["item-c", "item-a", "item-b"]
    assert (model.pending_rhs_index_map.index.name ==
            TTNG_RHS_PRIMARY_KEY_INDEX_NAME)
    assert list(model.pending_rhs_index_map[NODE_INDEX_COLUMN]) == [0, 1, 2]


def test_predict_lp_exception_aborts_already_closed_partial_output(
        tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0, 1], [2, 0]]),
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])

    class _FailAfterPredictEnd(_LPCallbackTrainer):
        def predict(self, **kwargs):
            super().predict(**kwargs)
            raise RuntimeError("upstream prediction failed after callbacks")

    stub.pl_trainer = _FailAfterPredictEnd(step_outputs)
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    with pytest.raises(RuntimeError, match="upstream prediction failed"):
        DiskgraphNestedTrainer._predict_lp(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / "out"),
            task=task,
        )

    assert not list((tmp_path / "out").rglob("part-*.parquet"))


def test_predict_lp_cleanup_failure_aborts_closed_output(tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0, 1], [2, 0]]),
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.pl_trainer = _LPCallbackTrainer(step_outputs)

    @contextmanager
    def defer_shutdown(_phase):
        yield

    def fail_shutdown(_phase):
        assert list((tmp_path / "out").rglob("part-*.parquet"))
        raise RuntimeError("synthetic worker cleanup failure")

    stub.datamodule.defer_dataloader_worker_shutdown = defer_shutdown
    stub.datamodule.shutdown_dataloader_workers = fail_shutdown
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    with pytest.raises(RuntimeError, match="synthetic worker cleanup failure"):
        DiskgraphNestedTrainer._predict_lp(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / "out"),
            task=task,
        )

    assert not list((tmp_path / "out").rglob("part-*.parquet"))


def test_predict_lp_cleanup_base_exception_preserves_prediction_error(
        tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0, 1], [2, 0]]),
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])

    class _FailBeforeCallbacks(_LPCallbackTrainer):
        def predict(self, **kwargs):
            raise RuntimeError("primary prediction failure")

    @contextmanager
    def defer_shutdown(_phase):
        yield

    def interrupt_shutdown(_phase):
        raise KeyboardInterrupt("secondary cleanup interruption")

    stub.pl_trainer = _FailBeforeCallbacks(step_outputs)
    stub.datamodule.defer_dataloader_worker_shutdown = defer_shutdown
    stub.datamodule.shutdown_dataloader_workers = interrupt_shutdown
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    with pytest.raises(RuntimeError, match="primary prediction failure"):
        DiskgraphNestedTrainer._predict_lp(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / "out"),
            task=task,
        )

    assert not list((tmp_path / "out").rglob("part-*.parquet"))


def test_predict_lp_writes_requested_partition_id(tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0, 1], [2, 0]]),
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.pl_trainer.predict = lambda **_: step_outputs
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        partition_id=1,
        total_partitions=2,
    )

    assert not (tmp_path / "out" / "predictions.parquet" /
                "part-00000-00000.parquet").exists()
    assert (tmp_path / "out" / "predictions.parquet" /
            "part-00001-00000.parquet").exists()


def test_predict_lp_slices_rhs_embeddings_for_single_assigned_output_partition(
        tmp_path) -> None:
    import torch

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0], [1]]),
        "score": torch.tensor([[0.9], [0.8]]),
        "time": torch.tensor([1700000000, 1700000001]),
        "lhs_embedding": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
    }]
    rhs_pkeys = ["item_a", "item_b", "item_c", "item_d", "item_e"]
    model = _TwoTowerStubModel(step_outputs)
    model.num_rhs_nodes = len(rhs_pkeys)
    model.mips_index = SimpleNamespace(_key=torch.tensor([
        [20.0, 0.0],
        [21.0, 0.0],
        [22.0, 0.0],
        [23.0, 0.0],
        [24.0, 0.0],
    ]))
    model.get_embeddings = lambda: {
        "users":
        torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        "items":
        torch.tensor([
            [20.0, 0.0],
            [21.0, 0.0],
            [22.0, 0.0],
            [23.0, 0.0],
            [24.0, 0.0],
        ]),
    }
    stub = _lp_stub(model, rhs_pkeys=rhs_pkeys)
    stub.pl_trainer.predict = lambda **_: step_outputs
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        return_embeddings=True,
        top_k=1,
        partition_id=1,
        total_partitions=2,
        output_partition_ids=[1],
    )

    assert (tmp_path / "out" / "embeddings.parquet" /
            "part-00001-00000.parquet").exists()
    assert model.pred_top_k == 1
    embedding_df = pd.read_parquet(tmp_path / "out" / "embeddings.parquet" /
                                   "part-00001-00000.parquet")
    assert list(
        embedding_df["ID"].astype(str)) == ["10", "11", "item_d", "item_e"]


def test_predict_lp_writes_assigned_output_partitions(tmp_path) -> None:
    import torch

    step_outputs = [
        {
            "pkey": torch.tensor([10]),
            "target": torch.tensor([[0]]),
            "score": torch.tensor([[0.9]]),
            "time": torch.tensor([1700000000]),
        },
        {
            "pkey": torch.tensor([11]),
            "target": torch.tensor([[1]]),
            "score": torch.tensor([[0.8]]),
            "time": torch.tensor([1700000001]),
        },
        {
            "pkey": torch.tensor([12]),
            "target": torch.tensor([[2]]),
            "score": torch.tensor([[0.7]]),
            "time": torch.tensor([1700000002]),
        },
    ]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.datamodule.predict_dataloader = lambda: [object(), object(), object()]
    stub.pl_trainer = _LPCallbackTrainer(step_outputs)
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        total_partitions=4,
        output_partition_ids=[2, 3],
    )

    assert stub.pl_trainer.predict_kwargs["return_predictions"] is False
    part_2 = tmp_path / "out" / "predictions.parquet" / \
        "part-00002-00000.parquet"
    part_3 = tmp_path / "out" / "predictions.parquet" / \
        "part-00003-00000.parquet"
    assert part_2.exists()
    assert part_3.exists()
    assert list(pd.read_parquet(part_2)["USER_ID"]) == [10, 11]
    assert list(pd.read_parquet(part_3)["USER_ID"]) == [12]


def test_predict_lp_chunks_lhs_and_rhs_embeddings_by_output_partition(
        tmp_path) -> None:
    import torch

    step_outputs = [
        {
            "pkey": torch.tensor([10]),
            "time": torch.tensor([1700000000]),
            "lhs_embedding": torch.tensor([[10.0, 0.0]]),
        },
        {
            "pkey": torch.tensor([11]),
            "time": torch.tensor([1700000001]),
            "lhs_embedding": torch.tensor([[11.0, 0.0]]),
        },
        {
            "pkey": torch.tensor([12]),
            "time": torch.tensor([1700000002]),
            "lhs_embedding": torch.tensor([[12.0, 0.0]]),
        },
    ]
    rhs_pkeys = ["item_a", "item_b", "item_c", "item_d", "item_e"]
    model = _TwoTowerStubModel(step_outputs)
    model.num_rhs_nodes = len(rhs_pkeys)
    model.mips_index = SimpleNamespace(_key=torch.tensor([
        [20.0, 0.0],
        [21.0, 0.0],
        [22.0, 0.0],
        [23.0, 0.0],
        [24.0, 0.0],
    ]))
    stub = _lp_stub(model, rhs_pkeys=rhs_pkeys)
    stub.datamodule.predict_dataloader = lambda: [object(), object(), object()]
    # Partitioned output reuses the PK-only catalog already owned by this
    # worker process instead of invoking a sampling dataloader per partition.
    stub.datamodule.rhs_predict_dataloader = lambda: pytest.fail(
        'rhs_predict_dataloader must not be used for catalog output')
    stub.pl_trainer = _LPCallbackTrainer(step_outputs)
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    out = DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
        return_embeddings=True,
        return_predictions=False,
        total_partitions=2,
        output_partition_ids=[0, 1],
    )

    assert stub.pl_trainer.predict_kwargs["return_predictions"] is False
    assert not hasattr(model, "_kumo_return_predict_embeddings")
    assert not (tmp_path / "out" / "predictions.parquet").exists()
    part_0 = tmp_path / "out" / "embeddings.parquet" / \
        "part-00000-00000.parquet"
    part_1 = tmp_path / "out" / "embeddings.parquet" / \
        "part-00001-00000.parquet"
    assert part_0.exists()
    assert part_1.exists()
    assert list(pd.read_parquet(part_0)["ID"].astype(str)) == [
        "10", "11", "item_a", "item_b", "item_c"
    ]
    assert list(pd.read_parquet(part_1)["ID"].astype(str)) == [
        "12", "item_d", "item_e"
    ]
    assert list(out.embedding["ID"].astype(str)) == ["12"]


def test_predict_lp_rejects_embeddings_when_family_has_no_embedding_hook(
        tmp_path) -> None:
    import torch

    class _NoHookModel(_TwoTowerStubModel):
        def _get_register_embedding_hook_fn(self) -> Any:
            return None  # ID-GNN / hybrid: no embedding hook

    step_outputs = [{
        "pkey": torch.tensor([10, 11]),
        "target": torch.tensor([[0, 1], [2, 0]]),
        "score": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time": torch.tensor([1700000000, 1700000001]),
    }]
    model = _NoHookModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.pl_trainer.predict = lambda **_: step_outputs
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    with pytest.raises(ValueError,
                       match='does not implement an embedding hook'):
        DiskgraphNestedTrainer._predict_lp(
            stub,  # type: ignore[arg-type]
            output_dir=str(tmp_path / "out"),
            task=task,
            return_embeddings=True,
        )

    assert model._hooked is False
    assert model._got is False
    assert not (tmp_path / "out" / "embeddings.parquet").exists()


# ---------------------------------------------------------------------------
# Regression: the prediction/eval-output TIMESTAMP egress must read the
# already-nanosecond seed time as datetime64[ns] directly.
#
# The runtime adapter normalizes the engine's seed time (engine unit, default
# seconds) to canonical nanoseconds at the batch boundary (#29499). So the
# ``outputs['time']`` reaching these egress sites is ALREADY int64 ns. The old
# egress re-applied a seconds interpretation (``.astype('datetime64[s]')`` via
# ``index_np_time_unit``), so a ~1.4e18 ns value got read as seconds and the
# subsequent s->ns promotion (x1e9) overflowed int64 -> wrapped to year ~1730.
# These tests feed a realistic ns seed time (2014-07-23) and assert the output
# lands in 2014, not 1730.
# ---------------------------------------------------------------------------

# 2014-07-23 00:00:00 UTC expressed in nanoseconds since the unix epoch.
# This is exactly the value the adapter-normalized ``outputs['time']`` carries.
_NS_2014_07_23 = 1_406_073_600_000_000_000


def test_seed_time_to_datetime64_ns_preserves_year() -> None:
    """The helper reads an already-ns int64 time directly as ns; a 2014 ns
    value must round-trip to 2014 (the old seconds-rescale overflowed to
    ~1730)."""
    import torch

    out = _seed_time_to_datetime64_ns(torch.tensor([_NS_2014_07_23]))
    ts = pd.Timestamp(np.asarray(out)[0])
    assert ts == pd.Timestamp("2014-07-23 00:00:00")

    # Demonstrate the old (buggy) seconds-rescale on the SAME ns value lands in
    # year 1730 -- this is the artifact the fix removes.
    buggy = (np.asarray(
        [_NS_2014_07_23],
        dtype='int64').astype('datetime64[s]').astype('datetime64[ns]'))
    assert pd.Timestamp(buggy[0]).year == 1730


def test_seed_time_to_datetime64_ns_moves_tensor_to_cpu() -> None:
    """Accelerator-backed output must reach CPU before NumPy conversion."""
    import torch

    class _AcceleratorTensor:
        def __init__(self) -> None:
            self.detached = False
            self.moved_to_cpu = False

        def detach(self) -> "_AcceleratorTensor":
            self.detached = True
            return self

        def cpu(self) -> "torch.Tensor":
            assert self.detached
            self.moved_to_cpu = True
            return torch.tensor([_NS_2014_07_23])

        def numpy(self) -> "np.ndarray":
            raise AssertionError("numpy() called before cpu()")

    time = _AcceleratorTensor()
    out = _seed_time_to_datetime64_ns(time)

    assert time.detached
    assert time.moved_to_cpu
    assert pd.Timestamp(out[0]) == pd.Timestamp("2014-07-23 00:00:00")


def test_build_eval_df_timestamp_is_ns_not_overflowed() -> None:
    """BC eval-df egress: an adapter-normalized ns seed time must surface in
    ``TIMESTAMP`` as the true year (2014), not the overflowed ~1730."""
    import torch

    stub = _EvalDFStub()
    outputs = _bc_outputs(n=2, with_time=False)
    outputs["time"] = torch.tensor(
        [_NS_2014_07_23, _NS_2014_07_23 + 86_400_000_000_000], dtype=torch.long)
    df = DiskgraphNestedTrainer._build_eval_df(stub, outputs)

    assert df is not None
    assert df["TIMESTAMP"].iloc[0] == pd.Timestamp("2014-07-23 00:00:00")
    assert df["TIMESTAMP"].iloc[1] == pd.Timestamp("2014-07-24 00:00:00")


def test_predict_bc_timestamp_is_ns_not_overflowed(tmp_path) -> None:
    """BC predict egress: ns seed time surfaces as the true year (2014)."""
    import torch

    class _PLModel:
        eval_top_k = None
        loss = object()

    class _PLTrainer:
        callbacks: list[Any] = []

        def predict(self, **_: Any) -> list[dict[str, Any]]:
            out = _bc_outputs(n=2, with_time=False)
            out["time"] = torch.tensor(
                [_NS_2014_07_23, _NS_2014_07_23 + 86_400_000_000_000],
                dtype=torch.long)
            return [out]

    datamodule = SimpleNamespace(
        _seed_node_type="customers",
        time_col_dict={"customers": "created_at"},
        predict_dataloader=lambda: [object()],
        **_worker_shutdown_stub(),
    )
    stub = SimpleNamespace(
        task=SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION),
        datamodule=datamodule,
        pl_model=_PLModel(),
        pl_trainer=_PLTrainer(),
        best_ckpt="/tmp/ckpt",
        target_encoder=_bc_target_encoder(),
        init=lambda **_: None,
        _preprocess_top_k=lambda top_k: top_k,
    )
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("CUSTOMER_ID", )))

    out = DiskgraphNestedTrainer.predict(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
    )

    assert out.df["created_at"].iloc[0] == pd.Timestamp("2014-07-23 00:00:00")
    assert out.df["created_at"].iloc[1] == pd.Timestamp("2014-07-24 00:00:00")


def test_predict_lp_timestamp_is_ns_not_overflowed(tmp_path) -> None:
    """LP predict egress (the originally-reported artifact): ns seed time
    surfaces as the true year (2014), repeated per RHS candidate."""
    import torch

    step_outputs = [{
        "pkey":
        torch.tensor([10, 11]),
        "target":
        torch.tensor([[0, 1], [2, 0]]),
        "score":
        torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
        "time":
        torch.tensor([_NS_2014_07_23, _NS_2014_07_23], dtype=torch.long),
    }]
    model = _TwoTowerStubModel(step_outputs)
    stub = _lp_stub(model, rhs_pkeys=["item_a", "item_b", "item_c"])
    stub.pl_trainer.predict = lambda **_: step_outputs
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=("USER_ID", "ITEM_ID")))

    out = DiskgraphNestedTrainer._predict_lp(
        stub,  # type: ignore[arg-type]
        output_dir=str(tmp_path / "out"),
        task=task,
    )

    assert (out.df["created_at"] == pd.Timestamp("2014-07-23 00:00:00")).all()


class _MetricListTrainer:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def validate(self, **_: Any) -> list[dict[str, Any]]:
        return self.results

    def test(self, **_: Any) -> list[dict[str, Any]]:
        return self.results


class _EvaluationResultTrainer(DiskgraphNestedTrainer):
    @property
    def best_ckpt(self) -> str:
        return str(self._test_log_dir / 'best.ckpt')

    @property
    def log_dir(self) -> str:
        return str(self._test_log_dir)


def _evaluation_result_trainer(
    tmp_path: Any,
    *,
    results: list[dict[str, Any]],
    task_type: TaskType,
    test_rows: int,
) -> _EvaluationResultTrainer:
    trainer = object.__new__(_EvaluationResultTrainer)
    trainer._prediction_mode = False
    trainer._test_log_dir = tmp_path
    trainer.pl_model = SimpleNamespace(eval_top_k=7)
    trainer.pl_trainer = _MetricListTrainer(results)
    trainer.datamodule = SimpleNamespace(task_manifest=SimpleNamespace(
        stage_counts={'test': test_rows}))
    trainer.task = SimpleNamespace(task_type=task_type)
    return trainer


def test_empty_link_test_returns_empty_metrics(tmp_path) -> None:
    trainer = _evaluation_result_trainer(
        tmp_path,
        results=[],
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        test_rows=0,
    )

    result = trainer._eval(TrainingStage.TEST)

    assert result.metrics == {}
    assert result.df is None
    assert result.embedding is None
    assert trainer.pl_model.outputs is None


@pytest.mark.parametrize(
    ('task_type', 'test_rows'),
    [
        (TaskType.REGRESSION, 0),
        (TaskType.STATIC_LINK_PREDICTION, 1),
    ],
)
def test_empty_metric_result_requires_an_empty_link_test(
    tmp_path: Any,
    task_type: TaskType,
    test_rows: int,
) -> None:
    trainer = _evaluation_result_trainer(
        tmp_path,
        results=[],
        task_type=task_type,
        test_rows=test_rows,
    )

    with pytest.raises(RuntimeError, match='evaluation returned no metric'):
        trainer._eval(TrainingStage.TEST)


def test_empty_link_validation_result_is_rejected(tmp_path) -> None:
    trainer = _evaluation_result_trainer(
        tmp_path,
        results=[],
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        test_rows=0,
    )

    with pytest.raises(RuntimeError, match='stage=val'):
        trainer._eval(TrainingStage.VAL)


def test_empty_link_test_requires_an_explicit_zero_count(tmp_path) -> None:
    trainer = _evaluation_result_trainer(
        tmp_path,
        results=[],
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        test_rows=0,
    )
    trainer.datamodule.task_manifest.stage_counts = {}

    with pytest.raises(RuntimeError, match='evaluation returned no metric'):
        trainer._eval(TrainingStage.TEST)


def test_nonempty_link_metrics_are_unchanged(tmp_path) -> None:
    metrics = {'test_recall_at_10': 0.75}
    trainer = _evaluation_result_trainer(
        tmp_path,
        results=[metrics],
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        test_rows=4,
    )

    result = trainer._eval(TrainingStage.TEST)

    assert result.metrics is metrics
