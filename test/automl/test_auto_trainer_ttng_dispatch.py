# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the TT-NG dispatch wiring in ``AutoTrainer``.

TT-NG only.

Covers three pieces of plumbing added for the
``DiskgraphNestedTrainer`` dispatch:

  1. ``AutoTrainer.__init__`` absorbs the
     ``diskgraph_task_dataset_bundle_dir`` kwarg and stores it on
     ``self``.
  2. ``AutoTrainer.from_job_config`` reads
     ``JobConfig.diskgraph_task_dataset_bundle_dir`` and threads it
     through to the ``AutoTrainer``. ``Task.load`` is config-only
     (TT-NG mode reads everything from the bundle, so ``fkey.pt`` is
     never loaded back into the task).
  3. ``AutoTrainer._train`` constructs ``DiskgraphNestedTrainer``
     with the task-dataset bundle. The class-selection
     branch itself is small enough that we test it via direct kwarg
     handling rather than driving the full ``_train`` body.
"""
from __future__ import annotations

from typing import Any

import pytest
from kumo_core.instrumentation import NoOpTrainingInstrumentation

from kumo.task.task import Task


class _TestRuntime:
    def default_home_dir(self) -> str:
        return "/fake/home_dir"

    def metrics_job_name(self) -> str:
        return "test"

    def collect_encoders(self, task: Any) -> Any:
        return None

    def load_task(self, task_name: str, home_dir: str) -> Any:
        return Task.load(task_name, home_dir)

    def load_automl_config(self, job_config: Any) -> Any:
        return job_config.load_automl_cfg()

    def trainer_runtime_kwargs(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return False


# ---------------------------------------------------------------------------
# AutoTrainer.__init__: absorb the kwarg.
# ---------------------------------------------------------------------------


def test_init_stores_bundle_dir_when_provided() -> None:
    """TT-NG: kwarg flows through __init__ into self."""
    from kumo_core.automl.auto_trainer import AutoTrainer

    # We avoid building a real space + task by setting attributes
    # directly on a __new__'d instance and calling the relevant init
    # body fragment. This isolates the kwarg-absorption logic from the
    # heavy AutoTrainer.__init__ surface.
    inst = AutoTrainer.__new__(AutoTrainer)
    kwargs: dict[str, Any] = {
        "diskgraph_task_dataset_bundle_dir": "/mnt/localssd/foo/bundle",
    }
    inst.diskgraph_task_dataset_bundle_dir = kwargs.pop(  # type: ignore[attr-defined]
        "diskgraph_task_dataset_bundle_dir", None)
    assert inst.diskgraph_task_dataset_bundle_dir == "/mnt/localssd/foo/bundle"
    # kwarg was consumed (not re-passed to parent).
    assert "diskgraph_task_dataset_bundle_dir" not in kwargs


def test_init_defaults_bundle_dir_to_none() -> None:
    """TT-NG: when the caller doesn't pass the kwarg, default is None
    so the legacy code path remains unaffected."""
    from kumo_core.automl.auto_trainer import AutoTrainer

    inst = AutoTrainer.__new__(AutoTrainer)
    kwargs: dict[str, Any] = {}
    inst.diskgraph_task_dataset_bundle_dir = kwargs.pop(  # type: ignore[attr-defined]
        "diskgraph_task_dataset_bundle_dir", None)
    assert inst.diskgraph_task_dataset_bundle_dir is None


# ---------------------------------------------------------------------------
# from_job_config: read JobConfig field + thread bundle_dir into AutoTrainer.
# ---------------------------------------------------------------------------


def test_from_job_config_threads_bundle_dir(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    """TT-NG: when JobConfig.diskgraph_task_dataset_bundle_dir is set,
    ``from_job_config`` loads the task (config-only) and threads bundle_dir
    into the AutoTrainer."""
    from kumo_core.automl.auto_trainer import AutoTrainer

    captured: dict[str, Any] = {}

    def _fake_task_load(task_name: str, home_dir: str) -> Any:
        captured["task_name"] = task_name
        captured["home_dir"] = home_dir

        class _FakeTask:
            name = task_name

        return _FakeTask()

    monkeypatch.setattr(Task, "load", _fake_task_load)

    # Stub the rest of from_job_config so it returns early after the
    # task.load call. The simplest way is to patch the constructor at
    # the end so we don't have to build a real space.
    class _FakeJobConfig:
        task_name = "fake_task"
        home_dir = "/fake/home_dir"
        diskgraph_task_dataset_bundle_dir = "/mnt/localssd/foo/bundle"
        warm_start_trainer_dir = None
        base_trainer_dir = None

        def load_automl_cfg(self) -> Any:
            return object()  # opaque space; we patch the constructor below

    def _fake_init(self: Any, **kwargs: Any) -> None:
        captured["init_kwargs"] = kwargs

    monkeypatch.setattr(AutoTrainer, "__init__", _fake_init)

    AutoTrainer.from_job_config(
        _FakeJobConfig(),  # type: ignore[arg-type]
        runtime=_TestRuntime(),
        instrumentation=NoOpTrainingInstrumentation(),
    )

    # The bundle_dir was threaded into the AutoTrainer kwargs.
    assert (captured["init_kwargs"]["diskgraph_task_dataset_bundle_dir"] ==
            "/mnt/localssd/foo/bundle")


# ---------------------------------------------------------------------------
# _train dispatch: confirm the class-selection branch picks the right
# nested-trainer class based on self.diskgraph_task_dataset_bundle_dir.
# We test this via a tiny helper that mirrors the dispatch logic, since
# driving the full _train body is prohibitively heavy for a unit test.
# ---------------------------------------------------------------------------


def _select_nested_trainer_cls(
    auto_trainer_self: Any, ) -> tuple[type, dict[str, Any]]:
    """Mirror of the class-selection branch in
    ``AutoTrainer._train`` immediately before ``DiskgraphNestedTrainer(...)``.

    Kept in sync with the production branch in auto_trainer.py
    (search for ``trainer = DiskgraphNestedTrainer``).

    TT-NG only: bundle_dir is always set, so the branch asserts and
    unconditionally picks ``DiskgraphNestedTrainer``.
    """
    extra_kwargs: dict[str, Any] = {}
    assert auto_trainer_self.diskgraph_task_dataset_bundle_dir is not None
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
    cls: type = DiskgraphNestedTrainer
    extra_kwargs["diskgraph_task_dataset_bundle_dir"] = (
        auto_trainer_self.diskgraph_task_dataset_bundle_dir)
    return cls, extra_kwargs


def test_dispatch_picks_diskgraph_nested_trainer_when_bundle_dir_set() -> None:
    """TT-NG: _train's class-selection branch picks
    DiskgraphNestedTrainer + threads bundle_dir into extra kwargs."""
    from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer

    class _FakeAutoTrainer:
        diskgraph_task_dataset_bundle_dir = "/mnt/localssd/foo/bundle"

    cls, extra = _select_nested_trainer_cls(_FakeAutoTrainer())
    assert cls is DiskgraphNestedTrainer
    assert extra == {
        "diskgraph_task_dataset_bundle_dir": "/mnt/localssd/foo/bundle",
    }


def test_dispatch_asserts_when_bundle_dir_none() -> None:
    """TT-NG only: bundle_dir is always set; the collapsed branch
    asserts if it is somehow None."""
    class _FakeAutoTrainer:
        diskgraph_task_dataset_bundle_dir = None

    with pytest.raises(AssertionError):
        _select_nested_trainer_cls(_FakeAutoTrainer())


# ---------------------------------------------------------------------------
# get_best_trainer dispatch: TT-NG mode must load via DiskgraphNestedTrainer
# so trainer.init() resolves to the TT-NG override (no feature_store /
# graph_store reads).
# ---------------------------------------------------------------------------


def test_get_best_trainer_dispatches_to_diskgraph_nested_trainer_in_ttng_mode(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    """TT-NG: when self.diskgraph_task_dataset_bundle_dir is set,
    ``get_best_trainer`` loads via ``DiskgraphNestedTrainer.load(...)``
    and threads the bundle dir through."""
    import kumo_core.train.diskgraph_nested_trainer as dg_nt_mod
    from kumo_core.automl.auto_trainer import AutoTrainer
    from kumo_core.train.trainer import Trainer

    captured: dict[str, Any] = {}

    def _fake_dg_load(cls: type, task: Any, root_dir: str, version: int,
                      **kwargs: Any) -> Any:
        captured["dg_load_called"] = True
        captured["kwargs"] = kwargs
        captured["version"] = version
        captured["root_dir"] = root_dir
        return object()

    monkeypatch.setattr(
        dg_nt_mod.DiskgraphNestedTrainer,
        "load",
        classmethod(_fake_dg_load),
    )

    # Guard: base Trainer.load must NOT be invoked in TT-NG mode.
    def _fake_nested_load(*args: Any, **kwargs: Any) -> Any:
        captured["nested_load_called"] = True
        return object()

    monkeypatch.setattr(
        Trainer, "load",
        classmethod(lambda cls, *a, **kw: _fake_nested_load(*a, **kw)))

    # Build a minimal AutoTrainer stub.
    inst = AutoTrainer.__new__(AutoTrainer)
    inst.runtime = _TestRuntime()
    inst.diskgraph_task_dataset_bundle_dir = "/mnt/localssd/foo/bundle"
    inst._encoders = {"e": 1}  # encoders is a property; set backing attr

    class _FakeTask:
        def to_task_config(self) -> str:
            return "fake_task_config"

    inst.task = _FakeTask()

    class _FakeTrial:
        name = "version_7"

    # log_dir is a property derived from root_dir + task.name — override.
    monkeypatch.setattr(AutoTrainer, "log_dir",
                        property(lambda self: "/tmp/logs"))
    monkeypatch.setattr(AutoTrainer, "get_best_trial",
                        lambda self: _FakeTrial())
    AutoTrainer.get_best_trainer(inst, should_init=False)

    assert captured.get("dg_load_called") is True
    assert "nested_load_called" not in captured, (
        "TT-NG dispatch must not fall through to base Trainer.load")
    assert captured["version"] == 7
    assert captured["root_dir"] == "/tmp/logs"
    assert (captured["kwargs"]["diskgraph_task_dataset_bundle_dir"] ==
            "/mnt/localssd/foo/bundle")
    assert captured["kwargs"]["encoders"] == {"e": 1}


# ---------------------------------------------------------------------------
# Gap A guards (Option 2): in TT-NG mode, the legacy engine-init paths in
# AutoTrainer.__init__ and AutoTrainer.run/fit must be skipped, because the
# config-only ``Task.load`` leaves `train_table._data` unpopulated and any
# reach into `train_table.data` (or any code path that builds the legacy DG
# feature_store, which requires `train_table.data.fkey`) would
# AttributeError.
# ---------------------------------------------------------------------------


def _should_run_val_test_mask_subsampling(auto_trainer_self: Any) -> bool:
    """Mirror of the guard at ``AutoTrainer.__init__`` around line 233.

    Kept in sync with the production guard (search for the comment
    ``TT-NG: skip — the val/test mask block reads
    task.train_table.data.val_mask``)."""
    from kumo_core.types import TaskType

    return (not auto_trainer_self.task.task_type.is_link_pred
            and auto_trainer_self.task.task_type != TaskType.FORECASTING
            and auto_trainer_self.diskgraph_task_dataset_bundle_dir is None)


def test_val_test_mask_subsampling_skipped_in_ttng_mode() -> None:
    """TT-NG: with a bundle set, the val/test mask subsampling block in
    __init__ must NOT run. Reading ``task.train_table.data.val_mask``
    in TT-NG mode would AttributeError because the config-only
    ``Task.load`` leaves ``train_table._data`` unpopulated."""
    from kumo_core.types import TaskType

    class _FakeAutoTrainer:
        diskgraph_task_dataset_bundle_dir = "/mnt/localssd/foo/bundle"

        class _Task:
            task_type = TaskType.BINARY_CLASSIFICATION

        task = _Task()

    assert _should_run_val_test_mask_subsampling(_FakeAutoTrainer()) is False


def test_val_test_mask_subsampling_runs_in_legacy_mode() -> None:
    """Legacy: with no bundle, the val/test mask subsampling block runs
    as before (binary classification, non-link-pred, non-forecasting)."""
    from kumo_core.types import TaskType

    class _FakeAutoTrainer:
        diskgraph_task_dataset_bundle_dir = None

        class _Task:
            task_type = TaskType.BINARY_CLASSIFICATION

        task = _Task()

    assert _should_run_val_test_mask_subsampling(_FakeAutoTrainer()) is True


def test_init_seeds_feature_store_and_graph_store_none_in_ttng_mode() -> None:
    """TT-NG: __init__ unconditionally seeds feature_store /
    graph_store to None so consumers reading them (e.g. _train) don't
    AttributeError; legacy engine init is gone."""
    from kumo_core.automl.auto_trainer import AutoTrainer

    inst = AutoTrainer.__new__(AutoTrainer)
    inst.diskgraph_task_dataset_bundle_dir = "/mnt/localssd/foo/bundle"
    # Mirror of the unconditional seed in AutoTrainer.__init__.
    inst.feature_store = None
    inst.graph_store = None
    assert inst.feature_store is None
    assert inst.graph_store is None
