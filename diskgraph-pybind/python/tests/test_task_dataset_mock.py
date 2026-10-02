import json
from types import SimpleNamespace

import pytest

import diskgraph
from diskgraph import (
    DiskgraphConfigError,
    FeatureSchemaInputs,
    GraphRuntimeMetadataInputs,
    SplitInfoInputs,
    TaskDatasetMetadataInputs,
    TrainTableColumnMappingInputs,
    build_node_prediction_task_dataset_bundle,
    RuntimeIndex,
    task_stage_size,
)


def make_runtime_batch_iterator(dataset_ref, **kwargs):
    """One-shot open-and-iterate, for tests that use an index exactly once.

    Deliberately NOT part of the public API. Production callers hold a
    :class:`diskgraph.RuntimeIndex` and call ``iter()`` per epoch -- that
    separation is the whole point, since opening the index is ~1.6 s and
    choosing rows is microseconds. These tests genuinely do open once and
    iterate once, so they keep a local convenience instead of repeating the
    two-step at every call site.
    """
    from diskgraph import RuntimeIndex

    index = RuntimeIndex(
        dataset_ref,
        rnd_seed=kwargs.get("rnd_seed"),
        num_threads=kwargs.pop("num_threads", None),
    )
    return index.iter(**kwargs)



def _metadata(**kwargs):
    defaults = dict(
        task_type="binary_classification",
        is_temporal=False,
        seed_node_type="USER",
        train_table_name="tt_user",
        train_table_columns=TrainTableColumnMappingInputs(
            seed_id_column="ENTITY",
            target_column="TARGET",
            time_column="TIMESTAMP",
            weight_column="WEIGHT",
        ),
    )
    defaults.update(kwargs)
    return TaskDatasetMetadataInputs(**defaults)


def _bundle(tmp_path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact_root = str(tmp_path / "task_dataset")
    args = dict(
        artifact_root=artifact_root,
        task_type="binary_classification",
        seed_node_ids=[10, 11, 12, 13, 14],
        targets=[1.0, 0.0, 1.0, 0.0, 1.0],
        split_bytes=bytes([0, 1, 0, 2, 255]),
        metadata=_metadata(),
        split_info=SplitInfoInputs(
            split_kind="manual",
            normalized_split_config={"kind": "manual"},
        ),
        stats={},
        feature_schema=FeatureSchemaInputs(
            node_types={"USER": {"mapper_key": "USER"}}
        ),
        graph_runtime_metadata=GraphRuntimeMetadataInputs(
            graph_index_id="graph-1",
            node_types=["USER"],
        ),
    )
    args.update(kwargs)
    return build_node_prediction_task_dataset_bundle(**args)


def test_bundle_writes_stage_table_refs_and_counts(tmp_path):
    result = _bundle(tmp_path)
    with open(result["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["ingested_stage_tables"] == {
        "train": {
            "table_name": "tt_user__train",
            "storage_kind": "mock_parquet",
            "path": "ingested_stage_tables/tt_user__train.parquet",
        },
        "val": {
            "table_name": "tt_user__val",
            "storage_kind": "mock_parquet",
            "path": "ingested_stage_tables/tt_user__val.parquet",
        },
        "test": {
            "table_name": "tt_user__test",
            "storage_kind": "mock_parquet",
            "path": "ingested_stage_tables/tt_user__test.parquet",
        },
    }

    with open(tmp_path / "task_dataset" / "split_info.json") as f:
        split_info = json.load(f)
    assert split_info["stage_counts"] == {
        "train": 2,
        "val": 1,
        "test": 1,
        "unused": 1,
    }


def test_metadata_rejects_missing_required_time_and_weight_inputs(tmp_path):
    with pytest.raises(DiskgraphConfigError, match="seed_times is required"):
        _bundle(tmp_path / "temporal", metadata=_metadata(is_temporal=True))

    with pytest.raises(DiskgraphConfigError, match="weights is required"):
        _bundle(tmp_path / "weighted", metadata=_metadata(has_weight=True))

    with pytest.raises(DiskgraphConfigError, match="weight_column"):
        _bundle(
            tmp_path / "weight_column",
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],
            metadata=_metadata(
                train_table_columns=TrainTableColumnMappingInputs(
                    seed_id_column="ENTITY",
                    target_column="TARGET",
                    time_column="TIMESTAMP",
                    weight_column=None,
                )
            ),
        )


def test_supplied_time_and_weight_are_reflected_in_metadata(tmp_path):
    result = _bundle(
        tmp_path,
        seed_times=[100, 101, 102, 103, 104],
        weights=[1.0, 0.5, 1.0, 0.5, 1.0],
    )
    with open(tmp_path / "task_dataset" / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["schema_version"] == "v1"
    assert metadata["has_time"] is True
    assert metadata["has_weight"] is True
    assert metadata["metadata"]["implementation_kind"] == "python_mock"
    assert result["manifest_path"]


def test_stage_size_surface_rejects_old_sidecar_manifest(tmp_path):
    result = _bundle(
        tmp_path,
        split_bytes=bytes([0, 0, 0, 0, 0]),
    )
    with pytest.raises(DiskgraphConfigError, match="runtime manifest kind"):
        task_stage_size(result["manifest_path"], "train")


def test_stage_table_refs_preserve_native_index_path():
    from diskgraph.runtime_batch_iterator import _stage_tables

    tables = _stage_tables(
        {
            "stages": {
                "train": {
                    "table_name": "train_seeds",
                    "index_path": "/idx/train_tables/tt1",
                    "row_count": 7,
                }
            }
        }
    )
    assert tables == {
        "train": {
            "table_name": "train_seeds",
            "index_path": "/idx/train_tables/tt1",
        }
    }


def test_native_iterator_surface_validates_worker_inputs_before_opening_native_index(tmp_path):
    manifest_path = str(tmp_path / "not_read_for_input_validation.json")

    with pytest.raises(DiskgraphConfigError, match="worker_id 2 out of range"):
        make_runtime_batch_iterator(
            manifest_path,
            stage="train",
            sampling_plan={"layers": []},
            num_workers=2,
            worker_id=2,
            shuffle=False,
            rnd_seed=None,
            epoch=0,
            batch_size=1,
        )

    with pytest.raises(DiskgraphConfigError, match="rnd_seed is required"):
        make_runtime_batch_iterator(
            manifest_path,
            stage="train",
            sampling_plan={"layers": []},
            num_workers=1,
            worker_id=0,
            shuffle=True,
            rnd_seed=None,
            epoch=0,
            batch_size=1,
        )

    with pytest.raises(DiskgraphConfigError, match="epoch must be >= 0"):
        make_runtime_batch_iterator(
            manifest_path,
            stage="train",
            sampling_plan={"layers": []},
            num_workers=1,
            worker_id=0,
            shuffle=True,
            rnd_seed=123,
            epoch=-1,
            batch_size=1,
        )


def test_native_iterator_surface_forwards_epoch_to_native_iterator(tmp_path, monkeypatch):
    # The arg-forwarding contract is tested directly against the flat
    # `diskgraph_task_dataset` runtime manifest the factory consumes
    # (kumo writes it during a real ingest). The mock-bundle writer
    # emits a different shape (no `kind`, uses `ingested_stage_tables`);
    # exercising the wrapper through it would just trip the kind/shape
    # checks before we get to the iterator constructor.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": "/idx"},
        "stages": {
            "train": {
                "table_name": "tt_user__train",
                "index_path": "/idx/train_tables/tt_user",
                "row_count": 7,
            }
        },
    }))
    captured = SimpleNamespace(args=None, index_args=None)

    class FakeIterator:
        def num_samples(self):
            return 2

    class FakeRuntimeIndex:
        def __init__(self, *args):
            captured.index_args = args

        def iter(self, *args):
            captured.args = args
            return FakeIterator()

    monkeypatch.setattr(diskgraph._c, "RuntimeIndex", FakeRuntimeIndex)

    iterator, num_samples = make_runtime_batch_iterator(
        str(manifest_path),
        stage="train",
        sampling_plan={"layers": []},
        num_workers=2,
        worker_id=1,
        shuffle=True,
        rnd_seed=123,
        epoch=7,
        batch_size=4,
        drop_last=True,
    )

    # RuntimeIndex ctor args: 0=parent_root, 1=tt_ref, 2=rnd_seed,
    #   3=num_threads.
    # RuntimeIndex.iter args: 0=stage_table, 1=plan_json, 2=num_workers,
    #   3=worker_id, 4=shuffle, 5=rnd_seed, 6=epoch, 7=batch_size,
    #   8=drop_last
    assert isinstance(iterator, FakeIterator)
    assert num_samples == 2
    assert captured.index_args[0] == "/idx"
    assert captured.index_args[1] == "train_tables/tt_user"
    assert captured.args[0] == "tt_user__train"
    assert captured.args[4:8] == (True, 123, 7, 4)
    assert captured.args[8] is True


def test_native_iterator_surface_forwards_selected_stage_node_ids(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": "/idx"},
        "stages": {
            "train": {
                "table_name": "tt_user__train",
                "index_path": "/idx/train_tables/tt_user",
                "row_count": 7,
            }
        },
    }))
    captured = SimpleNamespace(args=None, index_args=None)

    class FakeIterator:
        def num_samples(self):
            return len(captured.args[-4])

    class FakeRuntimeIndex:
        def __init__(self, *args):
            captured.index_args = args

        def iter(self, *args):
            captured.args = args
            return FakeIterator()

    monkeypatch.setattr(diskgraph._c, "RuntimeIndex", FakeRuntimeIndex)

    selected_stage_node_ids = [4, 1, 3]
    _, num_samples = make_runtime_batch_iterator(
        str(manifest_path),
        stage="train",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
        selected_stage_node_ids=selected_stage_node_ids,
    )

    assert num_samples == 3
    assert captured.args[-4] is selected_stage_node_ids
    assert captured.args[-3] is True
    assert captured.args[-2] is None
    assert captured.args[-1] is False


def test_native_iterator_surface_resolves_weighted_seed_sampling(tmp_path, monkeypatch):
    root = tmp_path / "idx" / "train_tables" / "tt1"
    (root / "train_seeds").mkdir(parents=True)
    false_node_ids = root / "train_seeds" / "false_node_ids.bin"
    true_node_ids = root / "train_seeds" / "true_node_ids.bin"
    false_node_ids.write_bytes(b"")
    true_node_ids.write_bytes(b"")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": str(tmp_path / "idx")},
        "stages": {
            "train": {
                "table_name": "train_seeds",
                "index_path": str(root),
                "row_count": 7,
                "binary_label_index": {
                    "false_node_ids": "train_seeds/false_node_ids.bin",
                    "false_count": 5,
                    "true_node_ids": "train_seeds/true_node_ids.bin",
                    "true_count": 2,
                },
            }
        },
    }))
    captured = SimpleNamespace(args=None, index_args=None)

    class FakeIterator:
        def num_samples(self):
            return 6

    class FakeRuntimeIndex:
        def __init__(self, *args):
            captured.index_args = args

        def iter(self, *args):
            captured.args = args
            return FakeIterator()

    monkeypatch.setattr(diskgraph._c, "RuntimeIndex", FakeRuntimeIndex)

    _, num_samples = make_runtime_batch_iterator(
        str(manifest_path),
        stage="train",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        weighted_seed_sampling={
            "false_weight": 1.0,
            "true_weight": 3.0,
            "num_samples": 6,
        },
    )

    assert num_samples == 6
    config = json.loads(captured.args[-5])
    assert config["buckets"] == [
        {
            "name": "false",
            "node_ids_path": str(false_node_ids),
            "count": 5,
            "mass": 5.0,
        },
        {
            "name": "true",
            "node_ids_path": str(true_node_ids),
            "count": 2,
            "mass": 6.0,
        },
    ]
    assert config["num_samples"] == 6


def test_native_iterator_surface_resolves_row_weight_sampling(tmp_path, monkeypatch):
    root = tmp_path / "idx" / "train_tables" / "tt1"
    (root / "train_seeds").mkdir(parents=True)
    ids = root / "train_seeds" / "sample_weight_node_ids.bin"
    cdf = root / "train_seeds" / "sample_weight_cdf_f64.bin"
    ids.write_bytes(b"")
    cdf.write_bytes(b"")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": str(tmp_path / "idx")},
        "stages": {
            "train": {
                "table_name": "train_seeds",
                "index_path": str(root),
                "row_count": 7,
                "sample_weight_index": {
                    "all": {
                        "node_ids": "train_seeds/sample_weight_node_ids.bin",
                        "cdf": "train_seeds/sample_weight_cdf_f64.bin",
                        "count": 7,
                        "total_weight": 11.5,
                    },
                },
            }
        },
    }))
    captured = SimpleNamespace(args=None, index_args=None)

    class FakeIterator:
        def num_samples(self):
            return 12

    class FakeRuntimeIndex:
        def __init__(self, *args):
            captured.index_args = args

        def iter(self, *args):
            captured.args = args
            return FakeIterator()

    monkeypatch.setattr(diskgraph._c, "RuntimeIndex", FakeRuntimeIndex)

    _, num_samples = make_runtime_batch_iterator(
        str(manifest_path),
        stage="train",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=3,
        weighted_seed_sampling={
            "use_sample_weights": True,
            "num_samples": 12,
        },
    )

    assert num_samples == 12
    config = json.loads(captured.args[-5])
    assert config["buckets"] == [{
        "name": "all",
        "node_ids_path": str(ids),
        "count": 7,
        "mass": 11.5,
        "cdf_path": str(cdf),
        "total_weight": 11.5,
    }]
    assert config["num_samples"] == 12


def test_native_iterator_surface_disables_seed_timestamps_for_static_task(
        tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "task": {
            "task_type": "multiclass_classification",
            "is_temporal": False,
        },
        "graph": {"index_path": "/idx"},
        "table": {
            "columns": {
                "timestamp": "TIMESTAMP",
            },
        },
        "stages": {
            "train": {
                "table_name": "tt_user__train",
                "index_path": "/idx/train_tables/tt_user",
                "row_count": 7,
            }
        },
    }))
    captured = SimpleNamespace(args=None, index_args=None)

    class FakeIterator:
        def num_samples(self):
            return 7

    class FakeRuntimeIndex:
        def __init__(self, *args):
            captured.index_args = args

        def iter(self, *args):
            captured.args = args
            return FakeIterator()

    monkeypatch.setattr(diskgraph._c, "RuntimeIndex", FakeRuntimeIndex)

    make_runtime_batch_iterator(
        str(manifest_path),
        stage="train",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=2,
    )

    assert captured.args[-3] is False


def test_native_iterator_surface_forwards_pred_table_extension(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": "/idx"},
        "stages": {
            "predict": {
                "table_name": "predict",
                "index_path": "/idx/pred_tables/job1",
                "row_count": 7,
            }
        },
    }))
    captured = SimpleNamespace(args=None, index_args=None)

    class FakeIterator:
        def num_samples(self):
            return 7

    class FakeRuntimeIndex:
        def __init__(self, *args):
            captured.index_args = args

        def iter(self, *args):
            captured.args = args
            return FakeIterator()

    monkeypatch.setattr(diskgraph._c, "RuntimeIndex", FakeRuntimeIndex)

    _, num_samples = make_runtime_batch_iterator(
        str(manifest_path),
        stage="predict",
        sampling_plan={"layers": []},
        num_workers=1,
        worker_id=0,
        shuffle=False,
        rnd_seed=None,
        epoch=0,
        batch_size=4,
    )

    assert num_samples == 7
    assert captured.index_args[0] == "/idx"
    assert captured.index_args[1] == "pred_tables/job1"
    assert captured.args[0] == "predict"


# Behavioral epoch determinism (PR 268 placeholder) — fully covered now by
# `test_iterator_shuffle_is_deterministic_per_epoch` in
# `test_runtime_batch_iterator_e2e.py`, which exercises the live
# `make_runtime_batch_iterator` against a real DG-ingested stage table and
# asserts both (1) (rnd_seed, epoch) replay determinism and (2) different
# epochs at the same rnd_seed produce different orderings. Placeholder
# removed — was waiting on PR #251, which we've since superseded.


def test_task_dataset_exports_are_minimal():
    assert diskgraph.TASK_DATASET_SCHEMA_VERSION == "v1"
    assert diskgraph.TASK_DATASET_IMPLEMENTATION_KIND == "python_mock"
    assert callable(diskgraph.task_stage_size)
    assert callable(diskgraph.RuntimeIndex)
    assert diskgraph.RuntimeBatchIterator is not None
    with pytest.raises(AttributeError):
        getattr(diskgraph, "DiskgraphTaskDataset")
    with pytest.raises(AttributeError):
        getattr(diskgraph, "open_task_dataset")


def test_stage_size_surface_reads_flat_kumo_runtime_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": "/idx"},
        "table": {
            "seed_entity_table": "USER",
            "columns": {
                "primary_entity": "ENTITY1",
                "primary_target": "TARGET1",
                "timestamp": "TIMESTAMP",
                "weight": None,
            },
        },
        "stages": {
            "train": {
                "table_name": "train_seeds",
                "index_path": "/idx/train_tables/tt1",
                "row_count": 7,
            },
            "val": {
                "table_name": "val_seeds",
                "index_path": "/idx/train_tables/tt1",
                "row_count": 2,
            },
        },
    }))

    assert task_stage_size(str(manifest_path), "train") == 7
    assert task_stage_size(str(manifest_path), "val") == 2
