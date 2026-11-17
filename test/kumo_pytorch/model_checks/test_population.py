"""Full populations retain every anchor, including tails and invalid targets."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from kumo.pytorch.api import Batch

from test.kumo_pytorch import population
from test.kumo_pytorch.population_anchors import project_population

from .test_aml_inputs import anchor_file
from .test_run_aml import CASES
from .test_run_aml import mapped as mapped  # noqa: PLC0414 - pytest fixture re-export


@pytest.mark.parametrize(
    "native,expected",
    [
        (True, {"train": 2, "valid": 1, "test": 1}),
        (False, {"train": 1, "valid": 1, "test": 1}),
    ],
)
def test_full_projection_preserves_splits_null_targets_and_v2(
    tmp_path, native, expected
):
    source = anchor_file(tmp_path)
    path = tmp_path / "projected.parquet"
    anchors, manifest = project_population(
        source, path, {"leading-0007": 42}, graph=not native, native=native
    )
    rows = pq.read_table(path).to_pylist()
    assert manifest["mode"] == "full" and manifest["counts"] == expected
    assert [r["sample_id"] for r in rows] == (
        ["excluded", "fit", "valid", "test"] if native else ["fit", "valid", "test"]
    )
    assert all(r["entity_id"] == 42 for r in rows)
    assert rows[0]["label"] is None
    assert all(
        r["as_of_time"] == r["prediction_time_ns"] // 10**9 - (not native) for r in rows
    )
    assert anchors.targets == ({} if native else {"label": "label"})


def test_full_projection_reads_tail_beyond_smoke_and_rejects_duplicate_ids(tmp_path):
    source = anchor_file(tmp_path)
    original = pq.read_table(source)
    # Put an additional train anchor after the first Arrow read batch.
    rows = original.to_pylist()
    repeated = [dict(rows[1], sample_id=f"train-{i}") for i in range(65536)]
    table = pa.Table.from_pylist(rows + repeated, schema=original.schema)
    pq.write_table(table, source)
    _, smoke = project_population(
        source,
        tmp_path / "smoke.parquet",
        {"leading-0007": 42},
        graph=True,
        native=False,
        limit=1,
    )
    _, full = project_population(
        source,
        tmp_path / "full.parquet",
        {"leading-0007": 42},
        graph=True,
        native=False,
    )
    assert smoke["counts"]["train"] == 1
    assert full["counts"]["train"] == 65537
    pq.write_table(pa.concat_tables([table, original.slice(1, 1)]), source)
    with pytest.raises(ValueError, match="duplicate"):
        project_population(
            source,
            tmp_path / "duplicate.parquet",
            {"leading-0007": 42},
            graph=True,
            native=False,
        )


def source_anchors(selections, selection, tmp_path):
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    groups = selections[selection["selector"]["anchor_version"]][
        "split" if native else "forecast_split"
    ]
    rows = [
        dict(row, sample_id=f"{row['sample_id']}-{i}")
        for group in groups.values()
        for row in group
        for i in range(3)
    ]
    table = pa.Table.from_pylist(rows)
    for clock in ("as_of_time", "graph_as_of_time"):
        table = table.set_column(
            table.schema.get_field_index(clock),
            clock,
            table[clock].cast(pa.timestamp("ns")),
        )
    path = tmp_path / "source-anchors.parquet"
    pq.write_table(table, path)
    return path


@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_raw_aml_population_completion(mapped, selection, family, mode, tmp_path):
    dataset, identities, selections = mapped
    path = source_anchors(selections, selection, tmp_path)
    result = population.run_case(
        dataset,
        identities,
        path,
        selection,
        family,
        directory=tmp_path / "case",
        capacity=5,
        mode=mode,
        smoke_anchors=1,
        batch_size=2,
    )
    assert result["status"] == "passed", result.get("traceback", result)
    assert result["full_population_verified"] == (mode == "full")
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    groups = selections[selection["selector"]["anchor_version"]][
        "split" if native else "forecast_split"
    ]
    for name, split in result["splits"].items():
        expected = len(groups[name]) * 3 if mode == "full" else 1
        assert split["delivered"] == split["expected"] == expected
        assert split["batches"] == (expected + 1) // 2
        assert split["status"] == "passed"


def delivery():
    rows = [
        {
            "sample_id": f"sample-{i}",
            "entity_id": i,
            "as_of_time": 123,
            "label": v,
            "prediction_time_ns": 124000000000,
            "registered_entity_id": f"account-{i}",
            "anchor_row_position": i,
            "split": "train",
        }
        for i, v in enumerate((None, 1, 0))
    ]
    batch = Batch(
        inputs={},
        targets={
            "label": {
                "values": torch.tensor([0.0, 1.0, 0.0]),
                "valid": torch.tensor([False, True, True]),
            }
        },
        metadata={
            "sample_ids": [r["sample_id"] for r in rows],
            "entity_ids": torch.tensor([0, 1, 2]),
            "as_of_time": torch.tensor([123] * 3),
            "is_padding": torch.tensor([False] * 3),
            "request_metadata": [
                {
                    k: row[k]
                    for k in (
                        "prediction_time_ns",
                        "registered_entity_id",
                        "anchor_row_position",
                        "split",
                    )
                }
                for row in rows
            ],
        },
    )
    return rows, batch


def test_delivery_checks_keep_null_targets_without_dropping_anchors():
    rows, batch = delivery()
    ids, eligible = population.check_batch(
        batch, iter(rows), native=False, family="combined", state={}
    )
    assert ids == [r["sample_id"] for r in rows] and eligible == 2


@pytest.mark.parametrize(
    "damage",
    ["duplicate", "reorder", "clock", "entity", "label", "validity", "padding"],
)
def test_delivery_corruption_is_a_failure(damage):
    rows, batch = delivery()
    if damage == "duplicate":
        batch.metadata["sample_ids"][1] = batch.metadata["sample_ids"][0]
    elif damage == "reorder":
        batch.metadata["sample_ids"].reverse()
    elif damage == "clock":
        batch.metadata["as_of_time"][0] += 1
    elif damage == "entity":
        batch.metadata["entity_ids"][0] += 1
    elif damage == "label":
        batch.targets["label"]["values"][1] = 0
    elif damage == "validity":
        batch.targets["label"]["valid"][0] = True
    else:
        batch.metadata["is_padding"][0] = True
    with pytest.raises(AssertionError):
        population.check_batch(
            batch, iter(rows), native=False, family="combined", state={}
        )


def test_full_mode_rejects_smoke_limits(tmp_path):
    with pytest.raises(SystemExit) as error:
        population.main(
            [
                "--mode",
                "full",
                "--smoke-anchors",
                "1",
                "--data-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_full_population_rejects_a_dropped_tail(mapped, tmp_path, monkeypatch):
    from itertools import islice

    from kumo.pytorch.api import Task

    dataset, identities, selections = mapped
    selection, family = next(
        (s, f)
        for s, f in CASES
        if f == "nuformer"
        and s["selector"]["workload_id"] == "account-involvement-next24h"
    )
    path = source_anchors(selections, selection, tmp_path)
    original = Task.batches

    def truncated(self, *args, **kwargs):
        return islice(original(self, *args, **kwargs), 1)

    monkeypatch.setattr(Task, "batches", truncated)
    result = population.run_case(
        dataset,
        identities,
        path,
        selection,
        family,
        directory=tmp_path / "dropped-tail",
        capacity=5,
        mode="full",
        batch_size=2,
    )
    assert result["status"] == "failed"
    assert "Iteration dropped anchors" in result["error"]
    assert result["full_population_verified"] is False
    assert result["splits"]["train"]["delivered"] == 2
    assert result["splits"]["train"]["status"] == "failed"
    assert result["splits"]["valid"]["status"] == "pending"
