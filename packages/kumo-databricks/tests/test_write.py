# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one implementation of loading a Delta table, driven through fakes."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest
from kumo_databricks import write

STAGE = "/Volumes/main/features/_kumo_staging/artifact-export/job-1"


class Warehouse:
    """Records statements and volume batches, and can fail either."""
    def __init__(self, fail=None):
        self.statements: list[str] = []
        self.batches: list[tuple] = []
        self.fail = fail or {}

    def execute(self, statement: str):
        self.statements.append(statement)
        for fragment, error in self.fail.items():
            if statement.startswith(fragment):
                raise error

    def run(self, commands, *, worker_count, staging_allowed_local_path,
            ignore_missing=False):
        self.batches.append((list(commands), worker_count,
                             staging_allowed_local_path, ignore_missing))
        if "volume" in self.fail and commands[0].startswith("PUT"):
            raise self.fail["volume"]

    @staticmethod
    def workers(files, commands):
        return 1

    def one(self, fragment):
        found = [s for s in self.statements if fragment in s]
        assert len(found) == 1, f"expected one {fragment!r}, got {found}"
        return found[0]


def load(warehouse, path, *, replace=False, files=None, staged=None):
    return write.load(execute=warehouse.execute, run=warehouse.run,
                      workers=warehouse.workers, catalog="main",
                      schema="features", table="churn", artifact_path=path,
                      staged_files=staged, replace=replace,
                      resolve_local=lambda _: list(files or []))


@pytest.fixture
def artifact(tmp_path):
    frame = pd.DataFrame({"ENTITY": ["a"], "SCORE": [1.0]})
    path = tmp_path / "part-00000000.parquet"
    write.to_parquet(frame, str(path))
    return path


# Addressing ###############################################################


@pytest.mark.parametrize("path", [
    "/Volumes/main/features/_kumo_staging/artifact-export/job",
    "dbfs:/Volumes/main/features/_kumo_staging/artifact-export/job",
])
def test_both_volume_spellings_name_the_same_stage(path):
    assert write.within_stage(path, "main", "features") == (
        "/Volumes/main/features/_kumo_staging/artifact-export/job")


@pytest.mark.parametrize("path", [
    "s3://example-bucket/predictions/",
    "abfss://a@b.dfs.core.windows.net/predictions/",
    "gs://example-bucket/predictions/",
    "/Volumes/main/features/external/predictions/",
    "/Volumes/other/features/_kumo_staging/artifact-export/job/",
    "/Volumes/main/features/_kumo_staging/elsewhere/job",
    "/Volumes/main/features",
])
def test_a_stage_outside_the_managed_volume_is_refused(path):
    with pytest.raises(ValueError):
        write.within_stage(path, "main", "features")


def test_a_stage_reaching_upwards_is_refused():
    with pytest.raises(ValueError, match="unsafe path component"):
        write.within_stage(f"{STAGE}/../../elsewhere", "main", "features")


def test_a_local_dataset_is_never_mistaken_for_a_volume(tmp_path):
    assert write.parse_volume_uri(str(tmp_path)) is None
    write.require_local_path(str(tmp_path))


def test_a_remote_artifact_is_refused_before_anything_runs():
    warehouse = Warehouse()
    with pytest.raises(ValueError, match="only local Parquet files"):
        load(warehouse, "s3://example-bucket/predictions/")
    assert warehouse.statements == []


@pytest.mark.parametrize(("staged", "message"), [
    (None, "exact uploaded-file manifest"),
    ([f"{STAGE}-other/part.parquet"], "outside its declared managed-volume"),
    ([f"{STAGE}/part.csv"], "not Parquet"),
    ([f"{STAGE}/part.parquet"] * 2, "duplicate files"),
])
def test_a_producer_manifest_fails_closed(staged, message):
    warehouse = Warehouse()
    with pytest.raises(ValueError, match=message):
        load(warehouse, STAGE, staged=staged)
    assert warehouse.statements == []


# Loading ##################################################################


def test_a_local_artifact_is_uploaded_then_copied(artifact):
    warehouse = Warehouse()
    load(warehouse, str(artifact), files=[artifact])

    put, _, allowed, _ = warehouse.batches[0]
    assert put == [f"PUT '{artifact}' INTO '{_staged(warehouse)}' OVERWRITE"]
    assert allowed == str(artifact.parent)
    assert f"FROM '{_staged(warehouse).rsplit('/', 1)[0]}'" in warehouse.one(
        "COPY INTO")


def test_a_producer_stage_is_copied_without_a_second_upload():
    warehouse = Warehouse()
    load(warehouse, STAGE, staged=[f"{STAGE}/worker-00.parquet"])

    assert f"FROM '{STAGE}'" in warehouse.one("COPY INTO")
    assert all(not batch[0][0].startswith("PUT") for batch in warehouse.batches)


def test_overwrite_replaces_the_destination(artifact):
    warehouse = Warehouse()
    load(warehouse, str(artifact), files=[artifact], replace=True)

    assert warehouse.one("CREATE OR REPLACE TABLE").startswith(
        "CREATE OR REPLACE TABLE `features`.`churn` ")
    assert not [s for s in warehouse.statements if s.startswith("INSERT")]


def test_append_adds_by_column_name_to_a_destination_it_may_create(artifact):
    warehouse = Warehouse()
    load(warehouse, str(artifact), files=[artifact], replace=False)

    assert not [
        s for s in warehouse.statements if s.startswith("CREATE OR REPLACE")
    ]
    assert "WHERE 1 = 0" in warehouse.one("CREATE TABLE IF NOT EXISTS "
                                          "`features`.`churn`")
    assert warehouse.one("INSERT INTO").startswith(
        "INSERT INTO `features`.`churn` BY NAME")


def test_the_destination_is_untouched_by_a_failed_copy(artifact):
    warehouse = Warehouse(fail={"COPY INTO": RuntimeError("copy failed")})

    with pytest.raises(RuntimeError, match="copy failed"):
        load(warehouse, str(artifact), files=[artifact], replace=True)

    assert not [
        s for s in warehouse.statements if s.startswith("CREATE OR REPLACE")
    ]


# Lifecycle ################################################################


def test_the_staging_table_and_the_staged_file_are_cleaned_up(artifact):
    warehouse = Warehouse()
    load(warehouse, str(artifact), files=[artifact])

    assert warehouse.statements[-1].startswith(
        "DROP TABLE IF EXISTS `features`.`churn__kumo_staging_")
    removals, _, _, ignore_missing = warehouse.batches[-1]
    assert removals == [f"REMOVE '{_staged(warehouse)}'"]
    assert ignore_missing, "a file already gone is not a failure"


def test_a_local_stage_and_staging_table_survive_only_until_a_failed_copy(
        artifact):
    r"""Unlike the producer stage in :func:`test_a_producer_stage_survives_a_
    failed_load`, a local upload's stage is this process's own -- a retry
    starts it over rather than resuming it -- so it is cleaned up on a
    ``COPY INTO`` failure exactly as it would be on success."""
    warehouse = Warehouse(fail={"COPY INTO": RuntimeError("copy failed")})

    with pytest.raises(RuntimeError, match="copy failed"):
        load(warehouse, str(artifact), files=[artifact], replace=True)

    assert any(
        statement.startswith(
            "DROP TABLE IF EXISTS `features`.`churn__kumo_staging_")
        for statement in warehouse.statements)
    assert warehouse.batches[-1][0] == [f"REMOVE '{_staged(warehouse)}'"]


def test_staged_files_are_removed_newest_first(tmp_path):
    r"""A cleanup that ran forwards would delete a directory's contents out
    from under the files still in it on stores that keep implicit prefixes."""
    frame = pd.DataFrame({"ENTITY": ["a"]})
    files = []
    for index in range(3):
        path = tmp_path / f"part-{index}.parquet"
        write.to_parquet(frame, str(path))
        files.append(path)
    warehouse = Warehouse()
    load(warehouse, str(tmp_path), files=files)

    staged = [
        command.partition(" INTO ")[2].removesuffix(" OVERWRITE").strip("'")
        for command in warehouse.batches[0][0]
    ]
    assert len(staged) == 3
    assert warehouse.batches[-1][0] == [
        f"REMOVE '{path}'" for path in reversed(staged)
    ]


def test_a_producer_stage_survives_a_failed_load():
    r"""Its files are the producer's, and a retry has to read them again."""
    warehouse = Warehouse(fail={"COPY INTO": RuntimeError("copy failed")})

    with pytest.raises(RuntimeError):
        load(warehouse, STAGE, staged=[f"{STAGE}/worker-00.parquet"])

    assert warehouse.batches == [], "nothing of the producer's was removed"


def test_a_partial_upload_removes_what_it_managed_to_stage(artifact):
    warehouse = Warehouse(fail={"volume": RuntimeError("upload failed")})

    with pytest.raises(RuntimeError, match="upload failed"):
        load(warehouse, str(artifact), files=[artifact])

    assert warehouse.batches[-1][0] == [f"REMOVE '{_staged(warehouse)}'"]


def test_a_load_provisions_the_staging_volume_it_writes_through(artifact):
    warehouse = Warehouse()
    load(warehouse, str(artifact), files=[artifact])

    assert warehouse.statements[0] == "describe schema `features`"
    assert warehouse.statements[1] == (
        "DESCRIBE VOLUME `main`.`features`.`_kumo_staging`")


# Parquet ##################################################################


def test_timestamps_are_written_at_a_resolution_copy_into_accepts(tmp_path):
    frame = pd.DataFrame(
        {"WHEN": pd.to_datetime(["2026-09-02 12:00:00.123456"])})
    path = tmp_path / "part.parquet"
    write.to_parquet(frame, str(path))

    written = pq.read_schema(path)
    assert str(written.field("WHEN").type) == "timestamp[us]"
    assert written.names == ["WHEN"], "the index is not a column"


def _staged(warehouse):
    """The one volume path this load put a file at."""
    put = warehouse.batches[0][0][0]
    return put.partition(" INTO ")[2].removesuffix(" OVERWRITE").strip("'")


def test_a_warehouse_value_error_is_not_restated_as_a_refusal(artifact):
    detail = "provider detail a caller must not restate"
    warehouse = Warehouse(fail={"DESCRIBE VOLUME": ValueError(detail)})

    with pytest.raises(ValueError) as raised:
        load(warehouse, str(artifact), files=[artifact])
    assert not isinstance(raised.value, write.ArtifactValidationError)


def test_sink_appends_files_stamped_with_extra_columns(artifact, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from kumo_databricks.sink import DatabricksSinkBackend
    from kumoapi.data_source import DatabricksConnectorResourceConfig
    from kumoapi.jobs import WriteMode

    sink = DatabricksSinkBackend()
    monkeypatch.setattr(sink, "_connected",
                        contextmanager(lambda *a, **k: (yield None)))
    loads = []

    def load(**kwargs):
        with kwargs["prepare_file"](str(artifact)) as staged:
            loads.append((kwargs["replace"], pq.read_table(staged).to_pydict()))

    monkeypatch.setattr(write, "load", load)
    connector = SimpleNamespace(id="dbx", config=DatabricksConnectorResourceConfig(
        catalog="main", schema_name="features", warehouse_id="w"))
    sink.write_files(connector, "features__kumo__churn", [str(artifact)],
                     WriteMode.APPEND, extra_columns={"PREDICTION_ID": "p1"})

    assert loads == [(False, {"ENTITY": ["a"], "SCORE": [1.0],
                              "PREDICTION_ID": ["p1"]})]
