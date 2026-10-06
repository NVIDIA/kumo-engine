# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Snowflake reader does that no other reader does."""

import os
import re
from datetime import datetime, timezone

import pytest
import requests
from fakes import FakeConnection, FakeCursor, FakeSessions
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    RowCountMismatchError,
    SourceAccessDeniedError,
    TransientSourceError,
)
from kumo_core.sources.types import Attestation, ReadMode
from kumo_snowflake import SnowflakeSourceReader
from kumoapi.typing import Dtype
from snowflake.connector.errors import (
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    RefreshTokenError,
    ServiceUnavailableError,
)

DATABASE, SCHEMA, TABLE = "KUMO_TESTS", "HM_SAMPLED", "TRANSACTIONS"
STAGE = '"KUMO_TESTS"."HM_SAMPLED"."KUMO_INGEST"'
ROWS = 1586527
ALTERED = datetime(2025, 2, 13, 12, 2, 38, tzinfo=timezone.utc)
COLUMNS = {"CUSTOMER_ID": Dtype.string, "T_DAT": Dtype.time}
PARTS = ("one.parquet", "two.parquet")
SIZES = (12, 8)
BODIES = {"https://one": b"PAR1" + b"a" * 8, "https://two": b"PAR1" + b"b" * 4}
DECLARED = {"CUSTOMER_ID": "TEXT", "T_DAT": "TIMESTAMP_TZ"}
UNLOAD = (
    'COPY INTO \'@"KUMO_TESTS"."HM_SAMPLED"."KUMO_INGEST"/{prefix}\' '
    'FROM (SELECT "CUSTOMER_ID", '
    'CAST(CONVERT_TIMEZONE(\'UTC\', "T_DAT") AS TIMESTAMP_NTZ) AS "T_DAT" '
    'FROM "KUMO_TESTS"."HM_SAMPLED"."TRANSACTIONS") '
    'FILE_FORMAT=(TYPE=PARQUET) MAX_FILE_SIZE=134217728 HEADER=TRUE '
    'OVERWRITE=TRUE')


def prefix_in(statement):
    """The stage-relative prefix a stage statement names."""
    return statement.split(f"@{STAGE}/", 1)[1].split("'", 1)[0]


def paths_in(statement):
    """The paths in a ``FROM (VALUES ('a'), ('b'))`` list."""
    return re.findall(r"\('([^']*)'\)", statement)


def url_for(path):
    return "https://%s" % os.path.basename(path).split(".")[0]


class Stage:
    """The objects one stage holds, keyed by their stage-relative path."""
    def __init__(self):
        self.objects = {}

    def unload(self, prefix):
        self.objects.update({
            f"{prefix}{part}": size
            for part, size in zip(PARTS, SIZES)
        })

    def remove(self, prefix):
        for path in [path for path in self.objects if path.startswith(prefix)]:
            del self.objects[path]

    def listed(self, prefix):
        # Snowflake prefixes each path with the stage's own name, which the
        # reader strips.
        return [(f"kumo_ingest/{path}", size, "m", "d")
                for path, size in sorted(self.objects.items())
                if path.startswith(prefix)]


class StageCursor(FakeCursor):
    """A warehouse whose unload materialises into a :class:`Stage` and whose
    listing answers from it, because the prefix a read unloads to is named
    per attempt and so cannot be canned ahead of the statement.

    ``during`` runs once, with the unload landed and nothing downloaded yet,
    which is where a second read of the same table overlaps this one.
    """
    def __init__(self, responses=None, raises=None, declared=None, stage=None,
                 during=None):
        super().__init__(responses, raises, declared)
        self.stage = Stage() if stage is None else stage
        self.during = during
        self.prefixes = []

    def execute(self, sql, params=None):
        if "LIST " in sql:
            self.responses["LIST "] = self.stage.listed(prefix_in(sql))
        elif "GET_PRESIGNED_URL" in sql:
            self.responses["GET_PRESIGNED_URL"] = [(path, url_for(path))
                                                   for path in paths_in(sql)]
        elif "COPY INTO" in sql:
            self.prefixes.append(prefix_in(sql))
        result = super().execute(sql, params)
        if "COPY INTO" in sql:
            self._landed()
        elif "REMOVE" in sql:
            self.stage.remove(prefix_in(sql))
        return result

    def _landed(self):
        rows = self.fetchall()
        if rows and rows[0][0]:
            self.stage.unload(self.prefixes[-1])
        during, self.during = self.during, None
        if during is not None:
            during()


def warehouse(declared=None, kind=None, stage=None, during=None):
    responses = {
        "TABLE_NAME in": [(TABLE, )],
        "TABLE_TYPE": [("BASE TABLE", )],
        "LAST_ALTERED": [(TABLE, ALTERED)],
        "COPY INTO": [(ROWS, 90104519, 71000000)],
    }
    return (kind or StageCursor)(responses, declared=declared or DECLARED,
                                 stage=stage, during=during)


class StubbornStage(StageCursor):
    """Refuses only the REMOVE that follows the unload, which is the one a
    finished read has nothing left to gain from."""
    def execute(self, sql, params=None):
        if "REMOVE" in sql and self.matching("COPY INTO"):
            raise ProgrammingError(msg="stage is busy", errno=1,
                                   sqlstate="XX000")
        return super().execute(sql, params)


@pytest.fixture(autouse=True)
def http(monkeypatch):
    """A read mints its own session, so the stand-in is installed where the
    reader asks for one rather than handed to it."""
    sessions = FakeSessions(BODIES)
    monkeypatch.setattr(requests, "Session", sessions)
    return sessions


def build(cursor, **extra):
    return SnowflakeSourceReader(FakeConnection(cursor), database=DATABASE,
                                 schema=SCHEMA, **extra)


def staged(tmp_path, cursor=None, **kwargs):
    reader = build(cursor or warehouse())
    return reader.prepare(TABLE, str(tmp_path), columns=COLUMNS,
                          mode=ReadMode.STAGED, **kwargs)


def sequence(cursor, *markers):
    return [
        marker for statement in cursor.sql() for marker in markers
        if marker in statement
    ]


def test_snowflake_is_read_through_a_stage_and_never_streamed():
    assert build(warehouse()).read_modes(TABLE) == (ReadMode.STAGED, )


def test_reader_never_refolds_exact_catalog_names():
    reader = SnowflakeSourceReader(FakeConnection(warehouse()),
                                   database="kumo_tests", schema="hm_sampled",
                                   stage_name="kumo_ingest")
    assert (reader.database, reader.schema,
            reader.stage_name) == ("kumo_tests", "hm_sampled", "kumo_ingest")
    assert reader._stage == '"kumo_tests"."hm_sampled"."kumo_ingest"'


def test_an_unquoted_table_falls_back_to_its_catalogued_uppercase_name(
        tmp_path):
    cursor = warehouse()
    cursor.responses["TABLE_NAME in"] = [(TABLE, )]

    reader = build(cursor)
    reader.prepare(TABLE.lower(), str(tmp_path), columns=COLUMNS,
                   mode=ReadMode.STAGED)

    lookup = next(params for statement, params in cursor.statements
                  if "TABLE_NAME in" in statement)
    assert lookup == (SCHEMA, TABLE.lower(), TABLE)
    assert f'FROM "{DATABASE}"."{SCHEMA}"."{TABLE}"' in cursor.matching(
        "COPY INTO")[0]


def test_an_exact_lowercase_table_wins_over_its_folded_candidate(tmp_path):
    cursor = warehouse()
    cursor.responses["TABLE_NAME in"] = [(TABLE, ), (TABLE.lower(), )]

    reader = build(cursor)
    reader.prepare(TABLE.lower(), str(tmp_path), columns=COLUMNS,
                   mode=ReadMode.STAGED)

    assert (f'FROM "{DATABASE}"."{SCHEMA}"."{TABLE.lower()}"'
            in cursor.matching("COPY INTO")[0])


def test_the_unload_pushes_the_projection_down_to_the_stage(tmp_path):
    cursor = warehouse()
    staged(tmp_path, cursor)
    prefix, = cursor.prefixes
    assert cursor.matching("COPY INTO") == [UNLOAD.format(prefix=prefix)]


def test_a_legacy_date_dtype_still_discovers_and_casts_physical_ltz(tmp_path):
    cursor = warehouse(declared={**DECLARED, "T_DAT": "TIMESTAMP_LTZ"})
    reader = build(cursor)
    reader.prepare(TABLE, str(tmp_path), columns={"T_DAT": Dtype.date},
                   mode=ReadMode.STAGED)
    query, = cursor.matching("COPY INTO")
    assert ('CAST(CONVERT_TIMEZONE(\'UTC\', "T_DAT") AS TIMESTAMP_NTZ) '
            'AS "T_DAT"') in query


def test_the_unload_names_a_prefix_of_its_own_under_the_tables_directory(
        tmp_path):
    first, second = warehouse(), warehouse()
    staged(tmp_path / "first", first)
    staged(tmp_path / "second", second)
    prefix, = first.prefixes
    directory, attempt, tail = prefix.split("/")
    assert (directory, tail) == (TABLE, "")
    assert attempt.isalnum() and len(attempt) == 32
    assert second.prefixes != first.prefixes


def test_the_stage_is_created_then_cleared_before_the_unload_and_after_it(
        tmp_path):
    """A resize renames the unload's files, so a stale set survives beside them."""
    cursor = warehouse()
    staged(tmp_path, cursor)
    assert sequence(cursor, "CREATE TEMPORARY STAGE", "REMOVE",
                    "COPY INTO") == [
                        "CREATE TEMPORARY STAGE", "REMOVE", "COPY INTO",
                        "REMOVE"
                    ]


def test_the_count_the_unload_reported_is_carried_as_the_warehouses_own(
        tmp_path):
    table = staged(tmp_path)
    assert table.num_rows == ROWS
    assert table.rows_attested_by is Attestation.WAREHOUSE
    assert table.scanned_bytes == 90104519


def test_a_table_of_no_rows_reports_zero_rather_than_a_missing_unload(
        tmp_path, http):
    cursor = warehouse()
    cursor.responses["COPY INTO"] = [(0, 0, 0)]
    table = staged(tmp_path, cursor)
    assert (table.paths, table.num_rows) == ((), 0)
    assert table.rows_attested_by is Attestation.WAREHOUSE
    assert cursor.matching("LIST ") == []
    assert http.opened == []


def test_a_short_read_is_reported_before_anything_is_downloaded(tmp_path, http):
    cursor = warehouse()
    with pytest.raises(RowCountMismatchError) as failure:
        staged(tmp_path, cursor, expected_rows=ROWS + 1)
    assert (failure.value.expected, failure.value.actual) == (ROWS + 1, ROWS)
    assert http.opened == []
    assert len(cursor.matching("REMOVE")) == 2


def test_a_failed_download_still_clears_the_stage(tmp_path, http):
    cursor = warehouse()
    http.raises = OSError("connection reset")
    with pytest.raises(TransientSourceError):
        staged(tmp_path, cursor)
    assert len(cursor.matching("REMOVE")) == 2


def test_a_refused_download_never_quotes_the_presigned_url(tmp_path, http):
    """The url carries a signature that stays usable for an hour."""
    http.refuses = requests.exceptions.HTTPError(
        "403 Client Error: Forbidden for url: https://one?X-Amz-Signature=abc")
    with pytest.raises(PermanentSourceError) as failure:
        staged(tmp_path)
    assert "X-Amz-Signature" not in str(failure.value)
    assert "https://" not in str(failure.value)
    assert "HTTPError" in str(failure.value)
    assert failure.value.__cause__ is None


def test_a_driver_failure_is_reported_by_its_codes_not_its_text(tmp_path):
    """Driver messages carry the statement, its bindings and its response."""
    cursor = warehouse()
    cursor.raises["COPY INTO"] = ProgrammingError(
        msg="SQL compilation error at SELECT secret_token='hunter2'",
        errno=2043, sqlstate="42S02", sfqid="01b7-0000-abcd")
    with pytest.raises(PermanentSourceError) as failure:
        staged(tmp_path, cursor)
    reported = str(failure.value)
    assert "hunter2" not in reported
    assert "ProgrammingError errno=2043 sqlstate=42S02 query=01b7-0000-abcd" \
        in reported
    assert failure.value.table == TABLE
    assert "staged" in reported
    assert failure.value.__cause__ is None


def test_internal_stage_policy_denial_has_an_allowlisted_action(tmp_path):
    cursor = warehouse()
    cursor.raises["COPY INTO"] = ProgrammingError(
        msg=("Unload to named stages has been disallowed; "
             "SELECT secret_token='hunter2'; "
             "https://stage/path?X-Amz-Signature=secret"), errno=3191,
        sqlstate="42000", sfqid="01b7-0000-policy")

    with pytest.raises(PermanentSourceError) as failure:
        staged(tmp_path, cursor)

    reported = str(failure.value)
    assert "ProgrammingError errno=3191 sqlstate=42000" in reported
    assert "query=01b7-0000-policy" in reported
    assert "PREVENT_UNLOAD_TO_INTERNAL_STAGES" in reported
    assert "blocks the required internal-stage unload" in reported
    assert "hunter2" not in reported
    assert "X-Amz-Signature" not in reported
    assert "Unload to named stages has been disallowed" not in reported


@pytest.mark.parametrize(("errno", "sqlstate"), [
    (3190, "42000"),
    (3191, "42001"),
])
def test_only_the_exact_internal_stage_policy_codes_are_allowlisted(
        tmp_path, errno, sqlstate):
    cursor = warehouse()
    cursor.raises["COPY INTO"] = ProgrammingError(msg="provider-secret-canary",
                                                  errno=errno,
                                                  sqlstate=sqlstate)

    with pytest.raises(PermanentSourceError) as failure:
        staged(tmp_path, cursor)

    reported = str(failure.value)
    assert "PREVENT_UNLOAD_TO_INTERNAL_STAGES" not in reported
    assert "provider-secret-canary" not in reported


@pytest.mark.parametrize("raised,reported", [
    (ServiceUnavailableError(), TransientSourceError),
    (OperationalError(msg="connection lost",
                      errno=250003), TransientSourceError),
    (RefreshTokenError(), ReconnectRequiredError),
    (ProgrammingError(msg="token expired", errno=390114,
                      sqlstate="28000"), ReconnectRequiredError),
    (ProgrammingError(msg="token expired", errno=390114,
                      sqlstate="08001"), ReconnectRequiredError),
    (ProgrammingError(msg="Insufficient privileges to operate on table",
                      errno=3001, sqlstate="42501"), SourceAccessDeniedError),
    (ProgrammingError(msg="Object does not exist", errno=2043,
                      sqlstate="42S02"), PermanentSourceError),
    (NotSupportedError(msg="no"), PermanentSourceError),
])
def test_a_driver_failure_is_classified_by_what_the_caller_should_do(
        tmp_path, raised, reported):
    """Exactly, not by subclass: a denial selects a fallback elsewhere, and
    Snowflake's not-found code cannot tell an ungranted object from an absent
    one, so reporting it as a denial would name the wrong fix."""
    cursor = warehouse()
    cursor.raises["COPY INTO"] = raised
    with pytest.raises(reported) as failure:
        staged(tmp_path, cursor)
    assert type(failure.value) is reported


def test_a_cleanup_that_cannot_run_neither_fails_a_read_nor_masks_one(tmp_path):
    """The stage is temporary and dies with the session, so objects it could
    not drop are a warning; the failure that led there is the one to report."""
    assert staged(tmp_path, warehouse(kind=StubbornStage)).num_rows == ROWS

    cursor = warehouse(kind=StubbornStage)
    cursor.raises["LIST "] = ProgrammingError(msg="listing failed", errno=4711,
                                              sqlstate="XX000")
    with pytest.raises(PermanentSourceError) as failure:
        staged(tmp_path, cursor)
    assert "errno=4711" in str(failure.value)


def test_each_read_opens_worker_sessions_and_gives_them_back(tmp_path, http):
    """Every active worker owns a session, and none outlives its read."""
    staged(tmp_path)
    http.raises = OSError("connection reset")
    with pytest.raises(TransientSourceError):
        staged(tmp_path)
    assert len(http.opened) >= 2
    assert http.released


def test_the_worker_ceiling_is_what_reaches_the_download_pool(
        tmp_path, monkeypatch):
    """Snowflake caps the CPU-derived pool at the measured optimum."""
    from kumo_snowflake import staging
    seen = []
    fetch_all = staging.fetch_all

    def record(*args, worker_ceiling=None, **kwargs):
        seen.append(worker_ceiling)
        return fetch_all(*args, worker_ceiling=worker_ceiling, **kwargs)

    monkeypatch.setattr(staging, "fetch_all", record)
    for ceiling in (None, 2):
        reader = build(warehouse())
        reader.prepare(TABLE, str(tmp_path / f"{ceiling}"), columns=COLUMNS,
                       mode=ReadMode.STAGED, worker_ceiling=ceiling)
    assert seen == [16, 2]


def test_two_overlapping_reads_of_one_table_never_stage_over_each_other(
        tmp_path):
    stage = Stage()
    finished = []

    def second_read():
        cursor = warehouse(stage=stage)
        finished.append((cursor, staged(tmp_path / "second", cursor)))

    first = warehouse(stage=stage, during=second_read)
    landed = staged(tmp_path / "first", first)
    second, overlapped = finished[0]

    assert set(first.prefixes).isdisjoint(second.prefixes)
    assert landed.num_rows == overlapped.num_rows == ROWS
    for table in (landed, overlapped):
        assert [os.path.basename(path) for path in table.paths] == list(PARTS)
        assert [os.path.getsize(path) for path in table.paths] == list(SIZES)
    assert stage.objects == {}


@pytest.mark.parametrize('phase,failure_type', [
    ('execute', RuntimeError),
    ('consume', RuntimeError),
    ('consume', KeyboardInterrupt),
])
def test_read_only_cleanup_preserves_primary(phase, failure_type, caplog):
    from unittest.mock import Mock

    from kumo_snowflake.source import read_cursor

    connection = Mock()
    cursor = connection.cursor.return_value
    failure = failure_type('private provider detail')
    cursor.close.side_effect = RuntimeError('private cleanup detail')
    if phase == 'execute':
        cursor.execute.side_effect = failure

    with pytest.raises(failure_type) as caught:
        with read_cursor(connection, 'SELECT id FROM events') as cursor:
            if phase == 'consume':
                raise failure

    assert caught.value is failure
    cursor.close.assert_called_once()
    assert 'private provider detail' not in caplog.text
    assert 'private cleanup detail' not in caplog.text


def test_read_only_cleanup_failure_is_reported_after_success(caplog):
    from unittest.mock import Mock

    from kumo_snowflake.source import ReadOnlyCursorCleanupError, read_cursor

    connection = Mock()
    cursor = connection.cursor.return_value
    cursor.close.side_effect = RuntimeError('private cleanup detail')

    with pytest.raises(ReadOnlyCursorCleanupError, match='finalize'):
        with read_cursor(connection, 'SELECT id FROM events'):
            pass

    cursor.close.assert_called_once()
    assert 'private cleanup detail' not in caplog.text
