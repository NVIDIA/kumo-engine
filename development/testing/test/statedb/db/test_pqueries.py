# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.pquery import QueryType

from statedb.config.common import PredictiveQueryID
from statedb.config.predictive_query import PQueryDefinition
from statedb.config.timestamp import get_current_timestamps
from statedb.db import StateDB
from statedb.db.pqueries import StateDBPQueryRepository
from statedb.db.repositories import JsonRepository
from statedb.db.services import PQueryService

_GRAPH_FPID = FPID("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
_PQUERY_FPID_1 = FPID("11111111111111111111111111111111")
_PQUERY_FPID_2 = FPID("22222222222222222222222222222222")


def _definition(name: str | None = "saved_query") -> PQueryDefinition:
    return PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name=name,
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=_GRAPH_FPID,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )


class _BulkPQueryRows:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.list_by_ids_calls = []

    async def get(self, *_args, **_kwargs):
        raise AssertionError("pquery get_definitions must not call scalar get")

    async def list_by_ids(self, tx, ids):
        self.list_by_ids_calls.append((tx, ids))
        return self.rows


@pytest.mark.asyncio
async def test_pquery_repository_definition_lookup_and_alias(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    definition = _definition()

    async with db.write_tx() as tx:
        fpid = await repo.add_definition(tx, definition)
        await repo.set_alias(tx, "saved_query", fpid)

    async with db.read_tx() as tx:
        by_fpid = await repo.get_definition(tx, fpid)
        by_alias_id = await repo.get_definition_by_id(
            tx,
            PredictiveQueryID("saved_query"),
        )
        by_public_id = await repo.get_definition_by_id(
            tx,
            PredictiveQueryID(f"saved_query-{fpid}"),
        )
        alias = await repo.get_alias(tx, "saved_query")

    assert by_fpid == definition
    assert by_alias_id == definition
    assert by_public_id == definition
    assert alias == fpid


@pytest.mark.asyncio
async def test_pquery_repository_get_definitions_uses_single_bulk_read(
) -> None:
    repo = StateDBPQueryRepository()
    first_definition = _definition("first_query")
    second_definition = _definition("second_query")
    missing_fpid = FPID("ffffffffffffffffffffffffffffffff")
    fake_definitions = _BulkPQueryRows([
        SimpleNamespace(id=str(_PQUERY_FPID_2), payload=second_definition),
        SimpleNamespace(id=str(_PQUERY_FPID_1), payload=first_definition),
    ])
    repo._definitions = fake_definitions
    tx = object()

    result = await repo.get_definitions(
        tx,
        [_PQUERY_FPID_1, missing_fpid, _PQUERY_FPID_2],
    )

    assert result == {
        _PQUERY_FPID_1: first_definition,
        _PQUERY_FPID_2: second_definition,
    }
    assert list(result) == [_PQUERY_FPID_1, _PQUERY_FPID_2]
    assert fake_definitions.list_by_ids_calls == [(
        tx,
        [str(_PQUERY_FPID_1),
         str(missing_fpid),
         str(_PQUERY_FPID_2)],
    )]


@pytest.mark.asyncio
async def test_pquery_repository_definition_name_lookup_without_alias(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    definition = _definition()

    async with db.write_tx() as tx:
        fpid = await repo.add_definition(tx, definition)

    async with db.read_tx() as tx:
        assert await repo.get_alias(tx, "saved_query") == fpid
        assert await repo.get_definition_by_id(
            tx,
            PredictiveQueryID("saved_query"),
        ) == definition
        assert await repo.list_aliases(tx) == ["saved_query"]


@pytest.mark.asyncio
async def test_pquery_repository_legacy_alias_lookup_still_works(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    definition = _definition(name=None)

    async with db.write_tx() as tx:
        fpid = await repo.add_definition(tx, definition)
        await repo.set_alias(tx, "saved_query", fpid)

    async with db.read_tx() as tx:
        assert await repo.get_alias(tx, "saved_query") == fpid
        assert await repo.get_definition_by_id(
            tx,
            PredictiveQueryID("saved_query"),
        ) == definition
        assert await repo.list_aliases(tx) == ["saved_query"]


@pytest.mark.asyncio
async def test_pquery_repository_canonical_name_preferred_over_stale_alias(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    canonical = _definition(name="saved_query")
    legacy = _definition(name=None)

    async with db.write_tx() as tx:
        canonical_fpid = await repo.add_definition(tx, canonical)
        legacy_fpid = await repo.add_definition(tx, legacy)
        await repo.set_alias(tx, "saved_query", legacy_fpid)

    async with db.read_tx() as tx:
        assert await repo.get_alias(tx, "saved_query") == canonical_fpid
        assert await repo.get_definition_by_id(
            tx,
            PredictiveQueryID("saved_query"),
        ) == canonical


@pytest.mark.asyncio
async def test_pquery_repository_bulk_aliases_prefer_canonical_names(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    canonical = _definition(name="saved_query")
    legacy = _definition(name=None)
    other = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name=None,
        query_str="PREDICT SUM(t.amount) FOR EACH t.user_id",
        graph_fpid=_GRAPH_FPID,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )

    async with db.write_tx() as tx:
        canonical_fpid = await repo.add_definition(tx, canonical)
        legacy_fpid = await repo.add_definition(tx, legacy)
        other_fpid = await repo.add_definition(tx, other)
        await repo.set_alias(tx, "saved_query", legacy_fpid)
        await repo.set_alias(tx, "legacy_query", other_fpid)

    async with db.read_tx() as tx:
        assert await repo.get_aliases(
            tx, ["saved_query", "legacy_query", "missing"]) == {
                "saved_query": canonical_fpid,
                "legacy_query": other_fpid,
            }
        assert await repo.get_definitions_by_ids(
            tx,
            [
                PredictiveQueryID("saved_query"),
                PredictiveQueryID("legacy_query"),
                PredictiveQueryID("missing"),
            ],
        ) == {
            PredictiveQueryID("saved_query"): canonical,
            PredictiveQueryID("legacy_query"): other,
        }


@pytest.mark.asyncio
async def test_pquery_repository_alias_delete_and_required_lookup(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    definition = _definition(name=None)

    async with db.write_tx() as tx:
        fpid = await repo.add_definition(tx, definition)
        await repo.set_alias(tx, "saved_query", fpid)
        await repo.delete_alias(tx, "saved_query")

    async with db.read_tx() as tx:
        assert await repo.get_alias(tx, "saved_query") is None
        assert await repo.get_definition_by_id(
            tx,
            PredictiveQueryID("saved_query"),
        ) is None
        with pytest.raises(KeyError, match="PQueryDefinition"):
            await repo.get_definition_required(
                tx,
                FPID("ffffffffffffffffffffffffffffffff"),
            )


@pytest.mark.asyncio
async def test_pquery_repository_prefix_pattern_does_not_list_all(
        tmp_path, monkeypatch) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    risk_definition = _definition(name="risk_accounts")
    legacy_definition = _definition(name=None)

    async with db.write_tx() as tx:
        risk_fpid = await repo.add_definition(tx, risk_definition)
        legacy_fpid = await repo.add_definition(tx, legacy_definition)
        await repo.set_alias(tx, "risk_accounts", legacy_fpid)
        await repo.set_alias(tx, "churn_accounts", risk_fpid)

    async def fail_list_all(*_args, **_kwargs):
        raise AssertionError(
            "prefixed pquery alias pattern must not call list_all")

    monkeypatch.setattr(JsonRepository, "list_all", fail_list_all)

    async with db.read_tx() as tx:
        assert await repo.list_aliases_matching(tx,
                                                "risk_.*") == ["risk_accounts"]


@pytest.mark.asyncio
async def test_pquery_repository_delete_definition(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    definition = _definition()

    async with db.write_tx() as tx:
        fpid = await repo.add_definition(tx, definition)
        assert await repo.delete_definition(tx, fpid)
        assert not await repo.delete_definition(tx, fpid)

    async with db.read_tx() as tx:
        assert await repo.get_definition(tx, fpid) is None


@pytest.mark.asyncio
async def test_pquery_repository_add_definition_is_idempotent(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    definition = _definition(name=None)

    async with db.write_tx() as tx:
        first = await repo.add_definition(tx, definition)
        second = await repo.add_definition(tx, definition)

    async with db.read_tx() as tx:
        stored = await repo.get_definition(tx, first)

    assert second == first
    assert stored == definition


@pytest.mark.asyncio
async def test_pquery_repository_list_definitions(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = StateDBPQueryRepository()
    first = _definition(name="first_query")
    second = _definition(name="second_query")

    async with db.write_tx() as tx:
        await repo.add_definition(tx, first)
        await repo.add_definition(tx, second)

    async with db.read_tx() as tx:
        definitions = await repo.list_definitions(tx)

    assert {definition.pquery_name
            for definition in definitions} == {
                "first_query",
                "second_query",
            }


@pytest.mark.asyncio
async def test_pquery_service_definition_lookup_and_alias(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = PQueryService(db)
    definition = _definition()

    fpid = await service.add_definition(definition)
    await service.set_alias("saved_query", fpid)

    assert await service.get_definition(fpid) == definition
    assert await service.get_definition_required(fpid) == definition
    assert await service.get_alias("saved_query") == fpid
    assert await service.get_definition_by_id(PredictiveQueryID("saved_query")
                                              ) == definition
    assert await service.get_definition_by_id(
        PredictiveQueryID(f"saved_query-{fpid}")) == definition

    await service.delete_alias("saved_query")

    assert await service.get_alias("saved_query") == fpid
    assert await service.get_definition_by_id(PredictiveQueryID("saved_query")
                                              ) == definition

    await service.delete_definition(fpid)

    assert await service.get_alias("saved_query") is None
    assert await service.get_definition_by_id(PredictiveQueryID("saved_query")
                                              ) is None


@pytest.mark.asyncio
async def test_pquery_service_legacy_alias_without_definition_name(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = PQueryService(db)
    definition = _definition(name=None)

    fpid = await service.add_definition(definition)
    await service.set_alias("saved_query", fpid)

    assert await service.get_alias("saved_query") == fpid
    assert await service.get_definition_by_id(PredictiveQueryID("saved_query")
                                              ) == definition

    await service.delete_alias("saved_query")

    assert await service.get_alias("saved_query") is None
    assert await service.get_definition_by_id(PredictiveQueryID("saved_query")
                                              ) is None


@pytest.mark.asyncio
async def test_pquery_service_delete_definition(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = PQueryService(db)
    definition = _definition()

    fpid = await service.add_definition(definition)
    assert await service.delete_definition(fpid)
    assert not await service.delete_definition(fpid)
    assert await service.get_definition(fpid) is None


@pytest.mark.asyncio
async def test_pquery_service_list_definitions(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    service = PQueryService(db)
    first = _definition(name="first_query")
    second = _definition(name="second_query")

    await service.add_definition(first)
    await service.add_definition(second)

    assert {
        definition.pquery_name
        for definition in await service.list_definitions()
    } == {
        "first_query",
        "second_query",
    }
