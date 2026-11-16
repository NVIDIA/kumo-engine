# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.repositories import JsonRepository


@dataclass
class ExamplePayload:
    name: str
    count: int


@dataclass
class PayloadOnlyExample:
    id: str
    name: str
    timestamps: ResourceTimestamps


@pytest.mark.asyncio
async def test_json_repository_upsert_get_and_delete(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = JsonRepository("projects", ExamplePayload)

    async with db.write_tx() as tx:
        await repo.upsert(tx, "project_1", ExamplePayload(name="one", count=1))

    async with db.read_tx() as tx:
        row = await repo.get(tx, "project_1")

    assert row is not None
    assert row.id == "project_1"
    assert row.payload == ExamplePayload(name="one", count=1)

    async with db.write_tx() as tx:
        await repo.delete(tx, "project_1")

    async with db.read_tx() as tx:
        assert await repo.get(tx, "project_1") is None


@pytest.mark.asyncio
async def test_json_repository_upsert_preserves_created_at(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = JsonRepository("projects", ExamplePayload)

    async with db.write_tx() as tx:
        await repo.upsert(tx, "project_1", ExamplePayload(name="one", count=1))
    async with db.read_tx() as tx:
        original = await repo.get(tx, "project_1")

    async with db.write_tx() as tx:
        await repo.upsert(tx, "project_1", ExamplePayload(name="two", count=2))
    async with db.read_tx() as tx:
        updated = await repo.get(tx, "project_1")

    assert original is not None
    assert updated is not None
    assert updated.payload == ExamplePayload(name="two", count=2)
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at


@pytest.mark.asyncio
async def test_json_repository_store_payload_only_uses_physical_fields(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = JsonRepository(
        "projects",
        PayloadOnlyExample,
        store_payload_only=True,
    )
    timestamps = ResourceTimestamps(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    async with db.write_tx() as tx:
        await repo.upsert(
            tx,
            "project_1",
            PayloadOnlyExample(
                id="payload-id",
                name="one",
                timestamps=timestamps,
            ),
        )
    async with db.read_tx() as tx:
        stored = await tx.fetchrow(
            "SELECT payload FROM projects WHERE id = ?",
            "project_1",
        )
        row = await repo.get(tx, "project_1")

    assert stored is not None
    assert json.loads(stored[0]) == {"name": "one"}
    assert row is not None
    assert row.payload == PayloadOnlyExample(
        id="project_1",
        name="one",
        timestamps=timestamps,
    )


@pytest.mark.asyncio
async def test_json_repository_lists_and_filters_by_payload_field(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = JsonRepository("projects", ExamplePayload)

    async with db.write_tx() as tx:
        await repo.upsert(tx, "project_2", ExamplePayload(name="two", count=2))
        await repo.upsert(tx, "project_1", ExamplePayload(name="one", count=1))
        await repo.upsert(tx, "project_3", ExamplePayload(name="one", count=3))

    async with db.read_tx() as tx:
        all_rows = await repo.list_all(tx)
        filtered = await repo.list_by_field(tx, "name", "one")
        one = await repo.find_one_by_field(tx, "count", 2)

    assert [row.id for row in all_rows] == [
        "project_1",
        "project_2",
        "project_3",
    ]
    assert [row.id for row in filtered] == ["project_1", "project_3"]
    assert one is not None
    assert one.payload == ExamplePayload(name="two", count=2)


@pytest.mark.asyncio
async def test_json_repository_find_one_rejects_multiple_rows(
        tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = JsonRepository("projects", ExamplePayload)

    async with db.write_tx() as tx:
        await repo.upsert(tx, "project_1", ExamplePayload(name="same", count=1))
        await repo.upsert(tx, "project_2", ExamplePayload(name="same", count=2))

    async with db.read_tx() as tx:
        with pytest.raises(ValueError, match="Expected at most one projects row"):
            await repo.find_one_by_field(tx, "name", "same")


def test_json_repository_rejects_invalid_table_name() -> None:
    with pytest.raises(ValueError, match="Invalid State DB identifier"):
        JsonRepository("projects; DROP TABLE projects", ExamplePayload)


@pytest.mark.asyncio
async def test_json_repository_rejects_invalid_field_name(tmp_path) -> None:
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    repo = JsonRepository("projects", ExamplePayload)

    async with db.read_tx() as tx:
        with pytest.raises(ValueError, match="Invalid State DB identifier"):
            await repo.list_by_field(tx, "name; DROP TABLE projects", "one")
