# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from statedb.config.common import APIKeyID
from statedb.config.timestamp import ResourceTimestamps
from statedb.db import StateDB
from statedb.db.api_keys import APIKeyRepository
from statedb.rest.api_key import (
    APIKeyMetadataRecord,
    APIKeySecretRecord,
    APIKeyType,
)


def _metadata(key_id: str, *, owner: str = "user-1", name: str = "default"):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return APIKeyMetadataRecord(
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
        id=APIKeyID(key_id),
        name=name,
        type=APIKeyType.USER,
        owner_user_id=owner,
        owner_user_hash="deadbeef",
        owner_display_name=owner,
        created_time=now,
        value_preview="deadbeef...",
    )


def _secret(key_id: str):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return APIKeySecretRecord(
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
        id=APIKeyID(key_id),
        hash_version="v1",
        hash_algorithm="scrypt",
        hash_params={
            "n": 2**14,
            "r": 8,
            "p": 1,
            "dklen": 32
        },
        salt_b64="salt",
        hash_b64="hash",
    )


@pytest_asyncio.fixture
async def state_db(tmp_path):
    db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    try:
        yield db
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_repository_create_list_and_verify(state_db) -> None:
    repo = APIKeyRepository()
    async with state_db.write_tx() as tx:
        await repo.create(tx, metadata=_metadata("key-1"),
                          secret=_secret("key-1"))

    async with state_db.read_tx() as tx:
        listed = await repo.list_metadata(tx, type=APIKeyType.USER)
        candidate = await repo.find_verification_candidate(
            tx, owner_user_hash="deadbeef", name="default")

    assert [record.id for record in listed] == [APIKeyID("key-1")]
    assert candidate is not None
    assert candidate.metadata.owner_user_id == "user-1"
    assert candidate.secret.hash_b64 == "hash"


@pytest.mark.asyncio
async def test_repository_delete_removes_metadata_and_secret(state_db) -> None:
    repo = APIKeyRepository()
    async with state_db.write_tx() as tx:
        await repo.create(tx, metadata=_metadata("key-1"),
                          secret=_secret("key-1"))
        await repo.delete(tx, APIKeyID("key-1"))
    async with state_db.read_tx() as tx:
        assert await repo.get_metadata(tx, APIKeyID("key-1")) is None
        assert await repo.get_secret(tx, APIKeyID("key-1")) is None
