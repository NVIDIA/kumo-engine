# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from rest.driver import api_key_driver
from rest.driver.api_key_driver import APIKeyDriver, APIKeyVerificationError
from statedb.config.common import APIKeyID
from statedb.config.timestamp import ResourceTimestamps, now_utc
from statedb.db import StateDB
from statedb.db.api_keys import APIKeyRepository
from statedb.db.services import StateServices
from statedb.rest.api_key import (
    APIKeyMetadataRecord,
    APIKeySecretRecord,
    APIKeyType,
)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.mark.asyncio
async def test_api_key_driver_state_db_create_verify_list_delete(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    driver = APIKeyDriver(
        state_services=StateServices(state_db),
        update_last_used_time=True,
    )

    created = await driver.create_admin_api_key(
        "state-key",
        APIKeyType.ADMIN,
        "direct_connect|admin1",
        user_name="Admin One",
    )

    assert created.name == "state-key"
    assert len(created.value.split(":")) == 3

    listed = await driver.list_admin_api_keys(APIKeyType.ADMIN)
    assert [record.id for record in listed] == [created.id]
    assert listed[0].owner_display_name == "Admin One"

    identity = await driver.verify_api_key(created.value)
    assert identity is not None
    assert identity.user_id == "direct_connect|admin1"
    assert identity.is_admin

    repo = APIKeyRepository()
    async with state_db.read_tx() as tx:
        secret = await repo.get_secret(tx, created.id)
    assert secret is not None
    assert secret.last_used_time is not None

    await driver.delete_admin_api_key(created.id)

    assert await driver.list_admin_api_keys() == []
    with pytest.raises(APIKeyVerificationError):
        await driver.verify_api_key(created.value)


@pytest.mark.asyncio
async def test_api_key_driver_verifies_existing_legacy_prefixed_key(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    services = StateServices(state_db)
    driver = APIKeyDriver(
        state_services=services,
        update_last_used_time=False,
    )
    owner_user_id = "external|existing-owner"
    owner_user_hash = api_key_driver._compute_user_hash(owner_user_id)
    name = "existing-key"
    key_value = f"legacy:{owner_user_hash}:{name}:secret"
    key_id = APIKeyID("k-existing-legacy-key")
    created_time = now_utc()
    timestamps = ResourceTimestamps(
        created_at=created_time,
        updated_at=created_time,
    )
    salt_b64, hash_b64 = await api_key_driver._hash_api_key_value_async(
        key_value)

    await services.api_keys.create_unique(
        metadata=APIKeyMetadataRecord(
            timestamps=timestamps,
            id=key_id,
            name=name,
            type=APIKeyType.USER,
            owner_user_id=owner_user_id,
            owner_user_hash=owner_user_hash,
            owner_display_name="Existing Owner",
            created_time=created_time,
            value_preview=api_key_driver._to_preview(key_value),
        ),
        secret=APIKeySecretRecord(
            timestamps=timestamps,
            id=key_id,
            hash_version=api_key_driver._API_KEY_HASH_VERSION,
            hash_algorithm=api_key_driver._API_KEY_HASH_ALGORITHM,
            hash_params={
                "n": api_key_driver._SCRYPT_N,
                "r": api_key_driver._SCRYPT_R,
                "p": api_key_driver._SCRYPT_P,
                "dklen": api_key_driver._SCRYPT_DKLEN,
            },
            salt_b64=salt_b64,
            hash_b64=hash_b64,
        ),
    )

    identity = await driver.verify_api_key(key_value)
    assert identity is not None
    assert identity.user_id == owner_user_id
    assert not identity.is_admin

    await state_db.close()


@pytest.mark.asyncio
async def test_api_key_driver_ignores_stale_legacy_metadata_rows(
        tmp_path) -> None:
    state_db = await StateDB.open_sqlite(str(tmp_path / "state.db"))
    driver = APIKeyDriver(
        state_services=StateServices(state_db),
        update_last_used_time=True,
    )
    repo = APIKeyRepository()
    legacy_id = "legacy-key"

    async with state_db.write_tx() as tx:
        await tx.execute(
            """
            INSERT INTO api_keys (id, payload, created_at, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            legacy_id,
            ('{"id":"legacy-key",'
             '"name":"old","type":"legacy","owner_user_id":"legacy|old",'
             '"owner_display_name":"Old","created_time":"2024-01-01T00:00:00Z",'
             '"value_preview":"****"}'),
        )

    async with state_db.read_tx() as tx:
        assert await repo.get_metadata(tx, legacy_id) is None
        with pytest.raises(ValidationError):
            await repo._metadata.get_payload(tx, legacy_id)

    assert await driver.list_admin_api_keys() == []
