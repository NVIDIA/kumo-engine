# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`Identity`'s slug is a random per-registration key, not derived from
`config`/`credentials` at all.

An earlier version fingerprinted `repr(credentials)`, which always hashed
the same redacted `SecretStr` placeholder regardless of the wrapped value:
two `Identity`s built from *different* secrets collided on one slug, and
the second silently resolved to the first's credentials. A later version
fixed that by unwrapping `SecretStr` before hashing, but that just traded
one problem for another: the slug became a deterministic, externally
visible function of the credentials themselves, which -- logged or
otherwise leaked, against an often-already-known connection configuration
-- is an offline verifier an attacker can check a guessed low-entropy
credential (e.g. a password) against. A random slug has neither problem.

Uses a fresh, isolated `PrincipalStore` (not the process-wide one) so
these tests don't depend on -- or pollute -- any other test's identities.
"""

import uuid

import pytest
from kumo_core.sources.principals import PrincipalStore

from kumo.engine.connector.identity import S3Identity


def test_distinct_credentials_get_distinct_slugs_and_resolve_to_themselves(
        monkeypatch) -> None:
    isolated_store = PrincipalStore()
    monkeypatch.setattr('kumo.engine.connector.identity.store', isolated_store)

    first = S3Identity(access_key_id='AKIA1', secret_access_key='secret1')
    second = S3Identity(access_key_id='AKIA2', secret_access_key='secret2')

    assert first.slug != second.slug

    first_principal = isolated_store.get(first.slug)
    second_principal = isolated_store.get(second.slug)
    assert (first_principal.credentials.access_key_id.get_secret_value() ==
            'AKIA1')
    assert (second_principal.credentials.access_key_id.get_secret_value() ==
            'AKIA2')
    assert (second_principal.credentials.secret_access_key.get_secret_value()
            == 'secret2')


def test_identical_credentials_still_get_distinct_slugs(
        monkeypatch) -> None:
    r"""Unlike the earlier credential-derived scheme, identical credentials
    no longer coalesce into one store entry -- each `Identity()` call is
    its own registration. Nothing in this package looks two `Identity`
    objects up by whether they share credentials, so this is fine."""
    isolated_store = PrincipalStore()
    monkeypatch.setattr('kumo.engine.connector.identity.store', isolated_store)

    first = S3Identity(access_key_id='AKIA1', secret_access_key='secret1')
    second = S3Identity(access_key_id='AKIA1', secret_access_key='secret1')

    assert first.slug != second.slug
    assert set(isolated_store.names()) == {first.slug, second.slug}


def test_credential_free_identities_still_get_distinct_slugs(
        monkeypatch) -> None:
    isolated_store = PrincipalStore()
    monkeypatch.setattr('kumo.engine.connector.identity.store', isolated_store)

    ambient = S3Identity()
    assume_role = S3Identity(role_arn='arn:aws:iam::123456789012:role/kumo')

    assert ambient.slug != assume_role.slug


def test_slug_never_encodes_the_credential_value(monkeypatch) -> None:
    r"""The slug must not be a deterministic function of the credentials at
    all -- not even a hashed one -- since a deterministic, externally
    visible slug is an offline verifier for a guessed credential. Two
    `Identity`s built from the same credentials getting unrelated slugs
    (not just different from each other, but not reproducible from the
    credential value itself) is what this asserts."""
    isolated_store = PrincipalStore()
    monkeypatch.setattr('kumo.engine.connector.identity.store', isolated_store)

    slugs = {
        S3Identity(access_key_id='AKIA1', secret_access_key='secret1').slug
        for _ in range(5)
    }
    assert len(slugs) == 5


def test_a_slug_collision_raises_instead_of_silently_aliasing(
        monkeypatch) -> None:
    r"""On the (astronomically unlikely) event of two `Identity()` calls
    generating the same `uuid4()` slug, registration must fail loudly, not
    silently resolve the second `Identity` to the first's stored
    credentials -- code review finding: an earlier version guarded
    `store.create(...)` behind `if self._slug not in store.names()`, which
    skipped registration (and silently aliased) on exactly this collision,
    backwards from what its own comment claimed. `store.create` already
    raises on a name collision -- calling it unconditionally is the fix.
    """
    isolated_store = PrincipalStore()
    monkeypatch.setattr('kumo.engine.connector.identity.store', isolated_store)
    monkeypatch.setattr(
        uuid, 'uuid4', lambda: uuid.UUID('12345678123456781234567812345678'))

    first = S3Identity(access_key_id='AKIA1', secret_access_key='secret1')
    with pytest.raises(ValueError, match='already exists'):
        S3Identity(access_key_id='AKIA2', secret_access_key='secret2')

    # The failed second construction must not have aliased -- the store
    # still only knows the first identity's credentials.
    assert isolated_store.names() == [first.slug]
    principal = isolated_store.get(first.slug)
    assert principal.credentials.access_key_id.get_secret_value() == 'AKIA1'
