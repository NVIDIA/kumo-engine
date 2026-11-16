# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from kumo_core.sources.principals import (
    PrincipalStore,
    identity_id,
    resolve,
    store,
)
from kumoapi.data_source import (
    CreateIdentityArgs,
    IdentityType,
    S3AmbientAuthentication,
    S3IdentityConfig,
    SnowflakeAuthenticationKind,
    SnowflakeIdentityConfig,
    SnowflakeKeyPairCredentials,
)
from pydantic import SecretStr

SNOWFLAKE = SnowflakeIdentityConfig(
    account="xy12345",
    username="KUMO_READER",
    authentication=SnowflakeAuthenticationKind.KEY_PAIR,
)
KEY = SnowflakeKeyPairCredentials(private_key=SecretStr("unused"))
S3 = S3IdentityConfig(authentication=S3AmbientAuthentication())


@pytest.fixture
def isolated(monkeypatch):
    r"""A store of this test's own.

    The module-level one is process-global on purpose -- the identity area and
    the data sources must not disagree about a name -- so a test that wants to
    put things in it swaps it out rather than mutating the shared one.
    """
    fresh = PrincipalStore()
    monkeypatch.setattr("kumo_core.sources.principals.store", fresh)
    return fresh


def _create(store, name, config=SNOWFLAKE, credentials=KEY, slug=None):
    return store.create(
        CreateIdentityArgs(display_name=name, slug=slug, config=config,
                           credentials=credentials))


def test_a_principal_reports_its_provider_from_its_config(isolated):
    principal = _create(isolated, "prod-sf").principal
    assert principal.type is IdentityType.SNOWFLAKE
    assert principal.name == "prod-sf"
    assert principal.credentials == KEY


def test_an_ambient_identity_holds_no_credentials(isolated):
    r"""``None`` is a configured state, not a missing one: an S3 ambient
    identity signs with the host's own chain and has nothing to store.
    """
    principal = _create(isolated, "prod-s3", config=S3,
                        credentials=None).principal
    assert principal.credentials is None
    assert principal.type is IdentityType.S3


def test_an_id_is_derived_from_the_name_and_is_stable():
    r"""Connectors and saved graphs record this, so it is a compatibility
    surface: the same name must yield the same ID in every process.
    """
    assert identity_id("prod-sf") == identity_id("prod-sf")
    assert identity_id("prod-sf") != identity_id("prod-bq")
    assert identity_id("prod-sf").startswith("identity-")


@pytest.mark.parametrize("spelling", ["name", "slug", "id", "object"])
def test_resolution_understands_every_way_to_name_an_identity(
        isolated, spelling):
    _create(isolated, "prod-sf", slug="prod-sf-slug")
    given = {
        "name": "prod-sf",
        "slug": "prod-sf-slug",
        "id": identity_id("prod-sf"),
        # Matched by shape, not by type: the SDK's identity object lives above
        # this layer, so anything carrying a slug is taken to mean that slug.
        "object": type("Identity", (), {"slug": "prod-sf-slug"})(),
    }
    assert isolated.resolve(given[spelling]) == "prod-sf"


def test_an_unknown_reference_resolves_to_none(isolated):
    r"""Not an error here: a caller who also supplied a default meant it to be
    used, and only the caller knows whether it had one.
    """
    assert isolated.resolve("nope") is None
    assert isolated.resolve(None) is None


def test_creating_a_name_twice_is_refused(isolated):
    _create(isolated, "prod-sf")
    with pytest.raises(ValueError, match="already exists"):
        _create(isolated, "prod-sf")


@pytest.mark.parametrize("name", ["", " prod-sf", "prod-sf "])
def test_a_name_that_would_not_match_exactly_is_refused(isolated, name):
    r"""Names are matched exactly and recorded in saved graphs, so padding
    that a caller cannot see would orphan a graph later.
    """
    with pytest.raises(ValueError, match="(?i)identity name"):
        _create(isolated, name)


def test_rotation_replaces_material_and_leaves_the_principal_alone(isolated):
    before = _create(isolated, "prod-sf").principal
    rotated = SnowflakeKeyPairCredentials(private_key=SecretStr("rotated"))
    after = isolated.rotate("prod-sf", rotated).principal

    assert after.config == before.config
    assert after.credentials == rotated
    assert isolated.get("prod-sf").credentials == rotated


def test_rotation_refuses_material_the_config_cannot_use(isolated):
    r"""The pairing is validated on the way in, so a key-pair identity cannot
    be rotated onto a password.
    """
    _create(isolated, "prod-sf")
    with pytest.raises(ValueError):
        isolated.rotate("prod-sf", None)


def test_the_store_hands_out_only_what_a_plugin_reads(isolated):
    r"""The slug and description stay on the registration; a principal is the
    four things a plugin actually uses.
    """
    registration = _create(isolated, "prod-sf", slug="prod-sf-slug")
    assert registration.slug == "prod-sf-slug"
    assert not hasattr(registration.principal, "slug")
    assert isolated.get("prod-sf") == registration.principal


def _resolve(**kwargs):
    return resolve(config_example="SnowflakeIdentityConfig(...)",
                   fallback="A default is also accepted.", **kwargs)


def test_resolution_prefers_the_connector_over_the_backend_default(isolated):
    _create(isolated, "on-connector")
    _create(isolated, "the-default")
    connector = type("C", (), {"id": "c1", "identity": "on-connector"})()

    assert _resolve(candidates=(connector.identity, None, "the-default"),
                    connector=connector).name == "on-connector"


def test_an_unresolvable_reference_falls_through_to_the_next(isolated):
    r"""A caller who supplied a default meant it to be used, so a stale name
    on a connector is not fatal while a usable default is standing by.
    """
    _create(isolated, "the-default")
    connector = type("C", (), {"id": "c1"})()

    assert _resolve(candidates=("gone", None, "the-default"),
                    connector=connector).name == "the-default"


def test_naming_nothing_that_resolves_reports_what_was_named(isolated):
    r"""An identity never created and a name misspelled at creation are the
    two ways to arrive here, and only the names tell them apart.
    """
    _create(isolated, "prod-sf")
    connector = type("C", (), {"id": "c1"})()

    with pytest.raises(ValueError) as raised:
        _resolve(candidates=("typo-sf", None, None), connector=connector)
    message = str(raised.value)
    assert "'c1'" in message
    assert "'typo-sf'" in message
    assert "'prod-sf'" in message
    assert "A default is also accepted." in message


def test_an_optional_identity_is_none_only_when_nothing_was_named(isolated):
    r"""An unset identity selects the ambient credential chain; a name that
    fails to resolve must not, since that would sign as a different principal
    without saying so.
    """
    connector = type("C", (), {"id": "c1"})()
    assert _resolve(candidates=(None, None, None), connector=connector,
                    optional=True) is None
    with pytest.raises(ValueError, match="'named'"):
        _resolve(candidates=("named", None, None), connector=connector,
                 optional=True)


def test_the_module_level_store_is_the_one_everything_shares():
    r"""One store, so the identity area and the data sources cannot disagree
    about what a name means.
    """
    assert isinstance(store, PrincipalStore)
    assert isinstance(store.names(), list)
