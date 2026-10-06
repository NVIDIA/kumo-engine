# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The digests a Custom SQL graph is identified by, written down.

`custom_sql_synthetic_name` is md5 over sqlglot's regeneration of a query, and
that name is inside every `graph_id` a SQL-bearing graph takes.

A failure here is never fixed by updating the expected value. Either the
sqlglot bump is reverted, or the identity change is deliberate and every
stored graph built against the old digests is migrated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from kumo_core.custom_sql import custom_sql_synthetic_name
from kumo_core.sources import identity
from kumo_core.sources.identity import sql_data_version
from kumo_core.types.enums import ConnectorType

ROOT = Path(__file__).resolve().parents[3]

#: One representative query per shape that normalization has an opinion
#: about. The commented projection must hash to the bare one.
SYNTHETIC_NAMES = {
    ConnectorType.SNOWFLAKE: {
        'SELECT ID, Amount FROM Orders WHERE Amount > 0':
        'custom_sql_b2e9b9b6a8250b99c4409c96122a3364',
        'SELECT ID, /* keep */ Amount -- daily\nFROM Orders WHERE Amount > 0':
        'custom_sql_b2e9b9b6a8250b99c4409c96122a3364',
        'SELECT "Id" AS id FROM "Orders"':
        'custom_sql_48ff86576389bdcdc05dbd609b6093c2',
        "WITH Recent AS (SELECT * FROM Orders WHERE Ts > '2026-01-01') "
        "SELECT r.ID, c.Segment FROM Recent r JOIN Customers c ON c.ID = r.ID":
        'custom_sql_e306311f367019620132c0dca41ac333',
        'SELECT CAST(Amount AS DECIMAL(10, 2)) AS amount FROM Orders':
        'custom_sql_ef8820b1be27f7af7f7705f50259b624',
        "SELECT a FROM Orders AT (TIMESTAMP => '2026-01-01'::TIMESTAMP)":
        'custom_sql_9e1125ecd50deb1d5c1b41416b91b07f',
    },
    ConnectorType.DATABRICKS: {
        'SELECT ID, Amount FROM Orders WHERE Amount > 0':
        'custom_sql_b2e9b9b6a8250b99c4409c96122a3364',
        'SELECT ID, /* keep */ Amount -- daily\nFROM Orders WHERE Amount > 0':
        'custom_sql_b2e9b9b6a8250b99c4409c96122a3364',
        'SELECT `Id` AS id FROM `Orders`':
        'custom_sql_3f22ee32a444618c2a672a341f600d1c',
        "WITH Recent AS (SELECT * FROM Orders WHERE Ts > '2026-01-01') "
        "SELECT r.ID, c.Segment FROM Recent r JOIN Customers c ON c.ID = r.ID":
        'custom_sql_e306311f367019620132c0dca41ac333',
        'SELECT CAST(Amount AS DECIMAL(10, 2)) AS amount FROM Orders':
        'custom_sql_ef8820b1be27f7af7f7705f50259b624',
        'SELECT ID, explode(Tags) AS tag FROM Orders':
        'custom_sql_db68dbc96ca23cd3a97c8f2bc7f33f36',
    },
    ConnectorType.BIGQUERY: {
        'SELECT ID, Amount FROM Orders WHERE Amount > 0':
        'custom_sql_4c0efb64d2d03cf63abd19bdc34fa994',
        'SELECT ID, /* keep */ Amount -- daily\nFROM Orders WHERE Amount > 0':
        'custom_sql_4c0efb64d2d03cf63abd19bdc34fa994',
        'SELECT id FROM `proj.ds.Orders`':
        'custom_sql_a9f5d16e98d383c46a53b7b15f9a1963',
        "WITH Recent AS (SELECT * FROM Orders WHERE Ts > '2026-01-01') "
        "SELECT r.ID, c.Segment FROM Recent r JOIN Customers c ON c.ID = r.ID":
        'custom_sql_b92cae3329763f9959ad021d583b0ee5',
        'SELECT CAST(Amount AS NUMERIC(10, 2)) AS amount FROM Orders':
        'custom_sql_15949573d8eecac2885ed9187a577bb4',
        'SELECT ID, t FROM Orders, UNNEST(Tags) AS t':
        'custom_sql_0b0a25b76a6b0325102000682b2dc3ea',
    },
}

#: The other digest a bump can move: the version a snapshot is restored on.
DATA_VERSIONS = {
    (ConnectorType.SNOWFLAKE, 'SELECT ID, Amount FROM Orders WHERE Amount > 0'):
    '0f8517bfc86c7793a7f3e64079da9f20',
    (ConnectorType.DATABRICKS, 'SELECT CAST(Amount AS DECIMAL(10, 2)) AS amount FROM Orders'):
    '25ea5babaabeceb114da53346c64e8d9',
    (ConnectorType.BIGQUERY, 'SELECT CAST(Amount AS NUMERIC) AS amount FROM Orders'):
    '128aeaff1e52690bb58c200854cdb7b2',
}


@pytest.mark.parametrize('connector', sorted(SYNTHETIC_NAMES, key=str))
def test_a_query_still_takes_the_name_it_always_took(connector) -> None:
    for sql, expected in SYNTHETIC_NAMES[connector].items():
        assert custom_sql_synthetic_name(sql, connector) == expected, sql


@pytest.mark.parametrize('key', sorted(DATA_VERSIONS, key=str))
def test_a_query_still_takes_the_version_it_always_took(key, monkeypatch):
    # The staleness ceiling is a wall-clock bucket; pinned so this asserts
    # only what it is about.
    monkeypatch.setattr(identity, 'time', lambda: 1_788_000_000.0)
    connector, sql = key
    assert sql_data_version(sql, connector, lambda leaf: 'v1',
                            {'amount': 'float64'}) == DATA_VERSIONS[key]


def _declared_pin() -> str:
    text = (ROOT / 'packages' / 'kumo-core' / 'pyproject.toml').read_text()
    found = re.findall(r'^\s*"sqlglot([^"]*)",\s*$', text, re.MULTILINE)
    assert len(found) == 1, f"expected one sqlglot dependency, got {found}"
    return found[0]


def _build_pin() -> str:
    text = (ROOT / '3rdparty' / 'BUILD').read_text()
    found = re.search(r'^SQLGLOT_VERSION = "([^"]+)"$', text, re.MULTILINE)
    assert found is not None, 'SQLGLOT_VERSION is not declared in 3rdparty/BUILD'
    return found.group(1)


@pytest.mark.skipif(not (ROOT / '3rdparty' / 'BUILD').exists(),
                    reason='the monorepo resolve is not present in a wheel')
def test_the_two_sqlglot_declarations_are_one_constant() -> None:
    assert _build_pin() == _declared_pin()


def test_the_pin_fixes_one_minor() -> None:
    assert re.fullmatch(r'==\d+\.\d+\.\*', _declared_pin()), (
        f'sqlglot must be pinned to one minor, got {_declared_pin()!r}')
