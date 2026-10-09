# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.types.schema import FPID
from kumoapi.typing import Dtype, Stype

from statedb.config.common import (
    ID,
    DataSnapshot,
    GraphIDv2,
    JobID,
    SourceColumn,
    TableIDv2,
    TableName,
)


def test_validate_id():
    # Valid ID should pass
    ID('abA2_')

    # Invalid IDs should fail
    with pytest.raises(ValueError, match=r"contain alphanumeric"):
        ID('a??')

    with pytest.raises(ValueError, match=r"begin with a letter"):
        ID('2eee')

    with pytest.raises(ValueError, match=r"fewer than 256 characters"):
        ID('a' * 300)

    with pytest.raises(ValueError, match=r"fill out this value"):
        ID("")


def test_validate_table_id():
    # Valid ID should pass
    TableIDv2.compose(None, TableName('asdfuh2o3u'), None)

    # Invalid IDs should fail
    with pytest.raises(ValueError, match=r"contain alphanumeric"):
        TableIDv2('NA.a??b-NA')

    with pytest.raises(ValueError, match=r"fewer than 256 characters"):
        TableIDv2('NA.' + 'a' * 300 + '-NA')

    with pytest.raises(ValueError, match=r"fill out this value"):
        TableIDv2("")


@pytest.mark.parametrize('parse', [GraphIDv2, GraphIDv2.parse])
@pytest.mark.parametrize('graph_id', ['graphabc', 'graph-a-b-c'])
def test_malformed_graph_id_has_clean_error(parse, graph_id):
    with pytest.raises(ValueError, match=f'Invalid Graph ID: {graph_id}'):
        parse(graph_id)


@pytest.mark.parametrize('snapshot', list(DataSnapshot))
@pytest.mark.parametrize('suffix', ['abc', 'abc@1@2'])
def test_malformed_snapshot_id_has_clean_error(snapshot, suffix):
    snapshot_id = f'{snapshot.prefix}{suffix}'
    with pytest.raises(
            ValueError,
            match=f'{snapshot.name} snapshot ID is invalid: {snapshot_id}'):
        snapshot.parse_id(snapshot_id)


@pytest.mark.parametrize('snapshot', list(DataSnapshot))
def test_snapshot_id_round_trip(snapshot):
    fpid = FPID('a' * 32)
    assert snapshot.parse_id(snapshot.format_id(fpid,
                                                'version')) == (fpid, 'version')


def test_validate_training_job_id():
    # Valid ID should pass
    JobID('abA2-wasdfijep98aw-jesp9f43')

    # Invalid IDs should fail
    with pytest.raises(ValueError, match=r"contain alphanumeric"):
        JobID('a_23q')

    with pytest.raises(ValueError, match=r"fewer than 64 characters"):
        JobID('b.' + 'a' * 300)

    with pytest.raises(ValueError, match=r"fill out this value"):
        JobID("")


def test_api_compatibility():
    for dtype in Dtype:
        if not dtype.is_public_facing():
            continue
        compatible_stypes = []
        for stype in Stype:
            try:
                supported = stype.supports_dtype(dtype)
            except ValueError:
                supported = False
            if not supported:
                continue
            kwargs = {}
            if (stype == Stype.multicategorical
                    and dtype == Dtype.string):
                kwargs["multicategorical_delimiter"] = ","
            a = SourceColumn(name="test", stype=stype, dtype=dtype,
                             is_primary=False, **kwargs)
            assert a is not None
            compatible_stypes.append(stype)
        if dtype != Dtype.unsupported:
            assert compatible_stypes, f'No public stype accepts {dtype}'
