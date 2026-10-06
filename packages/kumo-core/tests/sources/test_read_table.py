# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choosing a mode is one decision, so both platforms make it here."""

import pytest
from kumo_core.sources.errors import (
    SourceAccessDeniedError,
    TransientSourceError,
    UnsupportedReadModeError,
)
from kumo_core.sources.prepare import read_table
from kumo_core.sources.types import Attestation, ReadMode, StagedTable
from kumo_core.types.enums import ConnectorType


class Reader:
    """A reader that records the modes it was asked for, and can refuse."""
    def __init__(self, offered=(ReadMode.STAGED, ReadMode.STREAMED), denies=(),
                 raises=None):
        self.offered = tuple(offered)
        self.denies = set(denies)
        self.raises = raises
        self.attempts = []
        self.asked = 0

    source_type = ConnectorType.SNOWFLAKE

    def read_modes(self, table):
        self.asked += 1
        return self.offered

    def data_version(self, table):
        return "7"

    def prepare(self, table, dest, *, mode, **kwargs):
        self.attempts.append(mode)
        if self.raises is not None:
            raise self.raises
        if mode in self.denies:
            raise SourceAccessDeniedError("no grant", table=table)
        return StagedTable(directory=dest, paths=(f"{dest}/p.parquet", ),
                           version="7", source_size=4, mode_used=mode,
                           num_rows=1, rows_attested_by=Attestation.LOCAL)


def read(reader, **kwargs):
    return read_table(reader, "ORDERS", "/tmp/dest", columns={}, **kwargs)


def test_the_cheapest_mode_a_reader_offers_is_the_one_that_runs():
    reader = Reader()
    assert read(reader).mode_used is ReadMode.STAGED
    assert reader.attempts == [ReadMode.STAGED]


def test_a_denied_stage_is_the_one_failure_that_retries_streamed():
    reader = Reader(denies={ReadMode.STAGED})
    assert read(reader).mode_used is ReadMode.STREAMED
    assert reader.attempts == [ReadMode.STAGED, ReadMode.STREAMED]
    # Once across the whole read, fallback included: establishing the modes
    # can cost a warehouse round trip whose answer a reader is right not to
    # remember.
    assert reader.asked == 1


def test_a_denial_with_nothing_to_fall_back_to_is_raised():
    reader = Reader(offered=(ReadMode.STAGED, ), denies={ReadMode.STAGED})
    with pytest.raises(SourceAccessDeniedError):
        read(reader)
    assert reader.attempts == [ReadMode.STAGED]


def test_a_reader_that_only_streams_has_nothing_to_fall_back_to():
    """The denial is about the streamed read itself, so retrying it streamed
    would just ask the same question again. Distinct from the staged-only
    case beside it: there streaming is what is missing, here it is what was
    refused."""
    reader = Reader(offered=(ReadMode.STREAMED, ), denies={ReadMode.STREAMED})
    with pytest.raises(SourceAccessDeniedError):
        read(reader)
    assert reader.attempts == [ReadMode.STREAMED]


def test_any_other_failure_is_not_retried_in_a_second_mode():
    reader = Reader(raises=TransientSourceError("timed out", table="ORDERS"))
    with pytest.raises(TransientSourceError):
        read(reader)
    assert reader.attempts == [ReadMode.STAGED]


def test_an_explicit_mode_is_never_silently_replaced():
    reader = Reader(denies={ReadMode.STAGED})
    with pytest.raises(SourceAccessDeniedError):
        read(reader, mode=ReadMode.STAGED)
    assert reader.attempts == [ReadMode.STAGED]


def test_an_explicit_mode_does_not_ask_what_else_is_offered():
    reader = Reader()
    assert read(reader, mode=ReadMode.STREAMED).mode_used is ReadMode.STREAMED
    assert reader.asked == 0


def test_a_reader_offering_nothing_says_so_before_it_reads():
    reader = Reader(offered=())
    with pytest.raises(UnsupportedReadModeError, match="no way to read"):
        read(reader)
    assert reader.attempts == []


@pytest.mark.parametrize("named", ["staged", ReadMode.STAGED])
def test_a_mode_named_as_a_string_runs_the_mode_it_names(named):
    reader = Reader()
    assert read(reader, mode=named).mode_used is ReadMode.STAGED
    assert reader.attempts == [ReadMode.STAGED]


def test_a_mode_that_is_not_one_is_refused_before_anything_is_read():
    reader = Reader()
    with pytest.raises(ValueError, match="not a valid ReadMode"):
        read(reader, mode="fast")
    assert reader.attempts == []
