# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace

import pytest
from native_io_queue import EventBuffer, bio_next_offset, start_capture
from native_io_queue_results import EVENT


def test_ram_capacity_retains_failure():
    buffer = EventBuffer(EVENT.size)
    payload = ctypes.create_string_buffer(EVENT.size)
    buffer.receive(0, ctypes.addressof(payload), EVENT.size)
    buffer.receive(0, ctypes.addressof(payload), EVENT.size)
    assert len(buffer.data) == EVENT.size
    assert buffer.diagnostics['ram_overflow'] == 1


def test_transport_and_record_size_failures():
    buffer = EventBuffer(EVENT.size)
    payload = ctypes.create_string_buffer(EVENT.size)
    buffer.receive(0, ctypes.addressof(payload), EVENT.size - 1)
    buffer.lost(1, 12)
    assert buffer.diagnostics == dict(perf_lost=12, ram_overflow=0, record_size=1)
    assert not buffer.data


def test_invalid_capacity():
    with pytest.raises(ValueError):
        EventBuffer(EVENT.size - 1)


def test_perf_alignment_bytes_are_not_payload():
    buffer = EventBuffer(EVENT.size * 2)
    # Actual rejected model sample: 72 event bytes plus four unspecified bytes
    # skipped by perf_output_sample, included in BCC's 76-byte RAW callback.
    sample = bytes.fromhex(
        '70ac65eaaa1508000096bc930000ffff70ac65eaaa150800329f090000000000'
        '0000000000000000010000000000000008000000000000002051040012000000'
        'a30b00000000000001005800')
    payload = ctypes.create_string_buffer(sample)
    buffer.receive(0, ctypes.addressof(payload), EVENT.size + 4)
    assert bytes(buffer.data) == sample[:EVENT.size]
    assert buffer.received_sizes == {76: 1}
    assert buffer.diagnostics['record_size'] == 0


@pytest.mark.parametrize('size', [0, 71, 73, 74, 75, 77, 80])
def test_perf_unexpected_width_fails(size):
    buffer = EventBuffer(EVENT.size)
    payload = ctypes.create_string_buffer(80)
    buffer.receive(0, ctypes.addressof(payload), size)
    assert not buffer.data
    assert buffer.diagnostics['record_size'] == 1


def test_transport_registration_precedes_capture():
    calls = []
    value = SimpleNamespace(capture_enabled=0, pending=0, bootstrap_ns=0, end_ns=0)

    class StateMap:
        Key = int

        def __getitem__(self, key):
            return value

        def __setitem__(self, key, state):
            assert calls == ['registered']
            assert state.capture_enabled == 1
            calls.append('enabled')

    class Events:
        def open_perf_buffer(self, callback, lost_cb, page_cnt):
            assert value.capture_enabled == 0
            calls.append('registered')

    start_capture({'events': Events(), 'state': StateMap()}, EventBuffer(EVENT.size))
    assert calls == ['registered', 'enabled']


def test_bio_chain_offset_uses_actual_btf_pointer():
    types = [dict(id=1, kind='STRUCT', name='bio', members=[
        dict(name='bi_next', bits_offset=128, type_id=2)]), dict(id=2, kind='PTR')]
    assert bio_next_offset(types) == 16
    types[1]['kind'] = 'INT'
    with pytest.raises(ValueError, match='pointer'):
        bio_next_offset(types)
