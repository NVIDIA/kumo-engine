# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace

import pytest
from native_io_queue import (
    EventBuffer,
    bio_next_offset,
    empty_flush_layout,
    flush_layout,
    start_capture,
    unknown_provenance,
)
from native_io_queue_results import EVENT, UNKNOWN


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
    value = SimpleNamespace(capture_enabled=0, pending=0, pending_flush=0, bootstrap_ns=0, end_ns=0)

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


def flush_types():
    return [dict(id=1, kind='STRUCT', name='request', members=[
        dict(name='rq_flags', bits_offset=32, type_id=2),
        dict(name='__data_len', bits_offset=64, type_id=2),
        dict(name='start_time_ns', bits_offset=128, type_id=3),
        dict(name='cmd_flags', bits_offset=0, type_id=2)]),
        dict(id=2, kind='INT', size=4), dict(id=3, kind='INT', size=8),
        dict(id=4, kind='ENUM', values=[dict(name='__RQF_FLUSH_SEQ', val=1)])]


def test_flush_layout_proves_fields_and_mask():
    assert flush_layout(flush_types()) == dict(OFF_REQUEST_CMD_FLAGS=0, OFF_REQUEST_RQ_FLAGS=4, OFF_REQUEST_DATA_LEN=8,
                                               OFF_REQUEST_START_TIME_NS=16, RQF_FLUSH_SEQ_MASK=2)


@pytest.mark.parametrize('mutation', [
    lambda t: t[0]['members'][0].update(bits_offset=33),
    lambda t: t[0]['members'][0].update(bitfield_size=1),
    lambda t: t[0]['members'].pop(),
    lambda t: t[1].update(size=8),
    lambda t: t[2].update(kind='PTR'),
    lambda t: t[3]['values'][0].update(val=2),
    lambda t: t[3].update(values=[]),
])
def test_flush_layout_mismatch_fails(mutation):
    types = flush_types()
    mutation(types)
    with pytest.raises(ValueError, match='BTF'):
        flush_layout(types)


def test_unknown_harvest_only_referenced_stacks():
    walked = []

    class Stacks:
        def walk(self, index):
            walked.append(index)
            return [0xffff1000, 0xffff2000]

    class Bpf:
        def __getitem__(self, name):
            assert name == 'unknown_stacks'
            return Stacks()

        def ksym(self, pc):
            return f'kernel_{pc:x}'.encode()

    def packed(stack):
        return EVENT.pack(30, 1234, 10, 0x1201, 0, UNKNOWN, 1, 2, stack, 100, 3, 1)

    value = unknown_provenance(Bpf(), packed(7) + packed(0xffffffff - 11))
    assert walked == [7]
    assert value['stack_capacity'] == 128
    assert value['stack_errors'] == [-12]
    assert value['stacks'] == {'7': [0xffff1000, 0xffff2000]}
    assert value['symbols'] == {'4294905856': 'kernel_ffff1000', '4294909952': 'kernel_ffff2000'}
    assert value['events'][0]['cmd_flags'] == 0x1201
    assert value['events'][0]['start_time_ns'] == 10


def test_no_unknown_needs_no_stack_access():
    assert unknown_provenance({}, b'') == dict(
        encoding_version=1, stack_capacity=128, events=[], stacks={}, symbols={}, stack_errors=[])


def test_unknown_successful_stack_cannot_be_empty():
    class Stacks:
        def walk(self, index):
            return []

    raw = EVENT.pack(30, 1234, 10, 1, 0, UNKNOWN, 1, 0, 7, 100, 3, 1)
    with pytest.raises(ValueError, match='no kernel PCs'):
        unknown_provenance({'unknown_stacks': Stacks()}, raw)


def empty_types():
    return flush_types() + [
        dict(id=5, kind='TYPEDEF', name='bool', type_id=6),
        dict(id=6, kind='INT', name='_Bool', size=1),
        dict(id=7, kind='PTR', type_id=1),
        dict(id=8, kind='FUNC_PROTO', ret_type_id=5, params=[dict(name='rq', type_id=7)]),
        dict(id=9, kind='FUNC', name='blk_insert_flush', type_id=8),
        dict(id=10, kind='ENUM', values=[dict(name=name, val=value) for name, value in (
            ('REQ_OP_WRITE', 1), ('__REQ_PREFLUSH', 18), ('__REQ_FUA', 17), ('__REQ_SYNC', 11))])]


def test_empty_flush_entry_signature_and_enums():
    value = empty_flush_layout(empty_types())
    assert value['signature'] == 'bool(struct request *)'
    assert value['function_type_id'] == 9 and value['request_type_id'] == 1
    assert value['enums']['__REQ_PREFLUSH'] == 18


@pytest.mark.parametrize('mutation', [
    lambda t: t[8].update(name='missing'),
    lambda t: t[7].update(params=[]),
    lambda t: t[5].update(size=4),
    lambda t: t[6].update(kind='INT'),
    lambda t: t[0].update(name='bio'),
    lambda t: t[9]['values'][1].update(val=19),
    lambda t: t[9]['values'].pop(),
])
def test_empty_flush_btf_mismatch_fails(mutation):
    types = empty_types()
    mutation(types)
    with pytest.raises(ValueError, match='BTF'):
        empty_flush_layout(types)
