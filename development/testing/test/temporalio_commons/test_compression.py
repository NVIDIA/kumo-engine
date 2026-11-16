# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib

import numpy
import pytest
from temporalio.api.common.v1 import Payload

from temporalio_commons.compression_codec import (
    BINARY_COMPRESSION_ENCODING,
    BINARY_OFFLOADED_ENCODING,
    CompressionCodec,
)


def _make_rand_payload(min_size: int):
    rand_data = numpy.random.randint(2, size=(min_size + 1, ))
    rand_data = rand_data.astype(numpy.uint16).tobytes()
    return Payload(
        metadata={
            'encoding': b'binary',
        },
        data=rand_data,
    )


@pytest.mark.asyncio
async def test_offloading(monkeypatch: pytest.MonkeyPatch,
                          tmp_path: pathlib.Path):
    offload_size_threshold = 999
    monkeypatch.setenv('CODEC_OFFLOAD_SIZE_THRESHOLD',
                       str(offload_size_threshold))
    monkeypatch.setenv('CODEC_COMPRESS_SIZE_THRESHOLD', '9999999')
    payload_small = Payload(
        metadata={
            'encoding': b'json',
        },
        data=json.dumps({
            'key': 'value'
        }).encode(),
    )
    payload_large = _make_rand_payload(offload_size_threshold)

    tmp_payloads_dir = tmp_path.absolute().as_posix()
    codec = CompressionCodec(tmp_payloads_dir)
    encoded_payloads = await codec.encode([payload_small, payload_large])
    assert encoded_payloads[0] == payload_small  # noop

    # Offloaded to {tmp_path}/payload_<uuid>.bin
    assert encoded_payloads[1].metadata[
        'encoding'] == BINARY_OFFLOADED_ENCODING
    path = encoded_payloads[1].metadata['offload_path'].decode()
    assert path.startswith(f'{tmp_payloads_dir}/payload_')
    assert len(encoded_payloads[1].data) == 0

    payloads = await codec.decode(encoded_payloads)
    assert payloads[0] == payload_small
    assert payloads[1] == payload_large


@pytest.mark.asyncio
async def test_compression_round_trip(monkeypatch: pytest.MonkeyPatch):
    compression_size_threshold = 999
    monkeypatch.setenv('CODEC_COMPRESS_SIZE_THRESHOLD',
                       str(compression_size_threshold))
    payload_small = Payload(
        metadata={
            'encoding': b'json',
        },
        data=json.dumps({
            'key': 'value'
        }).encode(),
    )
    payload_large = _make_rand_payload(compression_size_threshold)

    codec = CompressionCodec()
    encoded_payloads = await codec.encode([payload_small, payload_large])
    assert encoded_payloads[0] == payload_small
    assert encoded_payloads[1].metadata[
        'encoding'] == BINARY_COMPRESSION_ENCODING
    assert len(encoded_payloads[1].data) < len(payload_large.data)

    payloads = await codec.decode(encoded_payloads)
    assert payloads[0] == payload_small
    assert payloads[1] == payload_large


@pytest.mark.asyncio
async def test_compression_and_offloading(monkeypatch: pytest.MonkeyPatch,
                                          tmp_path: pathlib.Path):
    compression_size_threshold = 100
    offload_size_threshold = compression_size_threshold * 10
    monkeypatch.setenv('CODEC_OFFLOAD_SIZE_THRESHOLD',
                       str(offload_size_threshold))
    monkeypatch.setenv('CODEC_COMPRESS_SIZE_THRESHOLD',
                       str(compression_size_threshold))
    payload_small = Payload(
        metadata={
            'encoding': b'json',
        },
        data=json.dumps({
            'key': 'value'
        }).encode(),
    )
    payload_medium = _make_rand_payload(compression_size_threshold)
    # Make sure it's large enough that compressed size is above threshold.
    payload_large = _make_rand_payload(offload_size_threshold * 10)

    tmp_payloads_dir = tmp_path.absolute().as_posix()
    codec = CompressionCodec(tmp_payloads_dir)
    encoded_payloads = await codec.encode(
        [payload_small, payload_medium, payload_large])
    assert encoded_payloads[0] == payload_small  # noop

    assert encoded_payloads[1].metadata[
        'encoding'] == BINARY_COMPRESSION_ENCODING
    assert encoded_payloads[2].metadata[
        'encoding'] == BINARY_OFFLOADED_ENCODING
    path = encoded_payloads[2].metadata['offload_path'].decode()
    assert path.startswith(f'{tmp_payloads_dir}/payload_')
    assert len(encoded_payloads[2].data) == 0

    payloads = await codec.decode(encoded_payloads)
    assert payloads[0] == payload_small
    assert payloads[1] == payload_medium
    assert payloads[2] == payload_large
