# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Iterable, List, Optional
from uuid import uuid4

import zstd
from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadCodec

from io_utils import logging
from io_utils.fsutil import read_binary, write_binary
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

# Payload size threshold to enable zstd compression.  Default 2KB
CODEC_COMPRESS_SIZE_THRESHOLD = EnvVar('CODEC_COMPRESS_SIZE_THRESHOLD', int,
                                       False, 2048)
# Payload size threshold to enable blobstore(e.g., s3) offloading.  Default 1MB
CODEC_OFFLOAD_SIZE_THRESHOLD = EnvVar('CODEC_OFFLOAD_SIZE_THRESHOLD', int,
                                      False, 1024 * 1024)
# TTL in days for offloaded payloads.  Default 7 days.
OFFLOAD_TTL_DAYS = EnvVar('CODEC_OFFLOAD_TTL_DAYS', int, False,
                          7).parse_value()
BINARY_COMPRESSION_ENCODING = b'binary/compressed'
BINARY_OFFLOADED_ENCODING = b'binary/offloaded'


def _maybe_compress(payload: Payload, size_threshold: int) -> Payload:
    if len(payload.data) >= size_threshold:
        binary_data = payload.SerializeToString(deterministic=True)
        compressed_data = zstd.compress(binary_data)
        log.debug('Zstd compression: %s bytes to %s bytes', len(binary_data),
                  len(compressed_data))
        return Payload(
            metadata={
                'encoding': BINARY_COMPRESSION_ENCODING,
                'compression': b'zstd'
            },
            data=compressed_data,
        )
    return copy.deepcopy(payload)


def _maybe_decompress(payload: Payload) -> Payload:
    if payload.metadata.get('encoding') == BINARY_COMPRESSION_ENCODING:
        return Payload.FromString(zstd.decompress(payload.data))
    return copy.deepcopy(payload)


def _maybe_offload(payload: Payload, payload_dir: str,
                   size_threshold: int) -> Payload:
    if len(payload.data) >= size_threshold:
        file_path = os.path.join(payload_dir, f'payload_{uuid4().hex}.bin')
        write_binary(file_path, payload.SerializeToString(deterministic=True),
                     ttl=timedelta(days=OFFLOAD_TTL_DAYS))
        log.info('Offload large payload of size %s to %s', len(payload.data),
                 file_path)
        return Payload(
            metadata={
                'encoding': BINARY_OFFLOADED_ENCODING,
                'offload_path': file_path.encode(),
            },
            data=b'',
        )
    return payload


def _maybe_load(payload: Payload) -> Payload:
    if payload.metadata.get('encoding') == BINARY_OFFLOADED_ENCODING:
        file_path = payload.metadata['offload_path'].decode()
        bytes = read_binary(file_path)
        log.info('Load offloaded payload of size %s from %s', len(bytes),
                 file_path)
        return Payload.FromString(bytes)
    return payload


_CODEC_THREADPOOL = ThreadPoolExecutor(max_workers=8,
                                       thread_name_prefix='codec-thread')


class CompressionCodec(PayloadCodec):
    def __init__(self, tmp_payloads_dir: Optional[str] = None) -> None:
        super().__init__()
        self._payload_dir = tmp_payloads_dir
        self._compress_size_threshold = (
            CODEC_COMPRESS_SIZE_THRESHOLD.parse_value())
        self._offload_size_threshold = (
            CODEC_OFFLOAD_SIZE_THRESHOLD.parse_value())

    async def encode(self, payloads: Iterable[Payload]) -> List[Payload]:
        loop = asyncio.get_event_loop()
        compression_tasks = [
            loop.run_in_executor(_CODEC_THREADPOOL, _maybe_compress, payload,
                                 self._compress_size_threshold)
            for payload in payloads
        ]
        payloads = await asyncio.gather(*compression_tasks)
        if self._payload_dir:
            offload_tasks = [
                loop.run_in_executor(_CODEC_THREADPOOL, _maybe_offload,
                                     payload, self._payload_dir,
                                     self._offload_size_threshold)
                for payload in payloads
            ]
            return await asyncio.gather(*offload_tasks)
        else:
            return payloads

    async def decode(self, payloads: Iterable[Payload]) -> List[Payload]:
        loop = asyncio.get_event_loop()
        load_tasks = [
            loop.run_in_executor(_CODEC_THREADPOOL, _maybe_load, payload)
            for payload in payloads
        ]
        payloads = await asyncio.gather(*load_tasks)
        decompression_tasks = [
            loop.run_in_executor(_CODEC_THREADPOOL, _maybe_decompress, payload)
            for payload in payloads
        ]
        return await asyncio.gather(*decompression_tasks)
