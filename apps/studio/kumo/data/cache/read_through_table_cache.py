# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from typing import Generic, Optional, Tuple, Type, TypeVar

import pandas as pd
from cachetools import TTLCache
from kumo_core.types.schema import TableFingerprint
from wrapt import synchronized

from async_utils.sync import FutureDedupeOption, blocking_method
from io_utils import fsutil, parquet_util
from io_utils.json import load_json, load_many_json, save_json
from kumo.data.cache.table_cache_dir import (
    CACHE_TTL,
    MAX_CACHE_SIZE,
    table_source_key,
    table_source_path_component,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL, MaterializationStage
from monitoring.tracing import TraceOptions

logger = logging.getLogger(__name__)

V = TypeVar('V')


class ReadThroughTableCache(Generic[V]):
    """The implementation of read through cache focus on table.
    The cache is stored in the following format:

    {cache_dir}/tables/
        connector_name1/source_table_name1/data_version1/{name}
        connector_name2/source_table_name2/data_version2/{name}
    """
    def __init__(self, cache_dir: str, name: str):
        self._cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
        self._cache_dir = cache_dir
        self.name = name

    def _table_dir(self, connector_name: str, source_table_name: str,
                   data_version: str) -> str:
        return osp.join(self._cache_dir, 'tables', connector_name,
                        table_source_path_component(source_table_name),
                        data_version)

    def _load(self, connector_name: str, source_name: str, data_version: str,
              cls: Type[V] | None = None) -> V | None:
        assert cls is not None
        table_key = table_source_key(source_name, connector_name, data_version)
        table_dir = self._table_dir(connector_name, source_name, data_version)
        file_path = osp.join(table_dir, self.name)

        value = load_json(file_path, cls, default=None, raise_on_missing=False)
        if value is None:
            logger.warning(f'No {self.name} found at {file_path}')
            return None
        else:
            logger.info(f'Loading {self.name} for {table_key} from {file_path}')
            self._cache[table_key] = value
        return value

    def has(self, connector_name: str, source_name: str, data_version: str):
        return self.get_v1(connector_name, source_name,
                           data_version) is not None

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    @synchronized
    def get_v1(self, connector_name: str, source_name: str, data_version: str,
               cls: Type[V] | None = None) -> V | None:
        table_key = table_source_key(source_name, connector_name, data_version)
        if table_key not in self._cache:
            ret = self._load(connector_name, source_name, data_version, cls)
        else:
            ret = self._cache.get(table_key, None)
        return ret

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    @synchronized
    def get(self, fingerprint: TableFingerprint, data_version: str,
            cls: Type[V] | None = None) -> V | None:
        return self.get_v1(fingerprint.connector_name, fingerprint.source_name,
                           data_version, cls)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    @synchronized
    def get_many(self, tables: list[Tuple[str, str, Optional[str]]],
                 cls: Type[V]) -> list[Optional[V]]:
        valid_tables: list[Tuple[str, str, str]] = [(t[0], t[1], t[2])
                                                    for t in tables
                                                    if t[2] is not None]
        # [t for t in tables if t[2] is not None] offends the type checker
        valid_results = load_many_json([
            osp.join(self._table_dir(t[0], t[1], t[2]), self.name)
            for t in valid_tables
        ], cls)  # type: ignore
        results = []
        idx = 0
        for t in tables:
            if t[2] is None:
                results.append(None)
            else:
                results.append(valid_results[idx])
                idx += 1
        return results

    @synchronized
    def remove(self, connector_name: str, source_name: str, data_version: str):
        table_key = table_source_key(source_name, connector_name, data_version)
        if table_key in self._cache:
            del self._cache[table_key]

        table_dir = self._table_dir(connector_name, source_name, data_version)
        file_path = osp.join(table_dir, self.name)
        fsutil.rm(file_path, ignore_errors=True)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    @synchronized
    def save(self, connector_name: str, source_name: str, data_version: str,
             value: V):
        table_dir = self._table_dir(connector_name, source_name, data_version)
        fsutil.makedirs(table_dir, exist_ok=True)
        file_path = osp.join(table_dir, self.name)
        logger.info(f'Saving {self.name} for {connector_name}.'
                    f'{source_name}.{data_version} to {file_path}')
        save_json(value, file_path)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    @synchronized
    def put(self, connector_name: str, source_name: str, data_version: str,
            value: V, save: bool = False):
        table_key = table_source_key(source_name, connector_name, data_version)
        self._cache[table_key] = value
        logger.info(f'ReadThroughTableCache size {len(self._cache)}')

        if save:
            self.save(connector_name, source_name, data_version, value)

    def reset(self, cache_dir: str | None = None) -> None:
        if cache_dir is not None:
            self._cache_dir = cache_dir
        self._cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)


class ReadThroughTableSampleCache(ReadThroughTableCache):
    r"""The read through table cache specialized for table sample.
    The `_load` and `put` function are implemented via pandas and parquet_util.
    """
    def _load(self, connector_name: str, source_name: str, data_version: str,
              cls: V = pd.DataFrame) -> pd.DataFrame | None:
        table_key = table_source_key(source_name, connector_name, data_version)
        table_dir = self._table_dir(connector_name, source_name, data_version)
        file_path = osp.join(table_dir, self.name)

        logger.info(f"Loading table samples for {table_key} from {file_path}")
        if not fsutil.exists(file_path):
            logger.info(f'No table samples found at {file_path}')
            return None
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.warning(f'Failed to find or load table samples for '
                           f'{table_key} from {file_path}: {e}')
            return None

    def put(self, connector_name: str, source_name: str, data_version: str,
            value: pd.DataFrame, save: bool = False):
        table_key = table_source_key(source_name, connector_name, data_version)
        self._cache[table_key] = value
        logger.info(f'ReadThroughTableSampleCache size {len(self._cache)}')

        if save:
            table_dir = self._table_dir(connector_name, source_name,
                                        data_version)
            fsutil.makedirs(table_dir, exist_ok=True)
            file_path = osp.join(table_dir, self.name)
            logger.info(f'Saving {self.name} for {table_key} to {file_path}')
            parquet_util.write_df(value, file_path)


class ReadThroughMaterializationStageCache(ReadThroughTableCache):
    r"""Materialization stages keyed by the staged artifact identity."""
    @synchronized
    def get_stage(
        self,
        table_fingerprint: TableFingerprint,
        data_version: str,
        check_local: bool = True,
    ) -> MaterializationStage | None:
        """Return the saved materialization stage for a table.

        By default also verifies the local materialize path exists, so a
        state-cache desync (local files wiped while the JSON cache still
        claims INGEST) is treated as not-set. Deferred-download connectors
        stage with the provider until graph assembly downloads them; those
        callers must pass
        ``check_local=False`` to avoid spurious "Table ingestion failed"
        errors between ingest and download.
        """
        stage = self.get_v1(
            table_fingerprint.connector_name,
            table_fingerprint.ingest_path_fpid,
            data_version,
            cls=MaterializationStage,
        )
        if stage is None:
            return None

        if not check_local:
            return stage

        from kumo.data.utils import get_materialize_path
        if not fsutil.exists(
                get_materialize_path(table_fingerprint=table_fingerprint,
                                     data_version=data_version, stage=stage)):
            return None
        return stage

    @synchronized
    def put(self, table_fingerprint: TableFingerprint, data_version: str,
            stage: MaterializationStage, save: bool = False):
        connector_name = table_fingerprint.connector_name
        ingest_path_fpid = table_fingerprint.ingest_path_fpid
        table_key = table_source_key(ingest_path_fpid, connector_name,
                                     data_version)
        self._cache[table_key] = stage
        logger.info(
            f'ReadThroughMaterializationStageCache size {len(self._cache)}')

        if save:
            self.save(connector_name, ingest_path_fpid, data_version,
                      stage.value)

    @synchronized
    def remove_stage(self, table_fingerprint: TableFingerprint,
                     data_version: str) -> None:
        super().remove(
            table_fingerprint.connector_name,
            table_fingerprint.ingest_path_fpid,
            data_version,
        )
