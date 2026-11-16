# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from typing import Dict, Generic, List, Type, TypeVar

from cachetools import TTLCache
from wrapt import synchronized

from io_utils import fsutil
from io_utils.json import load_json, save_json
from kumo.data.cache.table_cache_dir import CACHE_TTL, MAX_CACHE_SIZE
from kumo.stats import Stats, StatsConfig

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


class ReadThroughCache(Generic[K, V]):
    """One implementation of read through cache based on disk storage.
    The cache is stored in the following format:

    cache_dir/
        key_1/name
        key_2/name

    For example, when we use it for session id, the folder structure
    is organized as follows:
    data_persist_dir/cache_dir/
        job_id_1/session_id
        job_id_2/session_id

    When there's a get miss, the cache will read through lazily. Otherwise
    it will rely on cached data.

    Args:
        capacity: If the capacity is set to 0, the in memory cache
            will not store any values. All reads/write will be done
            from/to persistent storage. If the capacity is set to -1,
            the cache will be used. Allowed values are 0 and -1.
            (default: -1).
    """
    def __init__(self, cache_dir: str, name: str,
                 capacity: int = MAX_CACHE_SIZE):
        self._cache = TTLCache(maxsize=capacity,
                               ttl=CACHE_TTL) if capacity > 0 else {}
        self._cache_dir = cache_dir
        self.name = name
        self._capacity = capacity

    def load(self, key: K, cls: Type[V]) -> V | None:
        file_path = osp.join(self._cache_dir, str(key), self.name)

        value = load_json(file_path, cls, default=None, raise_on_missing=False)
        if value is None:
            logger.warning(f'No {self.name} found at {file_path}')
            return None
        else:
            logger.info(f'Loading {self.name} for {key} from {file_path}')
            self.maybe_set_cache_value(key, value)
        return value

    def maybe_set_cache_value(self, key: K, value: V):
        r"""Set the cache value for a given key provided
        the capacity is not 0."""
        if self._capacity != 0:
            self._cache[key] = value
            logger.info(f"ReadThroughCache size {len(self._cache)}")

    @synchronized
    def get(self, key: K, cls: Type[V]) -> V | None:
        r"""Retrieve the cached value for a given key. If not exists, read from
        the cache directory."""
        if key not in self._cache or self._capacity == 0:
            return self.load(key, cls)
        return self._cache.get(key, None)

    def has(self, key: K) -> bool:
        return key in self._cache

    def save(self, key: K, value: V):
        dir_path = osp.join(self._cache_dir, str(key))
        if not fsutil.is_s3(dir_path) and not fsutil.is_gcs(dir_path):
            fsutil.makedirs(dir_path, exist_ok=True)
        file_path = osp.join(dir_path, self.name)
        logger.info(f'Saving {self.name} for {key} to {file_path}')
        save_json(value, file_path)

    @synchronized
    def put(self, key: K, value: V):
        self.maybe_set_cache_value(key, value)
        self.save(key, value)

    @synchronized
    def remove(self, key: K):
        # Clean up cached statements.
        file_path = osp.join(self._cache_dir, str(key), self.name)
        fsutil.rm(file_path, ignore_errors=True)

        if key in self._cache:
            self._cache.pop(key)

    def reset(self, cache_dir: str | None = None) -> None:
        if cache_dir is not None:
            self._cache_dir = cache_dir
        self._cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)

    def __len__(self):
        return len(self._cache)


class ReadThroughStatsCache(ReadThroughCache):
    """The implementation of read through cache focus on Stats.

    It works with dictionary of stats, support update stats.
    When put stats, we call `config` function to convert to StatsConfig.
    When get stats, we call `from_config` function to convert to Stats.
    """
    @synchronized
    def put(self, key: K, value: Dict[str, Stats]):
        r"""Put a dictionary of stats and overwrite the cache.
        """
        value = {k: v.config() for k, v in value.items()}
        self.maybe_set_cache_value(key, value)
        self.save(key, value)

    def maybe_update_cache_value(self, key: K, value: Dict[str, Stats]):
        r"""Update the cache value for a given key provided
        the capacity is not 0."""
        if self._capacity != 0:
            value_old = self._cache.get(key, {})
            value_old.update(value)
            self.maybe_set_cache_value(key, value_old)

    @synchronized
    def update(self, key: K, value: Dict[str, Stats]):
        r"""Update a dictionary of stats. Other stats remain unchanged.
        """
        # For special case that value support update function.
        value = {k: v.config() for k, v in value.items()}
        self.maybe_update_cache_value(key, value)

        value_old = self.load(key, Dict[str, StatsConfig])
        if value_old is not None:
            value_old.update(value)
            value = value_old
        self.save(key, value)

    @synchronized
    def get_stats(self, key: K,
                  stats_list: List[str] | None = None) -> Dict[str, Stats]:
        r"""Returns a stats dictionary for the key.
        """
        stats_dict = self.get(key, Dict[str, StatsConfig])
        if stats_dict is None:
            return {}

        if stats_list is None:
            return {
                key: Stats.from_config(stats_dict[key])
                for key in stats_dict.keys()
            }
        else:
            if not all(key in stats_dict for key in stats_list):
                stats_not_present = list(
                    filter(lambda key: key not in stats_dict, stats_list))
                logger.warning(f'Values for stats {stats_not_present} are not '
                               f'present in backend cache.')
            return {
                key: Stats.from_config(stats_dict[key])
                for key in stats_dict.keys() & stats_list
            }
