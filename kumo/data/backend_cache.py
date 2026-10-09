# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import logging
import os.path as osp
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeAlias,
)

from cachetools import TTLCache
from kumo_core.types import ColStatType, TableStatType, persisted_type_token
from kumo_core.types.schema import (
    BaseColumnConfig,
    SourceTableID,
    TableFingerprint,
    VersionedSourceTableID,
    time_col_config_to_fp,
)
from wrapt import synchronized

from async_utils.sync import FutureDedupeOption, blocking_method
from io_utils import fsutil, kumo_dataclass
from io_utils.json import fast_from_json, load_json, save_json
from kumo.config.graph import GraphConfig
from kumo.data.cache import (
    CACHE_TTL,
    MAX_CACHE_SIZE,
    ExactStats,
    GraphCache,
    NoExactStatsFound,
    ReadThroughMaterializationStageCache,
    ReadThroughStatsCache,
    ReadThroughTableCache,
    ReadThroughTableSampleCache,
    TableCacheDir,
    TableVersionCache,
    parse_col_stat_from_key,
    table_source_path_component,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL, TableRefreshStage
from kumo.stats import Stats, StatsConfig
from kumo.stats.stats import APPROX_ONLY_COL_STATS
from kumo.validation.validation import ValidationResponse
from monitoring.tracing import TraceOptions

logger = logging.getLogger(__name__)

STATS_CACHE_FILE = 'stats_cache.json'
STAGES_CACHE_FILE = 'stages_cache.json'
CACHE_DIR = 'data_backend_cache'
PQUERY_TABLE_STATS = 'pquery_table_stats'
TRAIN_TABLE_SPLIT_STATS = 'train_table_split_stats'
DIRECT_DISKGRAPH_SOURCE_DECISION = 'direct_diskgraph_source_decision'


def _load_col_stats_json(file_path: str) -> Optional[Dict]:
    """Load full column stats, staging S3 objects on disk for decoding."""
    if fsutil.get_scheme(file_path) != 's3':
        return load_json(file_path, Dict, default=None, raise_on_missing=False)

    if not fsutil.exists(file_path):
        return None

    with fsutil.TemporaryDirectory() as temp_dir:
        local_path = osp.join(temp_dir, 'col_stats.json')
        download_start = time.perf_counter()
        try:
            fsutil.copy_file(file_path, local_path)
        except FileNotFoundError:
            return None
        download_seconds = time.perf_counter() - download_start

        decode_start = time.perf_counter()
        value = load_json(local_path, Dict)
        decode_seconds = time.perf_counter() - decode_start
        logger.info(
            'Loaded full column stats from %s via managed download '
            '(bytes=%d, download_seconds=%.3f, decode_seconds=%.3f)',
            file_path,
            osp.getsize(local_path),
            download_seconds,
            decode_seconds,
        )
        return value


class AsyncDataTaskStatus(str, Enum):
    NOT_STARTED = 'not_started'
    IN_PROGRESS = 'in_progress'
    DONE = 'done'
    FAILED = 'failed'

    @property
    def completed(self) -> bool:
        return self in [AsyncDataTaskStatus.DONE, AsyncDataTaskStatus.FAILED]


@kumo_dataclass
class TableDataTaskResult:
    # final status of the task
    status: AsyncDataTaskStatus
    # aggregate validation over all sub-steps
    validation: ValidationResponse
    # stage where data task ended
    stage: TableRefreshStage
    # workflow activity error if a step failed
    error: Optional[str] = None
    # Data version of the table.
    data_version: str = ''

    # Explicitly records that a terminal task intentionally omitted column
    # statistics. ``None`` preserves the legacy meaning of COMPLETED for
    # results written before this field existed.
    col_stats_computed: Optional[bool] = None

    # Last updated timestamp
    updated_at: Optional[datetime] = None

    # Created timestamp
    created_at: Optional[datetime] = None

    @property
    def ok(self) -> bool:
        return (self.status == AsyncDataTaskStatus.DONE and not self.error
                and self.validation.ok)

    @property
    def full_col_stats_computed(self) -> bool:
        if self.col_stats_computed is not None:
            return self.col_stats_computed
        if self.stage == TableRefreshStage.COMPLETED:
            return True
        return (self.stage == TableRefreshStage.COMPUTE_FULL_COL_STATS
                and self.status == AsyncDataTaskStatus.DONE)

    @property
    def min_col_stats_computed(self) -> bool:
        return self.full_col_stats_computed or (
            self.stage == TableRefreshStage.COMPUTE_MIN_COL_STATS
            and self.status == AsyncDataTaskStatus.DONE)


@kumo_dataclass
class TableIngestionInfo:
    """Debug information about a single table ingestion job, which is saved to
    S3 at the end of the ingest workflow. Some of these fields are exposed in
    the UI for MLOps purposes.
    """
    ingest_start_time: datetime
    ingest_end_time: datetime
    temporal_workflow_id: str
    code_version: str


@kumo_dataclass(frozen=True)
class DirectDiskgraphSourceDecision:
    """The file-source decision frozen for one table data version.

    ``source=None`` deliberately records the staged path. A missing cache
    entry is different: it belongs to a refresh created before this contract
    existed and may use the temporary compatibility fallback.
    """
    source: Optional[Dict[str, Any]]


ColumnName: TypeAlias = str
DataVersionResolver: TypeAlias = Callable[[Any, str], str]


@dataclass(frozen=True)
class PredStats:
    """All stats on a prediction table
    """
    col_stats: Dict[ColumnName, Dict[ColStatType, StatsConfig]]

    @staticmethod
    def parse(table_id: VersionedSourceTableID, col_json: bytes) -> 'PredStats':
        """Parse exact col stats from the given JSON files. Raise exception if
        exact stats are missing or failed to be parsed, for col stats only
        since prediction tables don't have table stats.
        """
        col_dict = fast_from_json(col_json, Dict[str, StatsConfig])
        col_stats: Dict[ColumnName, Dict[ColStatType,
                                         StatsConfig]] = defaultdict(dict)
        for key, stat in col_dict.items():
            try:
                col_name, stat_type, approx = parse_col_stat_from_key(key)
                if approx and stat_type not in APPROX_ONLY_COL_STATS:
                    continue
                col_stats[col_name][stat_type] = stat
            except ValueError as e:
                # If the stats in S3 have a stat type that doesn't exist
                # anymore don't return it but return the rest instead of
                # failing the whole parse function
                logger.exception(
                    'Unable to parse table %s key %s from exact stats, %s',
                    table_id, key, e)
        if len(col_stats) == 0:
            raise NoExactStatsFound(
                "No exact stats could be found in the col_json")

        return PredStats(col_stats)


# Unique key for column stats formed from from col name, version, and stat
# type.
ColStatKey: TypeAlias = str


def col_stat_key(
    col_config: BaseColumnConfig,
    stat_type: ColStatType,
    approx: bool,
) -> ColStatKey:
    # N.B. col_config.stype is Optional
    if col_config.stype is None:
        return "__".join([col_config.name, stat_type.value, str(approx)])
    return "__".join([
        col_config.name,
        persisted_type_token(col_config.stype), stat_type.value,
        str(approx)
    ])


def table_stat_key(stat_type: TableStatType, approx: bool = False) -> str:
    return "__".join([stat_type.value, str(approx)])


STAGE_KEY = "stage"

# Dicts for stats indexed by ColStatKey
ColStatsDict: TypeAlias = Dict[ColStatKey, StatsConfig]


class _CacheableTable(Protocol):
    """Structural contract for inputs to cache methods that key by
    ``source_table_id`` and ``fingerprint()``.

    Satisfied by ``rest.config.table_v2.TableDefinition``. Declared as a
    Protocol so this module does not need to import from ``rest.config``.
    """
    @property
    def source_table_id(self) -> SourceTableID:
        ...

    def fingerprint(self) -> TableFingerprint:
        ...


# Forwarder; canonical home is ``kumo.data.get_active_cache``.
def get_active_cache() -> 'BackendCacheManager':
    """Forwarder for :func:`kumo.data.get_active_cache`."""
    from kumo.data import get_active_cache as _gac
    return _gac()


class BackendCacheManager:
    """
    Class manages caching of backend data to S3 in json format. Includes stats
    and materialization stage.
    """
    def __init__(self, backend_cache_dir: str) -> None:
        self._cache_dir = fsutil.s3a_as_s3(backend_cache_dir)
        fsutil.makedirs(self._cache_dir, exist_ok=True)

        # These dicts are typed for json writeability (no indexing by tuples
        # using configs for the stats) - instance variables since the cache
        # instance is a class variable of the owner backend

        # Everything about the graph information.
        self.graph_cache = GraphCache(
            cache_dir=osp.join(backend_cache_dir, 'graphs'))

        # op_id is a unique ID per execution of a pquery table generation op,
        # identifying the generated pquery tables.
        # op_id --> stat name (see StaticTableStats/TemporalTableStats) --> stat
        self.pquery_table_stats_cache = ReadThroughStatsCache(
            cache_dir=osp.join(backend_cache_dir, 'train_tables'),
            name=PQUERY_TABLE_STATS,
            # Generated in one job and read in another, so skip in-memory cache.
            capacity=0)

        # op_id --> [NUM_EXPECTED_ITERATIONS/ITERATION_DATA] --> stat
        self.pquery_table_progress_cache = ReadThroughStatsCache(
            cache_dir=osp.join(backend_cache_dir, 'train_tables'),
            name='table_progress', capacity=0)

        # job_id -> stat name -> stat
        self.training_table_split_stats_cache = ReadThroughStatsCache(
            cache_dir=osp.join(backend_cache_dir, 'train_tables'),
            name=TRAIN_TABLE_SPLIT_STATS, capacity=0)

        # source name/connector/version --> stat type --> stat
        self._table_stats_cache = ReadThroughStatsCache(
            cache_dir=osp.join(backend_cache_dir, 'tables'), name="table_stats",
            capacity=MAX_CACHE_SIZE)

        # source name/connector/version --> ColStatsDict
        self._col_stats_cache = TTLCache(MAX_CACHE_SIZE, CACHE_TTL)
        self._min_col_stats_cache = TTLCache(MAX_CACHE_SIZE, CACHE_TTL)

        # set of tables for each cache type that have been modified since
        # the last persist to file
        self._col_stats_modified_since_save: Set[str] = set()

        self.materialization_stage = ReadThroughMaterializationStageCache(
            backend_cache_dir, name='materialization_stage')

        # source name/connector/version --> pd.DataFrame
        self.table_samples_cache = ReadThroughTableSampleCache(
            backend_cache_dir, name='table_samples')

        self.table_version_cache = TableVersionCache(backend_cache_dir)

        self.table_refresh_result = ReadThroughTableCache[TableDataTaskResult](
            backend_cache_dir, name='refresh_task_result')
        self.ingestion_info = ReadThroughTableCache[TableIngestionInfo](
            backend_cache_dir, name='ingestion_info')
        self.direct_diskgraph_source_decision = ReadThroughTableCache[
            DirectDiskgraphSourceDecision](
                backend_cache_dir, name=DIRECT_DISKGRAPH_SOURCE_DECISION)

    def col_stats_cache_size(self) -> int:
        ret = 0
        for k in self._col_stats_cache:
            ret += len(self._col_stats_cache[k])
        return ret

    def update_cache_dir(self, backend_cache_dir: str):
        self._cache_dir = fsutil.s3a_as_s3(backend_cache_dir)
        fsutil.makedirs(self._cache_dir, exist_ok=True)
        self.graph_cache.reset(osp.join(self._cache_dir, 'graphs'))
        self.pquery_table_progress_cache.reset(
            osp.join(self._cache_dir, 'train_tables'))
        self.pquery_table_stats_cache.reset(
            osp.join(self._cache_dir, 'train_tables'))
        self.training_table_split_stats_cache.reset(
            osp.join(self._cache_dir, 'train_tables'))
        self._table_stats_cache.reset(osp.join(self._cache_dir, 'tables'))
        self.table_samples_cache.reset(self._cache_dir)
        self.table_version_cache.reset(self._cache_dir)
        self.table_refresh_result.reset(self._cache_dir)
        self.ingestion_info.reset(self._cache_dir)
        self.direct_diskgraph_source_decision.reset(self._cache_dir)

    @staticmethod
    def _get_col_stats_cache_key(table_fingerprint: TableFingerprint,
                                 data_version: str) -> str:
        """Create a cache key from TableFingerprint and data_version.

        Args:
            table_fingerprint: The table fingerprint
            data_version: The data version

        Returns:
            The cache key string which depends on connector name, source name,
            primary key(optional), and time column(optional). This along
            with a columns type form a unique key for the col stats cache.
        """
        key_prefix = (
            table_fingerprint.connector_name[:4] + '.' +
            table_source_path_component(table_fingerprint.source_name)[:12])
        # Adding a hash prefix to ensure that the key is unique even if the
        # connector name and source name are long.
        hash_prefix = hashlib.md5(
            (table_fingerprint.connector_name + '.' +
             table_fingerprint.source_name).encode()).hexdigest()
        pkey = getattr(table_fingerprint, 'pkey', None)
        time_col = time_col_config_to_fp(
            getattr(table_fingerprint, 'time_col', None))
        if pkey is not None:
            key_prefix += f".{pkey}"
        if time_col is not None:
            key_prefix += f".{time_col}"
        return f"{key_prefix}_{hash_prefix}_{data_version}"

    @staticmethod
    def _get_table_stats_key(table_fingerprint: TableFingerprint,
                             data_version: str) -> str:
        """A key for the table stats cache, which has stats of type
        `TableStatType`.
        """
        return osp.join(
            table_fingerprint.connector_name,
            table_source_path_component(table_fingerprint.source_name),
            data_version)

    @synchronized
    def get_missing_col_stats(
        self,
        table_fingerprint: TableFingerprint,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        data_version: str,
        approx: bool,
    ) -> Dict[BaseColumnConfig, List[ColStatType]]:
        cache_key = BackendCacheManager._get_col_stats_cache_key(
            table_fingerprint, data_version)
        logger.debug('Getting missing col stats for '
                     f'{table_fingerprint.connector_name}.'
                     f'{table_fingerprint.source_name}.{data_version}')
        if not self._col_stats_cache.get(cache_key):
            return stats_dict
        return_dict = {}
        for col_config, stat_list in stats_dict.items():
            new_list = []
            for stat_type in stat_list:
                stat_key_approx = col_stat_key(col_config, stat_type,
                                               approx=True)
                stat_key_exact = col_stat_key(col_config, stat_type,
                                              approx=False)
                # either approx or exact can be returned for an approx request
                if approx or stat_type in APPROX_ONLY_COL_STATS:
                    if (self._col_stats_cache[cache_key].get(stat_key_exact)
                            is None and self._col_stats_cache[cache_key].get(
                                stat_key_approx) is None):
                        new_list.append(stat_type)
                # if exact is requested, stat is missing if exact not found
                else:
                    if self._col_stats_cache[cache_key].get(
                            stat_key_exact) is None:
                        new_list.append(stat_type)

            if len(new_list) > 0:
                return_dict[col_config] = new_list
        return return_dict

    @synchronized
    def add_col_stats(
        self,
        table_fingerprint: TableFingerprint,
        stats_dict: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
    ):
        for col_config, stat_dict in stats_dict.items():
            if not stat_dict:
                continue
            # note: this should be same for all cols, but it's cleaner to get
            # here than handle some column keys having an empty dict as value
            # at the top level
            data_version = _get_stats_data_version(stat_dict)
            cache_key = BackendCacheManager._get_col_stats_cache_key(
                table_fingerprint, data_version)
            self._col_stats_modified_since_save.add(cache_key)
            if cache_key not in self._col_stats_cache:
                self._col_stats_cache[cache_key] = {}
            for stat_type, stat in stat_dict.items():
                stat_key = col_stat_key(col_config, stat_type, stat.approx)
                self._col_stats_cache[cache_key][stat_key] = stat.config()
        logger.info(f"ColStatsCache size {self.col_stats_cache_size()}")

    def _get_col_stats(
        self,
        cache_key: str,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        col_stats_cache: Dict[str, ColStatsDict],
        approx: bool,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        return_dict = {}
        for col_config, stat_list in stats_dict.items():
            new_dict = {}
            for stat_type in stat_list:
                # first, try to get and return the exact stat
                stat_key_exact = col_stat_key(col_config, stat_type,
                                              approx=False)
                stat_exact = col_stats_cache[cache_key].get(stat_key_exact)
                if stat_exact is not None:
                    new_dict[stat_type] = Stats.from_config(stat_exact)
                # second, if approx requested try getting approx
                elif approx or stat_type in APPROX_ONLY_COL_STATS:
                    stat_key_approx = col_stat_key(col_config, stat_type,
                                                   approx=True)
                    stat_approx = col_stats_cache[cache_key].get(
                        stat_key_approx)
                    if stat_approx is not None:
                        new_dict[stat_type] = Stats.from_config(stat_approx)

            if new_dict:
                return_dict[col_config] = new_dict
        return return_dict

    @synchronized
    def get_col_stats(
        self,
        table_fingerprint: TableFingerprint,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        data_version: str,
        approx: bool,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        cache_key = BackendCacheManager._get_col_stats_cache_key(
            table_fingerprint, data_version)
        if not self._col_stats_cache.get(cache_key):
            logger.info(f"Cols Stats cache miss for {cache_key}")
            return {}
        return self._get_col_stats(cache_key, stats_dict, self._col_stats_cache,
                                   approx)

    @synchronized
    def get_min_col_stats(
        self,
        table_fingerprint: TableFingerprint,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        data_version: str,
        approx: bool,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        cache_key = BackendCacheManager._get_col_stats_cache_key(
            table_fingerprint, data_version)
        if not self._min_col_stats_cache.get(cache_key):
            return {}
        return self._get_col_stats(cache_key, stats_dict,
                                   self._min_col_stats_cache, approx)

    def load_min_col_stats_cache(self, connector_name: str,
                                 table_fingerprint: TableFingerprint,
                                 data_version: str) -> bool:
        cache_key = BackendCacheManager._get_col_stats_cache_key(
            table_fingerprint, data_version)
        table_dir = self.create_table_cache_dir(connector_name,
                                                table_fingerprint.source_name,
                                                data_version, table_fingerprint)
        if not fsutil.exists(table_dir._table_dir):
            logger.warning(f'Table cache not found at {table_dir._table_dir}; '
                           'popping applicable caches col_stats')
            self._min_col_stats_cache.pop(cache_key, None)
            return False

        file_path = table_dir.min_col_stats
        value = load_json(file_path, Dict, default=None, raise_on_missing=False)

        if value is not None:
            logger.info(
                f'Loading min col stats for {cache_key} from {file_path},'
                f'new cache size {len(self._min_col_stats_cache) + 1}')
            self._min_col_stats_cache[cache_key] = value
        else:
            logger.warning(f'No min col stats found at {file_path}')
            self._min_col_stats_cache.pop(cache_key, None)
            return False

        return True

    def load_col_stats_cache(self, table_fingerprint: TableFingerprint,
                             data_version: str) -> bool:
        cache_key = BackendCacheManager._get_col_stats_cache_key(
            table_fingerprint, data_version)
        table_dir = self.create_table_cache_dir(
            table_fingerprint.connector_name, table_fingerprint.source_name,
            data_version, table_fingerprint)
        if not fsutil.exists(table_dir._table_dir):
            logger.warning(f'Table cache not found at {table_dir._table_dir}; '
                           'popping applicable caches col_stats')
            self._col_stats_cache.pop(cache_key, None)
            return False

        file_path = table_dir.col_stats
        value = _load_col_stats_json(file_path)
        if value is not None:
            logger.info(f'Loading col stats for {cache_key} from {file_path}'
                        f'new cache size {len(self._col_stats_cache) + 1}')
            self._col_stats_cache[cache_key] = value
        else:
            logger.warning(f'No col stats found at {file_path}')
            self._col_stats_cache.pop(cache_key, None)
            return False
        logger.info(f"ColStatsCache size {self.col_stats_cache_size()}")

        return True

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def get_many_min_exact_stats(
        self,
        versioned_table_fingerprint_list: List[Tuple[VersionedSourceTableID,
                                                     TableFingerprint]],
        *,
        skip_col_stats: bool = False,
    ) -> List[Optional[ExactStats]]:
        """Get exact table and col stats for multiple tables. If any exact
        stats are missing/not computed for a particular table, then return None
        for that table.

        Args:
            skip_col_stats (bool): whether skip load col stats.
                (default: :obj:`False`)
        """
        # Fetch data in parallel from S3
        table_stats_paths: List[str] = []
        min_stats_paths: List[str] = []
        for (table, fingerprint) in versioned_table_fingerprint_list:
            cache = self.create_table_cache_dir(table.connector_name,
                                                table.source_table_name,
                                                table.version, fingerprint)
            table_stats_paths.append(cache.table_stats)
            if not skip_col_stats:
                min_stats_paths.append(cache.min_col_stats)

        col_stats_dict: Dict = {}
        tables = [table for table, _ in versioned_table_fingerprint_list]
        if not skip_col_stats:
            # If require only min col stats, fetch min_col_stats first and
            # fetch col_stats for tables that min_col_stats does not exists.
            min_stats_data = fsutil.get_many_if_present(min_stats_paths)
            for i, table_id in enumerate(tables):
                col_stats_dict[table_id] = min_stats_data[i]

        # Fetch table_stats for all tables.
        table_stats = fsutil.get_many_if_present(table_stats_paths)

        # Parse the resulting JSON files
        results: List[Optional[ExactStats]] = []
        for table_id, table_json in zip(tables, table_stats):
            col_json = col_stats_dict.get(table_id)

            # Skip over any tables that are missing stats
            if table_json is None or (col_json is None and not skip_col_stats):
                results.append(None)
                continue

            # Parse the stats data
            try:
                stats = ExactStats.parse(table_id, table_json, col_json)
            except NoExactStatsFound:
                # This may happen if table ingestion is not done, so we can
                # silently ignore it.
                results.append(None)
            except Exception as e:
                # All other errors are indicative of real code bugs
                logger.exception("Unexpected error when parsing exact stats %s",
                                 e)
                results.append(None)
            else:
                results.append(stats)

        return results

    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def get_many_pred_stats(
            self, pred_tables: Dict[str,
                                    SourceTableID]) -> Dict[str, PredStats]:
        """Get a PredStats instance (column stats) for each job in
        ``pred_tables``, keyed by job id.

        Jobs without available prediction table stats are omitted from the
        result.
        """
        # 1. get VersionedSourceTableID for each table
        # TODO (Alex): either remove this and pass in the table version, or
        # get all versions via batch operation
        tables = []
        job_ids = []
        for job_id, source_table in pred_tables.items():
            version = self.table_version_cache.get_for_job(
                job_id, source_table.connector_name,
                source_table.source_table_name)
            # even if table is not ready yet and doesn't have a version, add
            # placeholder in tables
            if version:
                tables.append(
                    VersionedSourceTableID(source_table.connector_name,
                                           source_table.source_table_name,
                                           version))
                job_ids.append(job_id)

        # 2. get cache paths
        col_stats_paths = []
        for id in tables:
            col_stats_paths.append(
                self.create_table_cache_dir(id.connector_name,
                                            id.source_table_name,
                                            id.version).col_stats)

        # 3. get data
        col_stats_data = fsutil.get_many_if_present(col_stats_paths)

        # 4. parse
        results = {}
        for table_id, col_json, job_id in zip(tables, col_stats_data, job_ids):
            if not col_json:
                continue
            # Parse the stats data
            try:
                stats = PredStats.parse(table_id, col_json)
            except NoExactStatsFound:
                # This may happen if bp stats are not done, so ignore it.
                pass
            except Exception as e:
                # All other errors are indicative of real code bugs
                logger.exception(
                    "Unexpected error when parsing pred table stats %s", e)
            else:
                results[job_id] = stats

        return results

    @synchronized
    def add_table_stats(
        self,
        table_fingerprint: TableFingerprint,
        stats_dict: Dict[TableStatType, Stats],
    ):
        """Merge ``stats_dict`` into the table stats cache, overwriting any
        existing entries for the same keys.
        """
        if not stats_dict:
            return

        data_version = _get_stats_data_version(stats_dict)
        table_key = self._get_table_stats_key(table_fingerprint, data_version)

        stats_dict_str = {
            table_stat_key(k, v.approx): v
            for k, v in stats_dict.items()
        }
        self._table_stats_cache.update(table_key, stats_dict_str)

    @synchronized
    def get_table_stats(
        self,
        table_fingerprint: TableFingerprint,
        stats_list: List[TableStatType],
        data_version: str,
        approx: bool = False,
    ) -> Dict[TableStatType, Stats]:
        table_key = self._get_table_stats_key(table_fingerprint, data_version)
        in_memory_stats = self._table_stats_cache.has(table_key)
        stats = self._table_stats_cache.get(table_key, Dict[str, StatsConfig])

        missing_stats = [
            k for k in stats_list if table_stat_key(k) not in stats and (
                not approx or table_stat_key(k, approx=True) not in stats)
        ] if stats else stats_list
        if len(missing_stats) > 0 and in_memory_stats:
            # Load again to check if there's update in backend storage.
            stats = self._table_stats_cache.load(table_key, Dict[str,
                                                                 StatsConfig])

        if not stats:
            return {}

        return_dict = {}
        for stat_type in stats_list:
            stat = stats.get(table_stat_key(stat_type))
            if stat is None and approx:
                stat = stats.get(table_stat_key(stat_type, approx=True))
            if stat is not None:
                return_dict[stat_type] = Stats.from_config(stat)
        return return_dict

    @synchronized
    def save_table_stats(self, table_fingerprint: TableFingerprint,
                         data_version: str, value: Dict[TableStatType, Stats]):
        table_key = self._get_table_stats_key(table_fingerprint, data_version)
        save_value = {
            table_stat_key(k, v.approx): v.config()
            for k, v in value.items()
        }
        self._table_stats_cache.save(table_key, save_value)

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def save_all_table_caches(
        self,
        graph_config: GraphConfig,
        *,
        data_version_resolver: Optional[DataVersionResolver] = None,
    ):
        """Persist graph table caches without constructing connectors.

        Graph refresh normally sets ``table.data_version`` before this call.
        Legacy callers with unresolved versions must provide the connector
        adapter explicitly through ``data_version_resolver``.
        """
        logger.info('Saving all modified table caches for graph'
                    f' {graph_config.name}')
        for table in graph_config.tables:
            data_version = table.data_version
            if data_version is None:
                if data_version_resolver is None:
                    raise ValueError(
                        "table data versions must be resolved before cache "
                        f"persistence; missing version for {table.source_name!r}"
                    )
                connector_config = graph_config.data_source_dict[
                    table.just_data_source]
                data_version = data_version_resolver(connector_config,
                                                     table.source_name)
            # Create or get table fingerprint
            table_fingerprint = table.fingerprint() if hasattr(
                table, 'fingerprint') else table.partial_fingerprint()
            self.save_col_stats_cache(table_fingerprint, data_version)
        logger.info('Finished saving all modified table caches for graph'
                    f' {graph_config.name}')

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def save_col_stats_cache(
        self,
        table_fingerprint: TableFingerprint,
        data_version: str,
    ) -> str:
        """Persist column stats for a source table, returning the full stats
        file path.

        Saves only when stats exist and are dirty. Uses read-merge-write
        because the same table key may be computed and saved concurrently by
        multiple workflows. A minimized copy is also written alongside.
        """
        cache_key = BackendCacheManager._get_col_stats_cache_key(
            table_fingerprint, data_version)
        connector_name = table_fingerprint.connector_name
        source_name = table_fingerprint.source_name
        table_dir = self.create_table_cache_dir(connector_name, source_name,
                                                data_version)
        logger.info(
            f"Requested backend cache save_col_stats_cache for key={cache_key}"
            f" and directory={table_dir._table_dir}.")

        stats_dict = self._col_stats_cache.get(cache_key)
        dirty_bit = cache_key in self._col_stats_modified_since_save
        if stats_dict and dirty_bit:
            file_path = table_dir.col_stats

            # TODO: read-merge-write still does not handle the case when
            # a column changes its dtype as result stype change. This issue
            # is tracked in https://github.com/kumo-ai/kumo/issues/6737
            #
            # As of now, col stats of the same source table key could be
            # concurrently computed and saved by multiple workflows. To
            # minimize data racing, we perform a read-merge-write here.
            exist_stats = load_json(file_path, Dict, default={},
                                    raise_on_missing=False)
            if len(exist_stats) > 0:
                logger.info(f"Read {len(exist_stats)} stats from {file_path}, "
                            f"performing a read-merge-write with "
                            f"{len(stats_dict)} new stats.")
            # TODO(dong) Separate min stats from full stats so partial writes
            # cannot replace richer existing stats for the same table key.
            merged_stats = exist_stats
            merged_stats.update(stats_dict)  # type: ignore

            logger.info(
                f"Saving column stats: key={cache_key}, path={file_path}")
            save_json(merged_stats, file_path)
            self._col_stats_modified_since_save.remove(cache_key)

            # Save min_col_stats when save col stats.
            _minify_stats_and_save(merged_stats, table_dir.min_col_stats)
        else:
            logger.info(f"Column statistics were requested to be saved to the "
                        f"backend cache, but were not saved either due to no "
                        f"modification or no statistics available. Debug info: "
                        f"key={cache_key}, dirty_bit={dirty_bit}.")
        return table_dir.col_stats

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=False),
                     fut_dedupe_option=FutureDedupeOption.DISABLED)
    def persist_sampled_col_stats(
        self,
        table_fingerprint: TableFingerprint,
        data_version: str,
        stats_dict: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
    ) -> None:
        """Best-effort persist of connector-sampled (approx) col stats (#28324).

        Read-merge-writes via ``save_col_stats_cache`` (no cross-pod lock);
        exact DiskGraph stats win on read and overwrite on ingest, so a stale
        sample is never served past ingestion. Never fails the caller -- log
        and swallow.
        """
        if not any(stats for stats in stats_dict.values()):
            return
        try:
            self.add_col_stats(table_fingerprint, stats_dict)
            self.save_col_stats_cache(table_fingerprint, data_version)
        except Exception as exc:
            logger.warning(
                "persist_sampled_col_stats: best-effort persist failed for "
                "%s@%s: %r", table_fingerprint.source_name, data_version, exc)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def load_all_table_caches(
        self,
        graph_config: GraphConfig,
        *,
        data_version_resolver: Optional[DataVersionResolver] = None,
        **kwargs,
    ) -> bool:
        """
        Convenient helper to load all tables in the graph. Optionally, specify
        additional kwargs to control which metadata/stats to load for all
        tables. This cache layer never imports or constructs connectors;
        unresolved versions require an explicit ``data_version_resolver``.
        """
        logger.info(f'Loading all table caches for graph {graph_config.name}')
        success_all = True
        for table in graph_config.tables:
            data_version = table.data_version
            if data_version is None:
                if data_version_resolver is None:
                    raise ValueError(
                        "table data versions must be resolved before cache "
                        f"loading; missing version for "
                        f"{table.just_source_name!r}")
                connector_config = graph_config.data_source_dict[
                    table.just_data_source]
                data_version = data_version_resolver(connector_config,
                                                     table.just_source_name)
            logger.info(f'Loading cache for table {table.just_source_name}'
                        ' from load_all_table_caches')
            # Create or get table fingerprint
            table_fingerprint = table.fingerprint()
            success = self.load_col_stats_cache(table_fingerprint, data_version,
                                                **kwargs)
            success_all = success_all and success
        logger.info(f'Finished load all table caches for graph '
                    f'{graph_config.name}')
        return success_all

    # TODO(siyang): Evaluate if there is a better approach to avoid copying
    # the train table stats.
    @blocking_method(DATA_BACKEND_THREAD_POOL)
    def copy_pquery_table_stats_cache(self, src_job_id: str, dst_job_id: str):
        """Copy train table stats from one job id to another.

        Train table stats are indexed by job id. When a standalone
        GenerateTrainTableJob produces them and a separate training job reuses
        the table, the stats must be copied from the generating job
        (``src_job_id``) to the training job (``dst_job_id``).

        A job that generated nothing wrote no stats, and there is nothing to
        copy. That is a no-op rather than a failure: every reader of this
        cache already answers ``None`` or ``{}`` for a key it does not hold
        (``read_through_cache.py:49``), while ``fsutil.copy`` raises
        ``FileNotFoundError`` for a source that is not there -- so without
        this guard the absence would fail the run rather than be read as the
        absence it is.
        """
        src_dir = osp.join(self.cache_dir, 'train_tables', src_job_id)
        dst_dir = osp.join(self.cache_dir, 'train_tables', dst_job_id)
        if not fsutil.exists(src_dir):
            logger.info('No train table stats at %s to copy', src_dir)
            return
        fsutil.makedirs(dst_dir, exist_ok=True)
        fsutil.copy(src_dir, dst_dir, overwrite=True, raise_on_exists=False)
        logger.info('Copied train table stats from %s to %s', src_dir, dst_dir)

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    def get_many_min_latest_exact_stats(
        self,
        table_args_list: List[_CacheableTable],
        fetch_column_stats: bool = True,
    ) -> List[Optional[ExactStats]]:
        """For each of the provided tables, get the exact table and col stats
        for the most recent version of the table in the cache. If the stats are
        not available, then return None. This may happen for tables that are
        not ingested yet.
        """
        tables = [table.source_table_id for table in table_args_list]
        versions = self.table_version_cache.get_many(tables)

        stats = self.get_many_min_exact_stats(
            [(k, v.fingerprint())
             for k, v in zip(versions, table_args_list) if k is not None],
            skip_col_stats=not fetch_column_stats,
        )
        # Since we did not fetch stats for tables with no versions, we need to
        # insert None in the result for each table with no verion.
        results: List[Optional[ExactStats]] = []
        i = 0
        for version in versions:
            if version is not None:
                results.append(stats[i])
                i += 1
            else:
                results.append(None)

        return results

    @synchronized
    def reset(self):
        self.graph_cache.reset()
        self._table_stats_cache.reset()
        self._col_stats_modified_since_save = set()
        self._col_stats_cache = TTLCache(MAX_CACHE_SIZE, CACHE_TTL)
        self._min_col_stats_cache = TTLCache(MAX_CACHE_SIZE, CACHE_TTL)
        self.pquery_table_progress_cache.reset()
        self.pquery_table_stats_cache.reset()
        self.training_table_split_stats_cache.reset()
        self.table_version_cache.reset()
        self.table_samples_cache.reset()
        self.materialization_stage.reset()
        self.table_refresh_result.reset()
        self.ingestion_info.reset()
        self.direct_diskgraph_source_decision.reset()

    def create_table_cache_dir(
        self,
        connector_name: str,
        source_table_name: str,
        version: str,
        table_fingerprint: Optional[TableFingerprint] = None,
    ) -> TableCacheDir:
        table_dir = osp.join(self.cache_dir, 'tables', connector_name,
                             table_source_path_component(source_table_name),
                             version)
        fsutil.makedirs(table_dir, exist_ok=True)
        return TableCacheDir(table_dir, table_fingerprint)


def _get_stats_data_version(stats_dict: Dict[Any, Stats]) -> str:
    data_version = list(stats_dict.items())[0][1].data_version
    if data_version is None:
        raise RuntimeError(f"Stat is missing data version: {stats_dict}")
    return str(data_version)


# TODO(dong) this is a hacky code to create mini stats and this should be
# deleted as soon as we migrated to a long term metadata layer.
def _minify_stats(stats_dict: Optional[Dict]) -> Dict:
    if stats_dict is None:
        return {}

    CATEGORY_LIMIT = 20

    filtered_dict: dict = dict()
    for key, value in stats_dict.items():
        if ((key.endswith("CATEGORY_COUNTS__False")
             or key.endswith("CATEGORY_COUNTS__True")) and value is not None):
            if isinstance(value, dict):
                original_count = len(value['value'])
                new_val_obj = copy.deepcopy(value)
                new_val_obj['value'] = value[
                    'value'][:min(original_count, CATEGORY_LIMIT)]
                filtered_dict[key] = new_val_obj
            elif isinstance(value, list):
                filtered_dict[key] = value[:min(len(value), CATEGORY_LIMIT)]
        else:
            filtered_dict[key] = value
    return filtered_dict


def _minify_stats_and_save(stats_dict: Optional[Dict], path: str) -> Dict:
    filtered_dict = _minify_stats(stats_dict)
    save_json(filtered_dict, path)
    return filtered_dict
