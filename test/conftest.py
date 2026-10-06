# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register fixtures defined under test/fixtures/ so they're discoverable
# session-wide without per-test imports. Currently exposes
# ``dg_indexed_financial_graph`` (see test/fixtures/dg_indexed_graph.py).
pytest_plugins = ["test.fixtures.dg_indexed_graph"]

import asyncio  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import functools  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections.abc import Callable  # noqa: E402
from typing import Dict  # noqa: E402
from typing import Generator  # noqa: E402
from typing import List  # noqa: E402
from typing import Optional  # noqa: E402
from typing import Union  # noqa: E402
from uuid import uuid4  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.catalog import PermissionsChange  # noqa: E402
from databricks.sdk.service.catalog import Privilege  # noqa: E402
from databricks.sdk.service.catalog import SecurableType  # noqa: E402
from databricks.sdk.service.catalog import VolumeType  # noqa: E402
from kumo_core import embedding_models  # noqa: E402
from kumo_core.glove import TokenEmbeddingTable  # noqa: E402
from kumo_core.train.mapping import NODE_INDEX_COLUMN  # noqa: E402
from kumo_core.types import TaskType  # noqa: E402
from kumoapi.encoder import Datetime  # noqa: E402
from kumoapi.encoder import Index  # noqa: E402
from kumoapi.encoder import MaxLogNumerical  # noqa: E402
from kumoapi.encoder import MultiCategorical  # noqa: E402
from kumoapi.encoder import Numerical  # noqa: E402
from kumoapi.encoder import TextEmbedding  # noqa: E402
from kumoapi.typing import Dtype, Stype  # noqa: E402
from snowflake.snowpark import Session as SnowparkSession  # noqa: E402

import kumo.data as data  # noqa: E402
import kumo.home  # noqa: E402
from io_utils import fsutil, kumo_dataclass  # noqa: E402
from io_utils.databricks import DatabricksConnectionInfo  # noqa: E402
from io_utils.fsutil.uri import UCURI  # noqa: E402
from kumo.connector import snowflake_connector  # noqa: E402
from kumo.connector.composite_file_connector import \
    CompositeFileConnector  # noqa: E402
from kumo.connector.csv_connector import CSVConnector  # noqa: E402
from kumo.connector.parquet_connector import ParquetConnector  # noqa: E402
from kumo.graph import column as graph_column  # noqa: E402
from kumo.graph.graph import Graph  # noqa: E402
from kumo.graph.table import Table  # noqa: E402
from kumo.task.task import Task  # noqa: E402
from kumo.testing import withDatabricksCredentials  # noqa: E402
from kumo.testing.generator import EdgeSpec, GraphSpec, TableSpec  # noqa: E402
from kumo.utils.embedding_models import embedding_model_store  # noqa: E402

# Set the test IAM role for get_temporary_aws_credential fallback. This
# allows tests that exercise warehouse → S3 flows (export/write) to
# obtain real STS credentials on dev VMs and CI runners (where IRSA
# OIDC tokens are not available). OIDC-migrated workflows override this
# with the dedicated preprod test role after configuring their scoped
# workflow identity; legacy workflows retain the existing default.
os.environ.setdefault('AWS_TEST_ROLE_ARN',
                      'arn:aws:iam::926922431314:role/unittest-iam-role')

_TEMPORAL_LOCAL_STARTUP_TIMEOUT_SECONDS = 30

# Production code resolves GloVe artifacts through deploy_env_config.json. Keep
# the public dataset fallback for tests of that configuration path; the
# session fixture below supplies small local artifacts to encoder tests.
os.environ.setdefault('CUSTOM_ARTIFACT_ROOT', 's3://kumo-public-datasets')

_ENABLE_FSUTIL_DEBUG_LOGS_ENV = 'KUMO_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS'
# Keep this in the environment so Kumo logging reinitialization in spawned
# test processes can preserve the pytest fsutil log default.
os.environ.setdefault(_ENABLE_FSUTIL_DEBUG_LOGS_ENV, '0')


@pytest.fixture(scope='session', autouse=True)
def configure_test_glove_artifacts(tmp_path_factory):
    artifacts = tmp_path_factory.mktemp('glove')
    tokens = [
        'get',
        'old',
        'hello',
        'world',
        'a',
        'test',
        'welcome',
        'goodbye',
        'this',
        'is',
        'text',
        'graph',
        'network',
        'greetings',
        'from',
        'europe',
        'wow',
        'cool',
    ]
    pd.DataFrame({
        'index': tokens
    }).to_parquet(artifacts / 'glove_6B_index.parquet')
    np.save(
        artifacts / 'glove_6B_50d_emb.npy',
        np.ones((len(tokens), 50), dtype=np.float32),
    )
    # As in the images: the baked directory is named by KUMO_GLOVE_PATH.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('KUMO_GLOVE_PATH', str(artifacts))
        embedding_model_store.cache_clear()
        previous = embedding_models.set_default_store(embedding_model_store())
        yield
        embedding_models.set_default_store(previous)
    embedding_model_store.cache_clear()
    TokenEmbeddingTable.instances.clear()


def _env_var_enabled(name: str) -> bool:
    return os.environ.get(name, '').lower() in {'1', 'true', 'yes', 'on'}


# Set KUMO_PYTEST_ENABLE_FSUTIL_DEBUG_LOGS=1 in a local run or PR branch when
# debugging fsutil behavior and the DEBUG success traces are needed.


def pytest_addoption(parser):
    parser.addoption("--runnightly", action="store_true", default=False,
                     help="run nightly tests")
    parser.addoption('--runsnowflake', action='store_true', default=False,
                     help="run snowflake tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runnightly"):
        skip_nightly = pytest.mark.skip(
            reason="need --runnightly option to run")
        for item in items:
            if "nightly" in item.keywords:
                item.add_marker(skip_nightly)

    # check if you got an option like --key=snowflake
    if not config.getoption("--runsnowflake"):
        skip_snowflake = pytest.mark.skip(
            reason="need --runsnowflake option to run")
        for item in items:
            if "snowflake" in item.keywords:
                item.add_marker(skip_snowflake)
    else:
        skip_snowflake = pytest.mark.skip(reason="skip because not snowflake")
        for item in items:
            if ("snowflake" not in item.keywords
                    and 'enable_snowflake_test' not in item.fixturenames):
                item.add_marker(skip_snowflake)


# On test failure, dump subprocess stacks (py-spy) and full Temporal
# workflow histories to $ARTIFACTS_DIR/<nodeid>/. The CI workflow
# uploads that directory as an artifact. See test/_flake_diagnostics.py.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    # Setup failures almost always mean fixtures didn't come up and
    # there is no temporal_client to dump anything from.
    if rep.failed and rep.when in ("call", "teardown"):
        from test import _flake_diagnostics
        tc = item.funcargs.get("temporal_client")
        _flake_diagnostics.run_on_failure(item.nodeid, tc)


def pytest_configure(config):
    # register nightly marker
    config.addinivalue_line("markers",
                            ("nightly(name): Used to mark long-running"
                             " tests that are only run nightly."))
    config.addinivalue_line("markers",
                            ("snowflake(name): Used to mark snowflake test"))

    if not _env_var_enabled(_ENABLE_FSUTIL_DEBUG_LOGS_ENV):
        logging.getLogger('io_utils.fsutil').setLevel(logging.INFO)


@pytest.fixture(scope="session", autouse=True)
def close_test_state_dbs():
    from statedb.db import StateDB

    original_descriptor = StateDB.__dict__["open_sqlite"]
    original_open = StateDB.open_sqlite
    opened = []

    async def tracked_open(_cls, *args, **kwargs):
        state_db = await original_open(*args, **kwargs)
        opened.append(state_db)
        return state_db

    StateDB.open_sqlite = classmethod(tracked_open)
    try:
        yield
    finally:
        StateDB.open_sqlite = original_descriptor
        for state_db in reversed(opened):
            try:
                asyncio.run(state_db.close())
            except Exception:  # pylint: disable=broad-except
                logging.getLogger(__name__).exception(
                    "Failed to close a test StateDB sqlite pool")


def pytest_sessionfinish(session, exitstatus):
    try:
        from async_utils import event_loop_watchdog
        event_loop_watchdog._reset_global_watchdog_for_tests()
    except Exception:  # pylint: disable=broad-except
        logging.getLogger(__name__).exception(
            "Failed to stop event-loop watchdog during pytest teardown")


def _assert_loopback_sockets_available_for_temporal() -> None:
    socket_error = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError as exc:
        socket_error = repr(exc)
    if socket_error is not None:
        pytest.fail(
            "Temporal local test server cannot start because this environment "
            f"does not allow loopback TCP sockets: {socket_error}. Run these "
            "tests outside the socket-restricted sandbox, or set "
            "TEMPORAL_SERVER=cloud to use Temporal Cloud.",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def temporal_workflow_env():
    """Start a version-pinned Temporal dev server once per pytest session.

    Skipped when TEMPORAL_SERVER=cloud — tests connect to Temporal Cloud
    directly. Otherwise starts the SDK-bundled dev server (version-pinned),
    registers custom search attributes, and yields the environment.

    Intentionally a sync fixture so it does not require a session-scoped
    event loop. A dedicated event loop is used only for server startup and
    shutdown; individual async tests keep their own function-scoped loops.
    """
    if os.environ.get("TEMPORAL_SERVER", "").lower() == "cloud":
        yield None
        return

    _assert_loopback_sockets_available_for_temporal()

    from temporalio.api.enums.v1 import IndexedValueType
    from temporalio.api.operatorservice.v1 import AddSearchAttributesRequest
    from temporalio.testing import WorkflowEnvironment

    _SEARCH_ATTRS = {
        "instance_id": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
        "rpc_method": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
        "rpc_service": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD,
    }

    async def _start() -> WorkflowEnvironment:
        env = await WorkflowEnvironment.start_local(namespace="devtest")
        await env.client.operator_service.add_search_attributes(
            AddSearchAttributesRequest(
                search_attributes=_SEARCH_ATTRS,
                namespace="devtest",
            ))
        return env

    try:
        env = asyncio.run(
            asyncio.wait_for(
                _start(),
                timeout=_TEMPORAL_LOCAL_STARTUP_TIMEOUT_SECONDS,
            ))
    except asyncio.TimeoutError:
        pytest.fail(
            "Timed out after "
            f"{_TEMPORAL_LOCAL_STARTUP_TIMEOUT_SECONDS}s starting the local "
            "Temporal test server. Run these tests outside the "
            "socket-restricted sandbox, set TEMPORAL_SERVER=cloud, or check "
            "the cached Temporal CLI/dev-server environment.",
            pytrace=False,
        )
    # Required because some activities call AutoRpcClientConfig.parse_from_env()
    # which falls back to deploy_env() and reads global DeployEnvConfig singleton
    # for the Temporal address.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('TEMPORAL_ADDRESS',
                  env.client.service_client.config.target_host)
        try:
            yield env
        finally:
            asyncio.run(env.shutdown())


@pytest.fixture()
def enable_snowflake_test(request):
    return request.config.getoption("--runsnowflake")


# Fixture to mock time.time() and time.sleep() in tests
@pytest.fixture()
def mock_time(monkeypatch):
    class MockTime:
        def __init__(self):
            self.current_time = time.time()

        def time(self):
            return self.current_time

        def sleep(self, x):
            self.current_time += x

    mock_time = MockTime()
    monkeypatch.setattr(time, "time", mock_time.time)
    monkeypatch.setattr(time, "sleep", mock_time.sleep)
    yield mock_time


def filter_tables(graph: Graph, tables: Optional[List[str]]) -> Graph:
    # Used to reduce the number of tables inside a graph - useful to
    # accelerate tests in case not all tables of a graph are necessary.
    if tables is not None:
        for table in [t for t in graph.table_names if t not in tables]:
            graph.remove(table)
    return graph


def _create_graph(
    name: str,
    tables: Optional[List[str]] = None,
    infer: bool = False,
    root_dir: Optional[str] = None,
    parquet: bool = False,
    temporal: bool = False,
) -> Graph:
    r"""
    Args:
        tables (list, optional): list of tables to be included in the graph.
            Must use fully-qualified names.
        root_dir (str, optional): the directory where the graph data will be
            located. Example folder is :obj:`test/csv_data`.
    """
    graph_name = f"{name}{'_TEMPORAL' if temporal else ''}"

    if not parquet:
        connector_name = 'csv'
        if root_dir is not None:
            # For graph with customized root dir, we use the name of the graph
            # to avoid same connector points to different data.
            # TODO(dong) clean up customization after refactor data backend
            # cache to be source data dependent.
            root = osp.join(root_dir, name)
            connector_name = name
        else:
            root = osp.join(osp.dirname(osp.realpath(__file__)), 'csv_data',
                            name)
        graph = Graph(name=f'{graph_name}_csv')
        graph.connect(CSVConnector(connector_name, root, na_values='?'))
    else:
        # NOTE this is necessary since we need parquet connectors for the
        # training and prediction tables in the V2 API. Drop once our testing
        # infrastructure is cleaned up to better support persisting pandas
        # connectors:
        root = osp.join(osp.dirname(osp.realpath(__file__)), 'parquet_data',
                        name)
        graph = Graph(name=f'{graph_name}_parquet')
        graph.connect(ParquetConnector('parquet', root))

    graph.add_from_config(osp.join(root, 'metadata.json'))

    if name == 'FINANCIAL' and temporal:
        graph['ACCOUNT'].set_time_col('DATE')
        graph['LOAN'].set_time_col('DATE')
        graph['TRANS'].set_time_col('DATE')
    elif name == 'COUPON_PURCHASE_PREDICTION' and temporal:
        graph['COUPON_LIST'].set_time_col('DISPFROM')
        graph['COUPON_LIST'].set_end_time_col('DISPEND')
        graph['COUPON_DETAIL'].set_time_col('I_DATE')
        graph['USER_LIST'].set_time_col('REG_DATE')
        graph['USER_LIST'].set_end_time_col('WITHDRAW_DATE')

    graph = filter_tables(graph, tables)

    # Fixture-side seeding: unit-test fixtures using `_create_graph` ->
    # CSVConnector / ParquetConnector -> `add_from_config` never go
    # through the production DG ingest path, so the col-stats cache stays
    # empty and downstream `Table.compute_col_stats` raises
    # `UnsupportedInDiskgraphError`. Seed the cache here using the same
    # helper the DataFrame-backed graph fixtures use, gated on
    # file-backed connectors (CSV/Parquet) so we don't double-seed paths
    # that are seeded elsewhere.
    _seed_file_backed_graph_col_stats(graph)
    if infer:
        graph.infer_links()
        graph.infer_metadata()
        graph.infer_encoders()

    return graph


def _seed_file_backed_graph_col_stats(graph: Graph) -> None:
    r"""Seed the DG col-stats cache for every file-backed table in `graph`.

    Reads each table via the connector's `read_table` and routes through the
    same `_seed_parquet_col_stats` helper used by DataFrame-backed graph
    fixtures. Per-table failures are logged and swallowed so a single
    unreadable table doesn't crash the whole fixture.
    """
    from kumoapi.typing import Dtype

    from kumo.connector.file_connector import FileConnector
    from kumo.connector.utils import cast_df_to_dtypes
    seed_logger = logging.getLogger(__name__)
    by_connector: Dict[str, List[Table]] = {}
    for table in graph.tables:
        if not table.has_connector:
            continue
        if not isinstance(table.connector, FileConnector):
            # DataFrame-backed / Snowflake / Databricks fixtures seed col
            # stats elsewhere; only the file-backed CSV/Parquet path falls
            # through `_create_graph` without seeding.
            continue
        by_connector.setdefault(table.data_source, []).append(table)

    for data_source, tables_for_conn in by_connector.items():
        connector = graph.connector(data_source)
        table_dfs: Dict[str, pd.DataFrame] = {}
        primary_keys: Dict[str, Optional[str]] = {}
        time_cols: Dict[str, Optional[str]] = {}
        for table in tables_for_conn:
            try:
                df = connector.read_table(table.source_name)
            except Exception as exc:  # pragma: no cover - defensive
                seed_logger.warning(
                    "Skipping col-stats seed for %s: read_table failed: %s",
                    table.source_name, exc)
                continue
            # CSV reads come back as object dtype even when the graph
            # config declares numeric / date / bool columns; coerce them
            # using the same cast helper the connector applies in
            # production so `compute_col_stats_python` emits the
            # dtype-appropriate stat shapes (NA_FRACTION/MIN/MAX/STD/...)
            # that survivor `Table.compute_col_stats` callers request.
            dtypes_to_cast: Dict[str, Dtype] = {}
            # Many fixtures load `metadata.json` whose
            # `cols` list is empty (only pkey/fkeys are declared); those
            # cols' dtypes only get populated post-`infer_metadata`. To
            # mirror what the consumer's `compute_col_stats` will see, fall
            # back to the connector's metadata-reported dtype (which
            # applies the all-NaN -> string fallback for empty columns).
            # Without this, the seed helper computes stats keyed off the
            # pandas-inferred dtype (e.g. float for an all-NaN column),
            # but the consumer requests stats based on the table's
            # declared dtype (string), and `compute_col_stats_python`
            # raises before the multi-cat stats (NUM_UNIQUE_MULTI etc.)
            # can be emitted.
            try:
                meta_list = connector.get_table_metadata([table.source_name])
            except Exception as exc:  # pragma: no cover - defensive
                seed_logger.warning(
                    "Col-stats seed: get_table_metadata failed for %s: %s",
                    table.source_name, exc)
                meta_list = []
            if meta_list:
                for meta_col in meta_list[0].cols:
                    if (meta_col.dtype is not None
                            and meta_col.name in df.columns):
                        dtypes_to_cast.setdefault(meta_col.name, meta_col.dtype)
            for col in table.cols:
                if col.dtype is None or col.name not in df.columns:
                    continue
                # Explicit `add_col(...)` dtype overrides the connector's
                # metadata-inferred dtype.
                dtypes_to_cast[col.name] = col.dtype
            if table.has_pkey() and table.pkey.dtype is not None and (
                    table.pkey.name in df.columns):
                dtypes_to_cast[table.pkey.name] = table.pkey.dtype
            if dtypes_to_cast:
                df = df.copy()
                try:
                    df = cast_df_to_dtypes(df, dtypes_to_cast)
                except Exception as exc:  # pragma: no cover - defensive
                    seed_logger.warning(
                        "Col-stats seed: dtype cast failed for %s: %s",
                        table.source_name, exc)
            table_dfs[table.source_name] = df
            primary_keys[table.source_name] = (table.pkey.name
                                               if table.has_pkey() else None)
            time_cols[table.source_name] = (table.time_col.name
                                            if table.has_time_col() else None)
        if table_dfs:
            try:
                _seed_parquet_col_stats(connector, table_dfs, primary_keys,
                                        time_cols)
            except Exception as exc:  # pragma: no cover - defensive
                seed_logger.warning(
                    "Col-stats seed failed for connector %s: %s",
                    connector.name, exc)


def _create_s3_graph(name: str, tables: Optional[List[str]] = None,
                     infer: bool = False) -> Graph:
    r"""
    Args:
        tables (list, optional): list of tables to be included in the graph.
            Must use fully-qualified names.
    """
    remote_root = f's3://kumo-public-datasets/{name.lower()}/parquet'
    local_root = osp.join(osp.dirname(osp.realpath(__file__)), 'csv_data')

    graph = Graph(name=name)
    graph.connect(ParquetConnector('parquet', remote_root))
    graph.add_from_config(osp.join(local_root, name, 'metadata.json'))

    graph = filter_tables(graph, tables)

    if infer:
        graph.infer_metadata()
        graph.infer_encoders()

    return graph


def _create_snowflake_graph(
    name: str,
    tables: Optional[List[str]] = None,
    infer: bool = False,
    temporal: bool = False,
    with_view: bool = False,
    time_format: bool = False,
) -> Graph:
    r"""
    Args:
        tables (list, optional): list of tables to be included in the graph.
            Must use fully-qualified names.
    """
    root = osp.join(osp.dirname(osp.realpath(__file__)), 'csv_data', name)

    connector = snowflake_connector.SnowflakeConnector(
        name='snowflake',
        account='xva19026',
        warehouse='WH_XS',
        database='KUMO',
        schema_name=name,
    )

    graph = Graph(name=name)
    graph.connect(connector)
    if time_format:
        metadata_file = 'metadata_timefmt.json'
    elif temporal:
        if with_view:
            metadata_file = 'metadata_timecol_view.json'
        else:
            metadata_file = 'metadata_timecol.json'
    else:
        metadata_file = 'metadata.json'
    graph.add_from_config(osp.join(root, metadata_file),
                          data_source='snowflake')
    graph = filter_tables(graph, tables)

    if infer:
        graph.infer_metadata()
        graph.infer_encoders()

    return graph


# ROW_SIZES specifies the number of rows in each table (and the number of
# tables) for each graph size. The first table is small in all cases so
# that SentenceTransformer is avoided. Otherwise, these sizes are pretty
# arbitrary
ROW_SIZES = {
    'SMALL': [50, 100, 25],
    'MEDIUM': [50, 20000, 20000, 10000, 20000],
    'LARGE': [50, 10000000, 5000000, 5000000, 10000000, 5000000],
}

# Density parameter for EdgeSpec
EDGE_DENSITY = {'LOW': 0.1, 'MEDIUM': 1, 'HIGH': 2}

# Out of all possible pairs of tables, how many are connected by an fkey
FKEY_RATIO = {'LOW': 1 / 3, 'MEDIUM': 2 / 3, 'HIGH': 1}


def _generate_graph(
    graph_name: str,
    size: str = 'SMALL',
    density: str = 'LOW',
    use_feat_cols: bool = True,
    feat_na_frac: float = 0.2,
    seed: int = 0,
) -> Graph:
    np.random.seed(seed)

    row_sizes = ROW_SIZES[size]
    edge_density = EDGE_DENSITY[density]

    feat_cols = [[], [], []]
    if use_feat_cols:
        # Handling N/A's in categorical encoder only implemented for float,
        # integer, string
        feat_cols = [
            [
                (Dtype.string, feat_na_frac),
                (Dtype.date),
                (Dtype.time),
            ],
            [
                (Dtype.int, feat_na_frac),
                (Dtype.bool),
                (Dtype.float, feat_na_frac),
            ],
            [
                (Dtype.int, feat_na_frac),
            ],
        ]

    table_specs = {}
    # Assign each element of feat_cols once before using the last element
    # repeatedly - this results in using each Dtype at least once without a
    # huge encoding cost if there are many tables in the test graph
    for table_id, _ in enumerate(row_sizes):
        table_specs[table_id] = TableSpec(row_sizes[table_id],
                                          feat_cols[min(table_id, 2)])
    edge_specs = {}

    num_tables = len(table_specs.keys())

    # Determine number of fkeys based on density * (# of possible table pairs)
    num_fkeys = round(FKEY_RATIO[density] * num_tables * (num_tables - 1))

    # Index through the first num_fkeys of the set of all non-repeated pairs
    # of table_ids
    for edge_id in range(num_fkeys):
        src = int(edge_id / (num_tables - 1))
        dst = edge_id % (num_tables - 1)
        if dst >= src:
            dst += 1
        edge_specs[(src, dst)] = EdgeSpec(edge_density)

    graph_spec = GraphSpec(
        table_specs=table_specs,
        edge_specs=edge_specs,
    )
    graph = graph_spec.generate(graph_name)
    return graph


@pytest.fixture()
def create_tiny_node_pred_graph() -> Callable:
    return tiny_node_pred_graph


def tiny_node_pred_graph(
    name: str,
    root_dir: str,
    add_entity_end_time: bool = False,
) -> Graph:
    tables_path = osp.join(root_dir, 'tables')
    os.makedirs(tables_path)

    entity_table_path = osp.join(tables_path, 'entity.csv')
    action_table_path = osp.join(tables_path, 'actions.csv')
    # Create entity table
    if add_entity_end_time:
        pd.DataFrame({
            'entity_id': [10, 20],
            'end_time': ['2022-02-08', '2022-02-08'],
        }).to_csv(entity_table_path, index=False)
    else:
        pd.DataFrame({
            'entity_id': [10, 20],
        }).to_csv(entity_table_path, index=False)
    # Create actions table
    pd.DataFrame({
        'entity_id_in_actions': [
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10
        ],
        'amount':
        [1, 1, 1, 1, 1, 1, 20, 15, 23, 47, 38, 44, 41, 40, 1, 1, 1, 1, 1, 1, 1],
        'date': [
            '2022-02-08', '2022-02-07', '2022-02-06', '2022-02-05',
            '2022-02-04', '2022-02-03', '2022-02-02', '2022-02-01',
            '2022-01-31', '2022-01-30', '2022-01-29', '2022-01-28',
            '2022-01-27', '2022-01-26', '2022-01-25', '2022-01-24',
            '2022-01-23', '2022-01-22', '2022-01-21', '2022-01-20', '2022-01-19'
        ]
    }).to_csv(action_table_path, index=False)

    # ``data_version`` is set explicitly so ``Table.data_version`` returns
    # a string straight from ``self._data_version`` instead of going
    # through the connector path, which can return a non-str sentinel
    # (``DEFAULT_DATA_VERSION`` is a float in ``io_utils.fsutil.common``)
    # that pydantic then rejects when ``Table.config()`` builds a
    # ``TableConfig``. Value must NOT equal ``connector/utils.py``'s
    # ``DEFAULT_DATA_VERSION = '-1'`` — otherwise ``has_valid_data_version``
    # returns False and the global-materialize-dir assert fires.
    if add_entity_end_time:
        entity = Table('entity', 'csv',
                       graph_column.PrimaryKey('entity_id', Dtype.int32),
                       source_name='entity', end_time_col='end_time',
                       data_version='1')
    else:
        entity = Table('entity', 'csv',
                       graph_column.PrimaryKey('entity_id', Dtype.int32),
                       source_name='entity', data_version='1')
    tables = [
        entity,
        Table(
            'actions', 'csv',
            graph_column.PrimaryKey('entity_id_in_actions', Dtype.int32), cols=[
                graph_column.Column('amount', dtype=Dtype.int32,
                                    stype=Stype.numerical)
            ], source_name='actions', fkeys=[
                graph_column.ForeignKey('entity_id_in_actions', 'csv.entity',
                                        'entity_id')
            ], time_col=graph_column.TimeColumn('date', Dtype.date),
            data_version='1')
    ]
    graph = Graph(name=name,
                  data_source_dict={'csv': CSVConnector('csv', tables_path)},
                  tables=tables)
    return graph


@pytest.fixture()
def create_node_pred_graph() -> Callable:
    return node_pred_test_graph


def node_pred_test_graph(
    name: str,
    root_dir: str,
    create_hypothetical_entity: bool = False,
    infer_encoders: bool = False,
    add_disconnected_tables: bool = False,
    rename_transaction_fkey: bool = False,
    user_with_nonexistent_location: bool = False,
    small_target_values: bool = False,
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
    add_high_precision_timestamp: bool = False,
    add_na_booleans: bool = False,
    add_dangling_fkeys: bool = False,
    users_time_col_fkey: bool = False,
) -> Graph:
    """Graph used to test temporal node prediction queries. Graph
    has tables `USERS`, `TRANSACTIONS`, `SESSIONS`, `ORDERS`,
    `BANKS`, and `LOCATIONS`.

    Args:
        name: Name of graph.
        root_dir: Directory where csv files of tables will be saved.
        create_hypothetical_entity: If True, create a hypothetical entity
            `NEW_USERS` that is not present in the graph, but is saved in
            in the same directory as the other tables.
        infer_encoders: If True, encoders will be inferred from scratch.
            Otherwise, they will be set manually (faster).
        add_disconnected_tables: If True, add tables that are not connected
            to the graph. Tables added are `OLD_USER` and `OLD_TRANSACTIONS`.
            This is used to test that that certain queries will raise an error.
        rename_transaction_fkey: If `True` rename the `USER_ID` column in
            `TRANSACTIONS` to `uSER_ID`. This is used to test that certain
            case sensitive column names are handled correctly.
        small_target_values (bool): If True, change `AMOUNT` column in
            transaction table to smaller values. This is mainly used for
            heatmap bin width testing (default: :obj:`False`)

        add_na_pkeys: If `True`, tables will additionally include rows with
            pd.NA primary keys.
        add_na_timestamps: If `True`, tables will additionally include rows
            with pd.NA timestamps.
        add_high_precision_timestamp: If `True`, timestamp in transaction table
            will contain timstamps of high precision, default to microsecond.
        add_na_booleans: If `True`, IS_ONLINE in TRANSACTIONS table will
            contain null values to test correct handline of Kleene-Priest three
            value logic
        add_dangling_fkeys: If `True`, BANK_ID in TRANSACTIONS_WITH_PKEY table
            will contain a value that doesn't point to a valid bank to test
            whether we handle those correctly as NULLs
        users_time_col_fkey: If `True`, add a time col `CREATE_TIME` to USERS
            and an fkey to transactions `FIRST_TRANSACT` (supports creating
            edge cases for pquery filters)
    """
    # Create tables:
    users = pd.DataFrame(
        {
            'USER_ID': [0, 1, 2, 3],
            'AGE': [19, 20, 21, 22],
            'LOCATION_ID': [0, 1, 0, 1],
        }, )
    if user_with_nonexistent_location:
        users['LOCATION_ID'][3] = 5  # 5 is a non-existent location
    if add_na_pkeys:
        users.loc[4] = [pd.NA, 23, pd.NA]

    new_users = pd.DataFrame(
        {
            'USER_ID': [4, 5],
            'AGE': [20, 21],
            'LOCATION_ID': [0, 1]
        }, )
    if users_time_col_fkey:
        users.insert(
            2, 'CREATE_TIME',
            pd.to_datetime(
                ['1989-12-01', '1989-12-10', '1990-01-02',
                 '1990-01-03']).astype('datetime64[ns]'))
        users.insert(2, 'FIRST_TRANSACT', [1, 2, 3, 4])

    users = users.set_index('USER_ID')
    locations = pd.DataFrame({
        'LOCATION_ID': [0, 1],
        'NAME': ['Pallet Town', 'Cerulean City'],
        'DESCRIPTION': [
            'Pallet Town is a fictional town in Kanto.',
            'Cerulean City is a seaside city located in northern Kanto.'
        ],
    })
    if add_na_pkeys:
        locations.loc[2] = [pd.NA, 'MissingNO.', '']
    user_id_col = 'uSER_ID' if rename_transaction_fkey else 'USER_ID'
    transactions_timestamps = [
        '1990-01-01', '1990-01-01', '1990-01-01', '1990-01-01', '1990-01-03',
        '1990-01-03', '1990-01-04', '1990-01-04', '1990-01-04', '1990-01-05'
    ]

    if add_high_precision_timestamp:
        for i in range(len(transactions_timestamps)):
            transactions_timestamps[i] += " 17:28:34.951234"

    transactions = pd.DataFrame({
        user_id_col: [0, 1, 2, 3, 1, 3, 3, 0, 1, 0],
        'AMOUNT': [0.0, 0.0, 0.0, 0.0, 150.0, 20.0, 30.0, 30.0, 150.0, 0.0],
        'CATEGORY': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'BANK_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'IS_ONLINE': [
            True,
            False,
            False,
            False,
            False if not add_na_booleans else None,
            True,
            True,
            False,
            False,
            True,
        ],
        'DATE':
        transactions_timestamps,
        'TYPE': ['A', 'B', 'C', 'D', 'E'] * 2,
        'MULTI_STATUS_PIPE': ['A|B', 'C|D', 'A|B|C', 'A|B|C|D', 'B'] * 2,
        'MULTI_STATUS_COMMA': ['A,B', 'C,D', 'A,B,C', 'A,B,C,D', 'B'] * 2,
        'MULTI_STATUS_MIXED_COMMA':
        ['A,|B', '|C,D', 'A,|B,|C', 'A,|B,|C,D', '|B'] * 2,
        'MULTI_STATUS_MIXED_PIPE':
        ['A|,B', ',C|D', 'A|,B|,C', 'A|,B|,C|D', ',B'] * 2,
    })
    transactions_with_pkey = pd.DataFrame({
        user_id_col: [0, 1, 2, 3, 1, 3, 3, 0, 1, 0],
        'TRANSACTION_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'AMOUNT': [0.0, 0.0, 0.0, 0.0, 150.0, 20.0, 30.0, 30.0, 150.0, 0.0],
        'CATEGORY': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'BANK_ID': [0, 0, 0, 0, 0, 1, 1, 1, 2 if add_dangling_fkeys else 1, 1],
        'IS_ONLINE': [
            True,
            False,
            False,
            False,
            False if not add_na_booleans else None,
            True,
            True,
            False,
            False,
            True,
        ],
        'DATE':
        transactions_timestamps,
        'TYPE': ['A', 'B', 'C', 'D', 'E'] * 2,
    })
    if small_target_values:
        transactions['AMOUNT'] /= 10000
    if add_na_timestamps:
        transactions.loc[10] = [
            0, 0.0, 0, 0, True, pd.NA, 'A', None, None, None, None
        ]
        transactions.loc[11] = [
            1, 100.0, 0, 0, True, pd.NA, 'B', None, None, None, None
        ]
    if add_na_pkeys:
        idx = len(transactions)
        transactions.loc[idx] = [
            pd.NA, 0.0, 0, 0, True, '1990-01-03', 'A', None, None, None, None
        ]
    # Convert to datetime with explicit nanosecond precision to ensure
    # compatibility with PyArrow 14
    transactions['DATE'] = pd.to_datetime(
        transactions['DATE']).astype('datetime64[ns]')
    transactions_with_pkey['DATE'] = pd.to_datetime(
        transactions_with_pkey['DATE']).astype('datetime64[ns]')
    sessions = pd.DataFrame({
        'USER_ID': [1, 3, 1, 3],
        'DURATION': [1.0, 1.0, 1.0, 1.0],
        'DATE': ['1990-01-02', '1990-01-02', '1990-01-04', '1990-01-04'],
    })
    if add_na_timestamps:
        sessions.loc[4] = [1, 1.0, pd.NA]
    if add_na_pkeys:
        idx = len(sessions)
        sessions.loc[idx] = [pd.NA, 1.0, '1990-01-03']
    sessions['DATE'] = pd.to_datetime(sessions['DATE']).astype('datetime64[ns]')
    orders = pd.DataFrame({
        'USER_ID': [1, 2, 1, 2],
        'BALANCE': [1.0, 1.0, 1.0, 1.0],
        'DATE': ['1990-01-02', '1990-01-02', '1990-01-04', '1990-01-04'],
    })
    if add_na_timestamps:
        orders.loc[4] = [1, 1.0, pd.NA]
    if add_na_pkeys:
        idx = len(orders)
        orders.loc[idx] = [pd.NA, 1.0, '1990-01-03']
    orders['DATE'] = pd.to_datetime(orders['DATE']).astype('datetime64[ns]')
    banks = pd.DataFrame({
        'BANK_ID': [0, 1],
        'BANK_TYPE': ['A', 'B'],
        'AMOUNT': [1, 1],
    })
    if add_na_pkeys:
        banks.loc[2] = [pd.NA, 'A', 1]

    # Save tables to csv files:
    tables_dir = 'tables-' + uuid4().hex[:16]
    tables_path = osp.join(root_dir, tables_dir)
    os.makedirs(tables_path)
    tables_names = [
        'USERS', 'TRANSACTIONS', 'SESSIONS', 'ORDERS', 'BANKS', 'LOCATIONS',
        'OLD_USERS', 'OLD_TRANSACTIONS', 'TRANSACTIONS_WITH_PKEY'
    ]
    tables = [
        users, transactions, sessions, orders, banks, locations, users,
        transactions, transactions_with_pkey
    ]
    for table_name, table in zip(tables_names, tables):
        table_path = osp.join(tables_path, table_name + '.parquet')
        # Ensure timestamps are written with int96 for PyArrow 14 compatibility
        # This matches how the production code writes timestamps
        table.to_parquet(table_path,
                         index=True if 'USERS' in table_name else False,
                         use_deprecated_int96_timestamps=True)
    if create_hypothetical_entity:
        new_users_path = osp.join(tables_path, 'NEW_USERS.parquet')
        new_users.to_parquet(new_users_path, index=False,
                             use_deprecated_int96_timestamps=True)

    connector = ParquetConnector('parquet', root_dir=tables_path)
    _seed_parquet_col_stats(
        connector,
        dict(zip(tables_names, tables)),
        {
            'USERS': 'USER_ID',
            'TRANSACTIONS': None,
            'SESSIONS': None,
            'ORDERS': None,
            'BANKS': 'BANK_ID',
            'LOCATIONS': 'LOCATION_ID',
            'OLD_USERS': 'USER_ID',
            'OLD_TRANSACTIONS': None,
            'TRANSACTIONS_WITH_PKEY': 'TRANSACTION_ID',
        },
        {
            'USERS': 'CREATE_TIME' if users_time_col_fkey else None,
            'TRANSACTIONS': 'DATE',
            'SESSIONS': 'DATE',
            'ORDERS': 'DATE',
            'OLD_TRANSACTIONS': 'DATE',
            'TRANSACTIONS_WITH_PKEY': 'DATE',
        },
    )

    # Create graph:
    graph = Graph(name=name)
    graph.connect(connector)
    graph.add('USERS', pkey='USER_ID', cols=['AGE'],
              time_col='CREATE_TIME' if users_time_col_fkey else None)
    graph.add(
        'TRANSACTIONS', cols=[
            'AMOUNT',
            'TYPE',
            'CATEGORY',
            'MULTI_STATUS_PIPE',
            'MULTI_STATUS_COMMA',
            'MULTI_STATUS_MIXED_COMMA',
            'MULTI_STATUS_MIXED_PIPE',
        ], time_col='DATE')
    graph.add('TRANSACTIONS_WITH_PKEY', cols=['AMOUNT', 'TYPE', 'CATEGORY'],
              time_col='DATE', pkey='TRANSACTION_ID')
    graph.add('SESSIONS', cols=['DURATION'], time_col='DATE')
    graph.add('ORDERS', cols=['BALANCE'], time_col='DATE')
    graph.add('BANKS', cols=['BANK_TYPE'], pkey='BANK_ID')
    graph.add('LOCATIONS', pkey='LOCATION_ID')
    graph.link(src_table='TRANSACTIONS', src_key=user_id_col, dst_table='USERS')
    graph.link(src_table='SESSIONS', src_key='USER_ID', dst_table='USERS')
    graph.link(src_table='ORDERS', src_key='USER_ID', dst_table='USERS')
    graph.link(src_table='TRANSACTIONS', src_key='BANK_ID', dst_table='BANKS')
    graph.link(src_table='USERS', src_key='LOCATION_ID', dst_table='LOCATIONS')
    if users_time_col_fkey:
        graph.link(src_table='USERS', src_key='FIRST_TRANSACT',
                   dst_table='TRANSACTIONS_WITH_PKEY')
    graph.link(src_table='TRANSACTIONS_WITH_PKEY', src_key=user_id_col,
               dst_table='USERS')
    graph.link(src_table='TRANSACTIONS_WITH_PKEY', src_key='BANK_ID',
               dst_table='BANKS')
    if add_disconnected_tables:
        graph.add('OLD_USERS', pkey='USER_ID')
        graph.add('OLD_TRANSACTIONS', cols=['AMOUNT'], time_col='DATE')
        graph.link(src_table='OLD_TRANSACTIONS', src_key='USER_ID',
                   dst_table='OLD_USERS')
    graph = graph.infer_metadata()
    graph['LOCATIONS', 'DESCRIPTION'].stype = Stype.text
    graph['TRANSACTIONS', 'MULTI_STATUS_PIPE'].stype = Stype.multicategorical
    graph['TRANSACTIONS', 'MULTI_STATUS_PIPE'].multicategorical_delimiter = '|'
    graph['TRANSACTIONS', 'MULTI_STATUS_COMMA'].stype = Stype.multicategorical
    graph['TRANSACTIONS', 'MULTI_STATUS_COMMA'].multicategorical_delimiter = ','
    graph['TRANSACTIONS',
          'MULTI_STATUS_MIXED_COMMA'].stype = Stype.multicategorical
    graph['TRANSACTIONS',
          'MULTI_STATUS_MIXED_COMMA'].multicategorical_delimiter = ','
    graph['TRANSACTIONS',
          'MULTI_STATUS_MIXED_PIPE'].stype = Stype.multicategorical
    graph['TRANSACTIONS',
          'MULTI_STATUS_MIXED_PIPE'].multicategorical_delimiter = '|'
    if infer_encoders:
        return graph.infer_encoders()
    else:
        graph['USERS', 'AGE'].encoder = MaxLogNumerical()
        graph['TRANSACTIONS', 'AMOUNT'].encoder = MaxLogNumerical()
        graph['TRANSACTIONS', 'DATE'].encoder = Datetime()
        graph['TRANSACTIONS', 'CATEGORY'].encoder = Index(min_occ=2)
        graph['TRANSACTIONS', 'TYPE'].encoder = Index(min_occ=2)
        graph['TRANSACTIONS', 'IS_ONLINE'].encoder = Index()
        graph['TRANSACTIONS', 'MULTI_STATUS_PIPE'].encoder = MultiCategorical()
        graph['TRANSACTIONS', 'MULTI_STATUS_COMMA'].encoder = MultiCategorical()
        graph['TRANSACTIONS',
              'MULTI_STATUS_MIXED_COMMA'].encoder = MultiCategorical()
        graph['TRANSACTIONS',
              'MULTI_STATUS_MIXED_PIPE'].encoder = MultiCategorical()
        graph['TRANSACTIONS_WITH_PKEY', 'AMOUNT'].encoder = MaxLogNumerical()
        graph['TRANSACTIONS_WITH_PKEY', 'DATE'].encoder = Datetime()
        graph['TRANSACTIONS_WITH_PKEY', 'CATEGORY'].encoder = Index(min_occ=2)
        graph['TRANSACTIONS_WITH_PKEY', 'TYPE'].encoder = Index(min_occ=2)
        graph['TRANSACTIONS_WITH_PKEY', 'IS_ONLINE'].encoder = Index()
        graph['SESSIONS', 'DURATION'].encoder = Index(min_occ=2)
        graph['SESSIONS', 'DATE'].encoder = Datetime()
        graph['ORDERS', 'BALANCE'].encoder = Index(min_occ=2)
        graph['ORDERS', 'DATE'].encoder = Datetime()
        graph['BANKS', 'AMOUNT'].encoder = Index(min_occ=2)
        graph['BANKS', 'BANK_TYPE'].encoder = Index(min_occ=2)
        graph['LOCATIONS', 'NAME'].encoder = Index(min_occ=2)
        graph['LOCATIONS', 'DESCRIPTION'].encoder = TextEmbedding()
        if add_disconnected_tables:
            graph['OLD_TRANSACTIONS', 'AMOUNT'].encoder = MaxLogNumerical()
            graph['OLD_TRANSACTIONS', 'DATE'].encoder = Datetime()
            graph['OLD_TRANSACTIONS', 'CATEGORY'].encoder = Index(min_occ=2)
            graph['OLD_TRANSACTIONS', 'TYPE'].encoder = Index(min_occ=2)
        return graph


@pytest.fixture()
def create_forecasting_graph() -> Callable:
    return forecasting_test_graph


def forecasting_test_graph(root_dir: str) -> Graph:
    os.makedirs(root_dir, exist_ok=True)
    user_df = pd.DataFrame({
        'USER_ID': [0, 1, 2],
        'AGE': [10, 20, 30],
    })
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    trans_df = pd.DataFrame({
        'USER_ID': [0] * 32 + [1] * 32 + [2] * 32,
        'AMOUNT':
        np.random.rand(3 * 32),
        'DATE': [
            '1989-01-01', '1989-01-03', '1989-01-05', '1989-01-07',
            '1989-01-09', '1989-01-11', '1989-01-13', '1989-01-15',
            '1989-01-17', '1989-01-19', '1989-01-21', '1989-01-23',
            '1989-01-25', '1989-01-27', '1989-01-29', '1989-01-31',
            '1990-01-01', '1990-01-03', '1990-01-05', '1990-01-07',
            '1990-01-09', '1990-01-11', '1990-01-13', '1990-01-15',
            '1990-01-17', '1990-01-19', '1990-01-21', '1990-01-23',
            '1990-01-25', '1990-01-27', '1990-01-29', '1990-01-31'
        ] * 3,
        'PRODUCT_ID': [0] * 64 + [1] * 32
    })
    parquet_path = osp.join(root_dir, 'TRANS.parquet')
    trans_df.to_parquet(parquet_path)
    product_df = pd.DataFrame({
        'PRODUCT_ID': [0, 1],
        'BRAND': ['HERMES', 'CHANEL']
    })
    parquet_path = osp.join(root_dir, 'PRODUCTS.parquet')
    product_df.to_parquet(parquet_path)

    connector = ParquetConnector('parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'TRANS': trans_df,
            'PRODUCTS': product_df,
        },
        {
            'USERS': 'USER_ID',
            'TRANS': None,
            'PRODUCTS': 'PRODUCT_ID',
        },
        {'TRANS': 'DATE'},
    )

    graph = Graph(name='FORECASTING_TEST_GRAPH')
    graph.connect(connector)
    graph.add('USERS', pkey='USER_ID', cols=['AGE'])
    graph.add('TRANS', fkeys=['USER_ID', 'PRODUCT_ID'], cols=['AMOUNT'],
              time_col='DATE')
    graph.add('PRODUCTS', pkey='PRODUCT_ID', cols=['BRAND'])

    graph.infer_links()
    graph.infer_metadata(include_cols=[])
    graph.infer_encoders()

    return graph


def _create_financial_task(
    task_type: TaskType,
    temporal: bool,
    root_dir: Optional[str] = None,
    parquet: bool = False,
) -> Task:
    graph = _create_graph(
        name='FINANCIAL',
        root_dir=root_dir,
        parquet=parquet,
        temporal=temporal,
    )
    graph.infer_metadata()
    graph.infer_encoders()

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        train_table_name = 'TRAINING_CLASSIFICATION'
        target_col = dict(name='STATUS', dtype=Dtype.string)
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        train_table_name = 'TRAIN_BINARY'
        target_col = dict(name='STATUS', dtype=Dtype.bool)
    elif task_type == TaskType.REGRESSION:
        train_table_name = 'TRAINING_REGRESSION'
        target_col = dict(name='AMOUNT', dtype=Dtype.float)
    else:
        raise NotImplementedError

    del graph['LOAN', target_col['name']]  # Prevent data leakage.

    # Build the Task with ``ignore_encoders=True`` so the
    # constructor skips encoder inference (which strict-loads from the DG
    # train-table col-stats cache and would otherwise raise here). Then
    # seed the cache from the parquet train table, after which downstream
    # callers (Trainer/AutoTrainer) re-run encoder inference and resolve
    # against the now-populated cache. Pin ``train_table_data_version`` so
    # the cache key is deterministic for the seed-then-lookup round-trip;
    # ``Task.seed_train_table_col_stats_from_dataframe`` raises if
    # ``data_version`` is None.
    task = Task(
        graph,
        task_type,
        train_table_name=train_table_name,
        target_col=target_col,
        fkeys=['LOAN_ID'],
        time_col='DATE' if temporal else None,
        task_name=f'{graph.name}_{train_table_name}',
        train_table_data_version=f'financial-{train_table_name}-v1',
        ignore_encoders=True,
    )
    if temporal:
        task.temporal_split([0.8, 0.1, 0.1])
    else:
        task.random_split([0.8, 0.1, 0.1])

    try:
        train_df = graph.connector(
            graph.tables[0].data_source).read_table(train_table_name)
        task.seed_train_table_col_stats_from_dataframe(train_df)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            'seed_train_table_col_stats_from_dataframe failed in '
            '_create_financial_task (%s); strict path will surface '
            'the genuine miss at the call site.', exc)

    # Now that the train-table col-stats cache is seeded, run the
    # encoder inference that the constructor skipped. Mirrors what
    # ``Task.__init__(ignore_encoders=False)`` would have done at
    # construction.
    task.infer_metadata()

    return task


def _create_synthetic_task(
    name: str,
    task_type: TaskType,
    temporal: bool,
    **kwargs,
) -> Task:
    # The training table created below via TableSpec (using train_col_specs)
    # has a column '0' which is specified as the corresponding type for each
    # task type. Also define train column via train_col_specs
    # for classification, generate a column with 0 na's and 5 distinct random
    # strings, for multilabel classification generate a column with 0 na's and
    # strings joined with "|" with max 4 categories out of 7 category choices,
    # otherwise just do a random int column
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        train_table_name = 'TRAINING_CLASSIFICATION'
        target_col = dict(name='0', dtype=Dtype.string)
        train_col_specs = [dict(dtype=Dtype.string, na_frac=0, distinct_vals=5)]
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        train_table_name = 'TRAINING_CLASSIFICATION'
        target_col = dict(name='0', dtype=Dtype.bool)
        train_col_specs = [dict(dtype=Dtype.bool, na_frac=0, distinct_vals=2)]
    elif task_type.is_multilabel:
        train_table_name = 'TRAINING_MULTILABEL'

        from kumo_core.training.options import allow_array_targets
        if allow_array_targets():
            target_col = dict(name='0', dtype=Dtype.stringlist)
            train_col_specs = [
                dict(dtype=Dtype.stringlist, na_frac=0, distinct_vals=7,
                     max_labels=4)
            ]
        else:
            train_table_name = 'TRAINING_MULTILABEL'
            target_col = dict(name='0', dtype=Dtype.string)
            train_col_specs = [
                dict(dtype=Dtype.string, na_frac=0, distinct_vals=7,
                     max_labels=4, delimiter="|")
            ]
    elif task_type == TaskType.REGRESSION:
        train_table_name = 'TRAINING_REGRESSION'
        target_col = dict(name='0', dtype=Dtype.float)
        train_col_specs = [dict(dtype=Dtype.float)]
    elif task_type.is_link_pred:
        train_table_name = 'TRAINING_LINK_PREDICTION'
        target_col = None
        train_col_specs = []
    else:
        raise NotImplementedError

    if temporal:
        train_col_specs.append(dict(dtype=Dtype.date))

    train_col_specs = [list(specs.values()) for specs in train_col_specs]

    train_table_name = kwargs.get('train_table_name', train_table_name)
    target_col = kwargs.get('target_col', target_col)

    num_entity_nodes = 300

    # TODO(manan, alex): re-enable EdgeSpec(2) when we have clear support for
    # FK/FK relatioships:
    graph_spec = GraphSpec(
        table_specs={
            0:
            TableSpec(num_entity_nodes,
                      [Dtype.float, (Dtype.float, 0.1), (Dtype.string, 0.4)]),
            1:
            TableSpec(25, [Dtype.int, Dtype.time]),
            2:
            TableSpec(36, [Dtype.int, Dtype.date]),
        },
        edge_specs={
            (0, 1): EdgeSpec(0.5),
            (2, 0): EdgeSpec(1),
            (1, 2): EdgeSpec(1)
        },
    )

    graph = graph_spec.generate(f'{name}_{temporal}_{task_type}')
    graph.set_table_data_versions(f'{name}_{temporal}_{task_type}')

    # Set semantic types:
    graph['t0', '0'].stype = Stype.numerical
    graph['t0', '1'].stype = Stype.numerical
    graph['t0', '2'].stype = Stype.categorical
    graph['t1', '0'].stype = Stype.categorical
    graph['t1', '1'].stype = Stype.timestamp
    graph['t2', '0'].stype = Stype.categorical
    graph['t2', '1'].stype = Stype.timestamp

    if temporal:
        graph['t1'].set_time_col('1')
        graph['t2'].set_time_col('1')

    graph.infer_metadata()
    graph.infer_encoders()

    # leave the integer colum (effectively pkey) as a col that isn't the pandas
    # index
    data_source = kwargs.get(
        'data_source',
        TableSpec(num_entity_nodes,
                  train_col_specs).generate(pkey_as_index=False))
    data_source[NODE_INDEX_COLUMN] = range(data_source.shape[0])
    fkeys = kwargs.get('fkeys', [{
        'name': 'index',
        'dst_table': 't0',
    }])
    time_col = kwargs.get('time_col', '1' if temporal else None)
    split_ratio = kwargs.get('split_ratio', [0.8, 0.1, 0.1])

    # Build the Task with ``ignore_encoders=True`` so the
    # constructor skips encoder inference (which strict-loads from the DG
    # train-table col-stats cache and would otherwise raise here). Then
    # seed the cache from the in-memory ``data_source``, after which we
    # re-run ``infer_metadata`` to populate encoders. See
    # ``_create_financial_task`` for context.
    task = Task(
        graph,
        task_type,
        train_table_name=train_table_name,
        target_col=target_col,
        fkeys=fkeys,
        time_col=time_col,
        data_source=data_source,
        task_name=f'{name}_{train_table_name}',
        train_table_data_version=f'{name}_{temporal}_{task_type}',
        metrics=DEFAULT_METRICS[task_type],
        ignore_encoders=True,
    )

    if temporal:
        task.temporal_split(split_ratio)
    else:
        task.random_split(split_ratio)

    if not task_type.is_link_pred and task.train_table.target_col is not None:
        try:
            task.seed_train_table_col_stats_from_dataframe(data_source)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                'seed_train_table_col_stats_from_dataframe failed in '
                '_create_synthetic_task (%s); strict path will surface '
                'the genuine miss at the call site.', exc)

    task.infer_metadata(include_cols=[target_col])

    return task


def _create_task(
    name: str,
    task_type: Union[TaskType, str],
    temporal: bool = False,
    materialize: Optional[bool] = False,
    root_dir: Optional[str] = None,
    parquet: bool = False,
    **kwargs,
) -> Task:
    r"""Calls a specific task construction method corresponding to name, either
    FINANCIAL or synthetic are current options. Also optionally materialize the
    task before returning it.

    Args:
        name (str): name of task to be created. Also specifies which task
            construction method to call.
        task_type (TaskType, str): type from kumo.typing.TaskType, determines
            type of Task constructed
        temporal (bool, optional): if task should be temporal (Default: False)
        materialize (bool, optional): if True, materialize task before
            returning (Default: False)
        root_dir (str, optional): the path to the folder contains the graph
            data, only for non-synthetic task. (Default: None)
        **kwargs: arguments passed to synthetic task construction to set Task
            initialization arguments
    """
    task_type = TaskType(task_type)
    if name == 'FINANCIAL':
        task = _create_financial_task(
            task_type,
            temporal,
            root_dir,
            parquet,
        )
    elif name.startswith('SYNTHETIC'):
        task = _create_synthetic_task(name, task_type, temporal, **kwargs)
    else:
        raise NotImplementedError

    # NOTE: ``materialize`` is retained on the fixture signature (and as a
    # ``SyntheticTaskDescriptor`` cache-key dimension) for API compatibility,
    # but in-process ``Task.materialize`` has been removed; the task is always
    # returned config-only.
    return task


@pytest.fixture()
def create_graph() -> Callable[..., Graph]:
    return _create_graph


@pytest.fixture()
def create_s3_graph() -> Callable:
    return _create_s3_graph


@pytest.fixture()
def create_snowflake_graph() -> Callable:
    return _create_snowflake_graph


@pytest.fixture()
def create_task() -> Callable:
    return _create_task


@pytest.fixture(scope="module")
def snowflake_session(schema: str = 'FINANCIAL_PYTEST') -> SnowparkSession:

    from io_utils.secrets import get_secrets

    private_key = get_secrets(
        "xva19026.snowflakecomputing.com")["kumo_user_secretkey"]

    from cryptography.hazmat.primitives import serialization

    # perform the same decoding used by SQLConnector.get_user_pass_or_key
    pkey_serialized = serialization.load_pem_private_key(
        private_key.encode(), password=None)

    pkey_serialized = pkey_serialized.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())

    connection_parameters = {
        "account": 'xva19026',
        "user": os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER),
        # Fetch private key from secrets manager if available, otherwise fallback to environment variable
        "private_key": pkey_serialized,
        "warehouse": 'WH_XS',
        "database": 'KUMO',
        "schema": schema,
    }
    return SnowparkSession.builder.configs(connection_parameters).create()


@pytest.fixture(name="snowflake_connector")
def snowflake_connector_factory() -> Callable:
    return _snowflake_connector


def _snowflake_connector(schema: str) -> snowflake_connector.SnowflakeConnector:
    return snowflake_connector.SnowflakeConnector(
        name='snowflake',
        account='xva19026',
        warehouse='WH_XS',
        database='KUMO',
        schema_name=schema,
    )


@pytest.fixture(scope='session')
def valid_fake_private_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


# Dataclass for indexing tasks by description to persist across tests
@kumo_dataclass(frozen=True)
class SyntheticTaskDescriptor:
    task_type: TaskType
    temporal: bool
    materialized: bool


# Tasks indexed by a SyntheticTaskDescriptor use these metrics values
DEFAULT_METRICS = {
    TaskType.REGRESSION:
    None,
    TaskType.MULTICLASS_CLASSIFICATION:
    None,
    TaskType.MULTILABEL_CLASSIFICATION:
    None,
    TaskType.MULTILABEL_RANKING:
    None,
    TaskType.BINARY_CLASSIFICATION:
    ['auroc', 'ap', 'auprc', 'recall@100', 'precision@100'],
    TaskType.TEMPORAL_LINK_PREDICTION:
    None,
    TaskType.STATIC_LINK_PREDICTION:
    None,
}


@pytest.fixture(scope="session")
def get_task() -> Callable:
    task_dict: Dict[SyntheticTaskDescriptor, Task] = {}

    def _get_task(
        task_type: TaskType,
        temporal: bool,
        metrics: Optional[List[str]] = None,
        create_new: bool = False,
        materialize: bool = True,
    ) -> Task:

        task_descriptor = SyntheticTaskDescriptor(
            task_type,
            temporal,
            materialize,
        )

        if task_descriptor not in task_dict or create_new:
            name = f'SYNTHETIC_{uuid4().hex}' if create_new else 'SYNTHETIC'
            task = _create_task(
                name=name,
                task_type=task_type,
                temporal=temporal,
                materialize=materialize,
            )

            if not create_new:  # Cache the task.
                task_dict[task_descriptor] = copy.deepcopy(task)

        else:
            task = copy.deepcopy(task_dict[task_descriptor])

        if metrics is not None:
            task._metrics = metrics

        return task

    return _get_task


@pytest.fixture()
def generate_graph() -> Callable:
    return _generate_graph


@pytest.fixture
def tempdir(request):
    with tempfile.NamedTemporaryFile(delete=False) as tempdir:
        os.unlink(tempdir.name)

        request.addfinalizer(
            lambda: shutil.rmtree(tempdir.name, ignore_errors=True))
        return tempdir


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    return True


@pytest.fixture(autouse=True)
def run_before_and_after_tests(auto_setup_kumo_backend: bool,
                               tmp_path: pathlib.Path):
    if not auto_setup_kumo_backend:
        yield
        return
    """With `autouse=True`, we run this function before and after each test
    to set appropriate backends for each test."""
    # The data persist dir is derived from deploy_env() (kumo.data
    # .get_persist_dir); in TEST mode it resolves to a stable path under the
    # repo cache dir. It's wiped after each test below to preserve the old
    # per-test-empty seeding contract.
    data_persist_dir = data.get_persist_dir()
    os.makedirs(data_persist_dir, exist_ok=True)

    # The engine layer (DiskGraph) holds no process-global state: the persist
    # dir is threaded explicitly and there is no engine mode, so there is
    # nothing to reset between tests here.

    restore_col_stats_fallback = _install_dg_col_stats_fallback()
    try:
        # Run the test:
        yield
    finally:
        restore_col_stats_fallback()

        # Drop the process-wide BackendCacheManager and wipe the (deploy_env
        # derived) data persist dir so the next test starts from an empty
        # data backend. Setup is re-driven by this autouse fixture.
        data.reset_active_cache_for_test()
        fsutil.rm(data.get_persist_dir(), ignore_errors=True)
        # Reset artifacts in home dir:
        fsutil.rm(
            kumo.home.HomeDirManager(kumo.home.get_home_dir()).pseudo_time_dir,
            ignore_errors=True)


@pytest.fixture(scope='session', autouse=True)
def run_before_and_after_test_sessions():
    # The data persist dir is derived from deploy_env() (kumo.data
    # .get_persist_dir); wipe it at session boundaries for clean seeding.
    data_persist_dir = data.get_persist_dir()

    if osp.exists(data_persist_dir):
        shutil.rmtree(data_persist_dir)

    yield

    if osp.exists(data_persist_dir):
        shutil.rmtree(data_persist_dir)


@pytest.fixture(scope='session', autouse=True)
def _redirect_test_diskgraph_data_root():
    """Point the DiskGraph data_root at a writable location under TEST.

    The production default for ``DiskgraphConfig.data_root`` is
    ``/mnt/localssd`` (see ``service_framework/_raw_deploy_config.py``).
    On CI runners that path is not writable, so DG-native trainer integ
    fixtures (which copy the raw index under ``svc._config.data_root``)
    fail. ``data_plane_storage_root`` is already resolved to a writable
    home dir in TEST mode, so we override ``data_compute.data_root`` to
    match it — but ONLY when we are in TEST mode and the value is still
    the untouched production default. This never changes production
    behavior; it only fixes the test deploy_env singleton.
    """
    from service_framework import deploy_env_config as _dec
    from service_framework.deploy_env_config import (
        DeployStage,
        DiskgraphConfig,
    )

    env = _dec.deploy_env()
    needs_redirect = (env.deploy_mode == DeployStage.TEST
                      and isinstance(env.data_compute, DiskgraphConfig)
                      and env.data_compute.data_root == '/mnt/localssd')
    if not needs_redirect:
        yield
        return

    original = _dec._deploy_env
    writable_root = env.data_plane_storage_root
    os.makedirs(writable_root, exist_ok=True)
    patched = dataclasses.replace(
        env,
        data_compute=dataclasses.replace(env.data_compute,
                                         data_root=writable_root),
    )
    _dec._deploy_env = patched
    try:
        yield
    finally:
        _dec._deploy_env = original


@pytest.fixture
def seed_col_stats() -> Callable[..., None]:
    """Opt-in fixture that pre-populates the DG col-stats cache.

    Tests that hit ``Table.compute_col_stats`` (or ``compute_table_stats``)
    fail because the DG read path is cache-only and DG ingest isn't run by
    the test harness. Request this fixture and call the returned function
    before constructing the Graph/Table; it mirrors what DG ingest writes
    in production.

    Usage::

        def test_xyz(seed_col_stats):
            df = pd.DataFrame({...})
            seed_col_stats(df, table_name='ACCOUNT', data_version='123',
                           primary_key='ACCOUNT_ID')

    Notes:
        - This fixture is NOT autouse; tests pay the (small) seeding cost
          only when they need it.
        - The autouse ``run_before_and_after_tests`` fixture sets up the
          backing ``BackendCacheManager`` for us; the helper picks up that
          cache by default.
        - Implementation lives in :mod:`test.util.cache_seeding`.
    """
    from test.util.cache_seeding import seed_col_stats_cache
    return seed_col_stats_cache


def _seed_parquet_col_stats(
    connector,
    table_dfs: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, Optional[str]],
    time_cols: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    from test.util.cache_seeding import seed_col_stats_cache

    time_cols = time_cols or {}
    for table_name, df in table_dfs.items():
        primary_key = primary_keys.get(table_name)
        if (primary_key is not None and primary_key not in df.columns
                and df.index.name == primary_key):
            df = df.reset_index()
        elif df.index.name is not None and df.index.name not in df.columns:
            # See `_seed_from_connector` for the symmetric
            # comment. Parquet round-trips can preserve a column as the
            # pandas index; reset it so the connector-reported column gets
            # stats seeded.
            df = df.reset_index()
        time_col = time_cols.get(table_name)
        date_like_cols = [
            col for col in df.columns
            if (col == time_col or _looks_like_time_col(col))
        ]
        for col in date_like_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.copy()
                df[col] = pd.to_datetime(df[col], errors='coerce')
        seed_col_stats_cache(
            df,
            table_name=table_name,
            data_version=connector.get_data_version(table_name),
            connector_name=connector.name,
            primary_key=primary_key,
            time_col=time_col,
        )


def _column_config_name(column) -> Optional[str]:
    if column is None:
        return None
    if isinstance(column, str):
        return column
    if isinstance(column, dict):
        return column.get('name')
    return getattr(column, 'name', None)


def _looks_like_time_col(column_name: str) -> bool:
    upper_name = column_name.upper()
    return (upper_name in {'DATE', 'TIMESTAMP'} or upper_name.endswith('_DATE')
            or upper_name.endswith('_TIME')
            or upper_name.endswith('_TIMESTAMP'))


def _install_dg_col_stats_fallback() -> Callable[[], None]:
    """Seed local file-backed table stats on first DG cache miss in tests."""
    from test.util.cache_seeding import seed_col_stats_cache

    from statedb.utils.errors import UnsupportedInDiskgraphError

    original_compute_col_stats = Table.compute_col_stats

    def _is_local_file_connector(table: Table) -> bool:
        root_dir = getattr(table.connector, 'root_dir', None)
        if root_dir is None:
            return False
        return not fsutil.is_s3(root_dir) and not fsutil.is_gcs(root_dir)

    def _is_pandas_connector(table: Table) -> bool:
        # DataFrame-built graphs use ``PandasConnector`` (backed
        # by an in-memory ``df_dict``), which has no ``root_dir``. Treat it as
        # seedable so the autouse fallback covers DataFrame-built test graphs
        # (the gap left by the file-backed seed path in ``_create_graph``).
        return (table.has_connector
                and table.connector.__class__.__name__ == 'PandasConnector'
                and getattr(table.connector, 'df_dict', None) is not None
                and table.source_name in table.connector.df_dict)

    def _can_seed_from_connector(table: Table) -> bool:
        if not table.has_connector:
            return False
        if _is_local_file_connector(table):
            return True
        if _is_pandas_connector(table):
            return True
        return table.connector.__class__.__name__ in {
            'BigQueryConnector', 'DatabricksConnector', 'SnowflakeConnector'
        }

    def _read_seed_df(table: Table) -> pd.DataFrame:
        if _is_local_file_connector(table):
            df = table.connector.read_table(table.source_name)
        elif _is_pandas_connector(table):
            # Read the in-memory frame directly; ``read_table`` would also
            # work but adds the ``get_df`` filtering layer, which we don't
            # need for seeding (we want the full schema/dtype payload).
            df = table.connector.df_dict[table.source_name]
        else:
            df = table.connector.read_table_sample(table.source_name,
                                                   num_rows=10000,
                                                   skip_cache=True)

        # Parquet date columns round-trip as object strings (or as
        # ``datetime.date`` python objects) instead of ``datetime64``. The
        # downstream ``compute_supplemental_col_stats_from_series`` infers
        # ``Dtype.string`` for object columns, so the seeded cache entry's
        # fingerprint dtype mismatches the ``Dtype.date`` the test's
        # ``Table`` advertises - causing a cache miss for ``DATE:[MIN,
        # MAX]`` lookups even though we just seeded. Mirror the same date
        # coercion ``_seed_parquet_col_stats`` applies to parquet
        # round-trips so the seeded stats line up with the
        # production-inferred dtype.
        time_col_name = (table.time_col.name if table.has_time_col() else None)
        date_like_cols = [
            col for col in df.columns
            if (col == time_col_name or _looks_like_time_col(col))
        ]
        for col in date_like_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.copy()
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def _seed_from_connector(table: Table) -> None:
        from kumo.connector.utils import cast_df_to_dtypes

        data_version = table.data_version or table.connector.get_data_version(
            table.source_name)
        df = _read_seed_df(table)
        primary_key = table.pkey.name if table.has_pkey() else None
        if (primary_key is not None and primary_key not in df.columns
                and df.index.name == primary_key):
            df = df.reset_index()
        elif df.index.name is not None and df.index.name not in df.columns:
            # Parquet round-trips can preserve a column as
            # the pandas index. The connector's `get_table_metadata` still
            # reports it as a regular column (per
            # ParquetConnector._infer_metadata) and `infer_metadata` adds
            # it to `table.cols`, so the consumer eventually requests
            # stats for it. Reset the index here so the seed helper
            # actually computes/writes those stats.
            df = df.reset_index()
        time_col = table.time_col.name if table.has_time_col() else None

        # Mirror the dtype-coercion pass that
        # `_seed_file_backed_graph_col_stats` applies via the
        # `_create_graph` autouse path. Bare-`Graph` test fixtures
        # bypass that path (the cache miss falls through to this hook
        # instead), but the same dtype gap exists: `compute_col_stats`
        # eventually requests stats based on the table's *declared*
        # dtype (set either via explicit `add_col(..., dtype=...)` or by
        # `infer_metadata`'s connector-metadata fallback for all-NaN
        # cols), while the seed helper otherwise computes stats based on
        # the pandas-inferred dtype. Coerce upfront so the supplemental
        # stat set the helper emits matches what the consumer asks for.
        dtypes_to_cast: Dict[str, Dtype] = {}
        try:
            meta_list = table.connector.get_table_metadata([table.source_name])
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger(__name__).warning(
                "Col-stats seed: get_table_metadata failed for %s: %s",
                table.source_name, exc)
            meta_list = []
        if meta_list:
            for meta_col in meta_list[0].cols:
                if (meta_col.dtype is not None and meta_col.name in df.columns):
                    dtypes_to_cast.setdefault(meta_col.name, meta_col.dtype)
        for col in table.cols:
            if col.dtype is None or col.name not in df.columns:
                continue
            dtypes_to_cast[col.name] = col.dtype
        if table.has_pkey() and table.pkey.dtype is not None and (
                table.pkey.name in df.columns):
            dtypes_to_cast[table.pkey.name] = table.pkey.dtype
        if dtypes_to_cast:
            df = df.copy()
            try:
                df = cast_df_to_dtypes(df, dtypes_to_cast)
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger(__name__).warning(
                    "Col-stats seed: dtype cast failed for %s: %s",
                    table.source_name, exc)

        seed_col_stats_cache(
            df,
            table_name=table.source_name,
            data_version=data_version,
            connector_name=table.connector.name,
            primary_key=primary_key,
            time_col=time_col,
        )

    def _is_cache_miss_error(exc: Exception) -> bool:
        # Two distinct cache-read sites raise ``UnsupportedInDiskgraphError``
        # on miss: ``Table.compute_col_stats`` ("Requested col stats missing
        # from the DiskGraph-populated cache") and ``Table.compute_exact_stats``
        # ("compute_exact_stats: requested col stats missing from the
        # DiskGraph-populated cache"). Both are unblocked by seeding via
        # :func:`seed_col_stats_cache`.
        msg = str(exc)
        return ('Requested col stats missing from the DiskGraph-populated '
                'cache' in msg or 'requested col stats missing from the '
                'DiskGraph-populated cache' in msg)

    def _compute_col_stats_with_seed(table: Table, *args, **kwargs):
        try:
            return original_compute_col_stats(table, *args, **kwargs)
        except UnsupportedInDiskgraphError as exc:
            if not _is_cache_miss_error(exc) or not _can_seed_from_connector(
                    table):
                raise
            _seed_from_connector(table)
            return original_compute_col_stats(table, *args, **kwargs)

    Table.compute_col_stats = _compute_col_stats_with_seed

    # ``Table.compute_exact_stats`` is a separate cache-read site in
    # ``kumo/graph/table.py``. Tests that invoke ``Table.materialize``
    # reach this site before any ``compute_col_stats`` call seeds the
    # cache. Mirror the same seed-on-miss behavior so the autouse
    # fallback covers both cache-read entry points uniformly.
    original_compute_exact_stats = Table.compute_exact_stats

    def _compute_exact_stats_with_seed(table: Table, *args, **kwargs):
        try:
            return original_compute_exact_stats(table, *args, **kwargs)
        except UnsupportedInDiskgraphError as exc:
            if not _is_cache_miss_error(exc) or not _can_seed_from_connector(
                    table):
                raise
            _seed_from_connector(table)
            return original_compute_exact_stats(table, *args, **kwargs)

    Table.compute_exact_stats = _compute_exact_stats_with_seed

    def _restore() -> None:
        Table.compute_col_stats = original_compute_col_stats
        Table.compute_exact_stats = original_compute_exact_stats

    return _restore


def _create_fixed_static_graph(
    tmp_path: pathlib.Path,
    num_users: int,
    num_stores: int,
    num_edges_order: int,
    num_edges_view: int,
    key_name_match: bool = True,
) -> Graph:
    # last user/store is isolated.
    user_df = pd.DataFrame({'USER_ID': np.arange(num_users)})
    store_df = pd.DataFrame({'STORE_ID': np.arange(num_stores)})

    user_store_edge_index_all = np.stack([
        np.repeat(np.arange(num_users - 1), num_stores - 1),
        np.tile(np.arange(num_stores - 1), num_users - 1)
    ])

    order_sampled_idx = np.concatenate(
        (np.repeat(np.arange(num_edges_order / 8), 4),
         np.arange(num_edges_order - int(num_edges_order / 8) * 4))).astype(int)

    order_edge_index = user_store_edge_index_all[:, order_sampled_idx]

    view_sampled_idx = np.concatenate(
        (np.repeat(np.arange(num_edges_order / 6), 3),
         np.arange(num_edges_order - int(num_edges_order / 6) * 3))).astype(int)
    view_edge_index = user_store_edge_index_all[:, view_sampled_idx]

    # deliberately add some duplication
    order_edge_index = np.concatenate(
        [order_edge_index, order_edge_index[:, :10]], axis=1)

    fkey_user = 'USER_ID'
    fkey_store = 'STORE_ID'
    if not key_name_match:
        fkey_user = 'USER_ID_FK'
        fkey_store = 'STORE_ID_FK'

    order_df = pd.DataFrame({
        fkey_user: order_edge_index[0],
        fkey_store: order_edge_index[1],
    })

    view_df = pd.DataFrame({
        'USER_ID': view_edge_index[0],
        'STORE_ID': view_edge_index[1],
    })

    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    store_df.to_parquet(osp.join(root_dir, 'STORES.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))
    view_df.to_parquet(osp.join(root_dir, 'VIEWS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'STORES': store_df,
            'ORDERS': order_df,
            'VIEWS': view_df,
        },
        {
            'USERS': 'USER_ID',
            'STORES': 'STORE_ID',
            'ORDERS': None,
            'VIEWS': None,
        },
    )

    graph = Graph(name='static_graph')
    graph.connect(connector)
    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int))
    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.add(
        name='ORDERS',
        fkeys=[
            dict(name=fkey_user, dst_table='USERS', dst_key='USER_ID',
                 dtype=Dtype.int),
            dict(name=fkey_store, dst_table='STORES', dst_key='STORE_ID',
                 dtype=Dtype.int),
        ],
    )
    graph.add(
        name='VIEWS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
    )

    return graph


def _create_static_graph(
    tmp_path: pathlib.Path,
    num_users: int,
    num_stores: int,
    num_edges_order: int,
    num_edges_view: int,
    key_name_match: bool = True,
) -> Graph:
    # last user/store is isolated.
    user_df = pd.DataFrame({'USER_ID': np.arange(num_users)})
    store_df = pd.DataFrame({'STORE_ID': np.arange(num_stores)})

    user_store_edge_index_all = np.stack([
        np.repeat(np.arange(num_users - 1), num_stores - 1),
        np.tile(np.arange(num_stores - 1), num_users - 1)
    ])
    order_sampled_idx = np.random.permutation(
        (num_users - 1) * (num_stores - 1))[:num_edges_order]
    view_sampled_idx = np.random.permutation(
        (num_users - 1) * (num_stores - 1))[:num_edges_view]
    order_edge_index = user_store_edge_index_all[:, order_sampled_idx]
    view_edge_index = user_store_edge_index_all[:, view_sampled_idx]
    # deliberately add some duplication
    order_edge_index = np.concatenate(
        [order_edge_index, order_edge_index[:, :10]], axis=1)

    fkey_user = 'USER_ID'
    fkey_store = 'STORE_ID'
    if not key_name_match:
        fkey_user = 'USER_ID_FK'
        fkey_store = 'STORE_ID_FK'

    order_df = pd.DataFrame({
        fkey_user: order_edge_index[0],
        fkey_store: order_edge_index[1],
    })

    view_df = pd.DataFrame({
        'USER_ID': view_edge_index[0],
        'STORE_ID': view_edge_index[1],
    })

    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    store_df.to_parquet(osp.join(root_dir, 'STORES.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))
    view_df.to_parquet(osp.join(root_dir, 'VIEWS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'STORES': store_df,
            'ORDERS': order_df,
            'VIEWS': view_df,
        },
        {
            'USERS': 'USER_ID',
            'STORES': 'STORE_ID',
            'ORDERS': None,
            'VIEWS': None,
        },
    )

    graph = Graph(name='static_graph')
    graph.connect(connector)
    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int))
    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.add(
        name='ORDERS',
        fkeys=[
            dict(name=fkey_user, dst_table='USERS', dst_key='USER_ID',
                 dtype=Dtype.int),
            dict(name=fkey_store, dst_table='STORES', dst_key='STORE_ID',
                 dtype=Dtype.int),
        ],
    )
    graph.add(
        name='VIEWS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
    )

    return graph


@pytest.fixture()
def create_temporal_graph(tmp_path: pathlib.Path) -> Callable:
    return _create_temporal_graph


@pytest.fixture()
def create_static_node_pred_graph_v2(tmp_path: pathlib.Path) -> Callable:
    return static_node_pred_test_graph


@pytest.fixture()
def create_static_node_pred_graph(tmp_path: pathlib.Path) -> Callable:
    return functools.partial(static_node_pred_test_graph, tmp_path)


@pytest.fixture()
def create_large_static_node_pred_test_graph(
        tmp_path: pathlib.Path) -> Callable:
    return functools.partial(large_static_node_pred_test_graph, tmp_path)


def add_df(
    graph: Graph,
    df: pd.DataFrame,
    name: str,
    tmp_path: pathlib.Path,
    **kwargs,
):
    path: str = osp.join(tmp_path.resolve().as_posix(), 'parquet')
    os.makedirs(path, exist_ok=True)
    df.to_parquet(osp.join(path, f'{name}.parquet'))

    # The data source is named 'parquet_source' and not 'parquet'
    # to avoid conflict with other fixtures that are using the same
    # data source name and table names (e.g. 'USERS').
    if not graph.has_data_source('parquet_source'):
        graph.connect(CompositeFileConnector('parquet_source', path))

    connector = graph.data_source_dict['parquet_source']
    primary_key = _column_config_name(kwargs.get('pkey'))
    time_col = _column_config_name(kwargs.get('time_col'))
    _seed_parquet_col_stats(
        connector,
        {name: df},
        {name: primary_key},
        {name: time_col},
    )

    graph.add(name, **kwargs)
    return graph


def large_static_node_pred_test_graph(
        tmp_path: pathlib.Path,
        name: str = 'kumo_large_static_test_graph',
        seed: int = 42  # Add a seed parameter for deterministic results
) -> Graph:
    r"""
    Create a large static node prediction test graph with predefined
        properties.

    Args:
        tmp_path (pathlib.Path): Temporary path to save any necessary files.
        name (str): Name of the graph.
            (default: :obj:`kumo_large_static_test_graph`)
        seed (int): Random seed for deterministic results. (default: :obj:`42`)

    Returns:
        Graph: A kumo graph object.
    """
    # Set the random seed for deterministic results
    np.random.seed(seed)

    num_entities = 60
    df = pd.DataFrame({
        'USER_ID':
        list(range(num_entities)),
        'STATUS':
        np.random.choice(['A', 'B', 'C'], size=num_entities),
        'PAYMENTS':
        np.random.randint(10, 51, size=num_entities),
        'DATE':
        (['2000-01-01'] * 10 + ['2000-02-01'] * 5 + ['2000-03-01'] * 10 +
         ['2000-04-01'] * 5 + ['2000-05-01'] * 10 + ['2000-06-01'] * 5 +
         ['2000-07-01'] * 10 + ['2000-08-01'] * 5)
    })
    df = df.set_index('USER_ID')
    graph = Graph(name=name)
    add_df(
        graph,
        df,
        'USERS',
        tmp_path,
        pkey='USER_ID',
        cols=['PAYMENTS', 'STATUS'],
        time_col='DATE',
    )
    graph = graph.infer_metadata().infer_encoders()
    return graph


def static_node_pred_test_graph(
    tmp_path: pathlib.Path,
    mode: str = 'joint',
    name: str = 'kumo_static_test_graph',
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Graph:
    r"""Graph used to test static node prediction queries.
    Args:
        mode (str): 'joint', 'separate' or 'static. If 'joint' is selected,
            entity table and target table will be the same. If 'separate' is
            selected, entity table and target table will be two different
            tables. If 'static' is selected, the tables will be the same as if
            'joint' is selected but without a time column.
        name (str): name of the Graph object.
        add_na_pkeys: If `True`, tables will additionally include rows with
            pd.NA primary keys.
        add_na_timestamps: If `True`, tables will additionally include rows
            with pd.NA timestamps.
    Returns:
        Graph: a kumo graph used for unit tests.
    """
    assert mode in ['joint', 'separate', 'static', 'separate-unique']

    df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'USER_ID_FKEY': [0, 1, 2, 3],
        'STATUS': ['A', 'B', pd.NA, 'B'],
        'target': ['A', 'B', pd.NA, 'B'],
        'PAYMENTS': [10.0, 20.0, 40.0, 80.0],
        'DATE': ['1990-01-01', '1990-01-02', '1990-01-03', '1990-01-04'],
        'STATUS_BOOL': [True, False, False, True],
        'MULTI_STATUS': ['A,B', 'C,D', pd.NA, 'A,B,C'],
    })
    if add_na_pkeys:
        df.loc[len(df)] = [
            pd.NA, pd.NA, 'A', 'B', 10.0, '1990-01-01', True, 'A,B'
        ]
    if add_na_timestamps and mode != 'static':
        df.loc[len(df)] = [4, 4, 'A', 'B', 10.0, pd.NA, True, 'A,B']
    df = df.set_index('USER_ID')
    if mode == 'joint':
        graph = Graph(name=name)
        add_df(
            graph, df, 'USERS', tmp_path, pkey='USER_ID', cols=[
                'PAYMENTS', 'STATUS', 'STATUS_BOOL', 'target', 'MULTI_STATUS'
            ], time_col='DATE')
        graph['USERS', 'MULTI_STATUS'].stype = Stype.multicategorical
        graph['USERS', 'MULTI_STATUS'].multicategorical_delimiter = ','
        return graph.infer_metadata().infer_encoders()
    if mode == 'static':
        df = df.drop(columns=['DATE'])
        graph = Graph(name=name)
        add_df(graph, df, 'USERS', tmp_path, pkey='USER_ID',
               cols=['PAYMENTS', 'STATUS', 'STATUS_BOOL', 'MULTI_STATUS'])
        graph['USERS', 'MULTI_STATUS'].stype = Stype.multicategorical
        graph['USERS', 'MULTI_STATUS'].multicategorical_delimiter = ','
        return graph.infer_metadata().infer_encoders()
    if mode == 'separate':
        entity_df = df.drop(columns=['STATUS'])
        target_df = pd.DataFrame({
            'USER_ID_PKEY': [0, 1, 3],
            'STATUS': ['A', 'B', 'B'],
            'STATUS_BOOL': [True, False, True],
            'MULTI_STATUS': ['A,B', 'C,D', 'A,B,C'],
        })
        graph = Graph(name=name)
        add_df(graph, entity_df, 'USERS', tmp_path, pkey='USER_ID',
               fkeys=['USER_ID_FKEY'], cols=['PAYMENTS'], time_col='DATE')
        add_df(graph, target_df, 'USERS_TARGET', tmp_path, pkey='USER_ID_PKEY',
               cols=['STATUS', 'STATUS_BOOL', 'MULTI_STATUS'])
        graph.link(src_table='USERS', src_key='USER_ID_FKEY',
                   dst_table='USERS_TARGET', dst_key='USER_ID_PKEY')
        graph['USERS_TARGET', 'MULTI_STATUS'].stype = Stype.multicategorical
        graph['USERS_TARGET', 'MULTI_STATUS'].multicategorical_delimiter = ','
        graph = graph.infer_metadata().infer_encoders()
        return graph
    if mode == 'separate-unique':
        entity_df = df.drop(columns=['STATUS'])
        entity_df['USER_ID_FKEY'] = entity_df.index
        target_df = pd.DataFrame({
            'USER_ID': [0, 1, 3],
            'STATUS': ['A', 'B', 'B'],
            'STATUS_BOOL': [True, False, True],
            'MULTI_STATUS': ['A,B', 'C,D', 'A,B,C'],
        })
        target_df = target_df.set_index('USER_ID')

        graph = Graph(name=name)
        add_df(graph, entity_df, 'USERS', tmp_path, pkey='USER_ID',
               cols=['PAYMENTS'], time_col='DATE', fkeys=['USER_ID_FKEY'])
        add_df(graph, target_df, 'USERS_TARGET', tmp_path, pkey='USER_ID',
               cols=['STATUS', 'STATUS_BOOL', 'MULTI_STATUS'])
        graph.link('USERS', 'USER_ID_FKEY', 'USERS_TARGET', 'USER_ID')
        graph['USERS_TARGET', 'MULTI_STATUS'].stype = Stype.multicategorical
        graph['USERS_TARGET', 'MULTI_STATUS'].multicategorical_delimiter = ','
        return graph.infer_links().infer_metadata().infer_encoders()


def _create_temporal_graph(tmp_path: pathlib.Path, num_users: int,
                           num_stores: int, num_edges: int,
                           seed: int = 0) -> Graph:
    np.random.seed(seed)
    # the last user/store are isolated.
    user_df = pd.DataFrame({'USER_ID': np.arange(num_users)})

    store_df = pd.DataFrame({
        'STORE_ID':
        np.arange(num_stores),
        'FEATURE':
        np.arange(num_stores),
        'TIMESTAMP':
        pd.to_datetime(10 * np.arange(num_stores), unit='D'),
    })

    order_value_length_1 = int(num_edges / 3)
    order_value_length_2 = num_edges - 3 * order_value_length_1

    # Note that using randint for `ORDER_VALUE` will cause task creation
    # to crash intermittently as at times train table, val table or test
    # table only contains one type of value.
    order_df = pd.DataFrame({
        'ORDER_ID':
        np.arange(num_edges),
        'USER_ID':
        np.random.randint(0, num_users - 1, (num_edges, )),
        'STORE_ID':
        np.random.randint(0, num_stores - 1, (num_edges, )),
        'ORDER_VALUE':
        np.concatenate((np.array(["1", "2", "3"] * order_value_length_1),
                        np.array(["2"] * order_value_length_2))),
        'ORDER_SCORE':
        np.concatenate((np.array([1, 2, 3] * order_value_length_1),
                        np.array([1] * order_value_length_2))),
        'TIMESTAMP':
        pd.to_datetime(np.arange(num_edges), unit='D'),
    })

    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    store_df.to_parquet(osp.join(root_dir, 'STORES.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'STORES': store_df,
            'ORDERS': order_df
        },
        {
            'USERS': 'USER_ID',
            'STORES': 'STORE_ID',
            'ORDERS': 'ORDER_ID'
        },
        {
            'STORES': 'TIMESTAMP',
            'ORDERS': 'TIMESTAMP'
        },
    )

    graph = Graph(name='temporal_graph')
    graph.connect(connector)
    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int))
    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int),
              cols=['FEATURE'], time_col='TIMESTAMP')
    graph.add(
        name='ORDERS',
        pkey=dict(name='ORDER_ID', dtype=Dtype.int),
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
        cols=['ORDER_VALUE', 'ORDER_SCORE'],
        time_col='TIMESTAMP',
    )
    graph['ORDERS', 'ORDER_VALUE'].stype = Stype.categorical
    graph['ORDERS', 'ORDER_VALUE'].dtype = Dtype.string
    graph['STORES', 'FEATURE'].stype = Stype.numerical
    graph['STORES', 'FEATURE'].dtype = Dtype.int

    graph['ORDERS', 'ORDER_SCORE'].stype = Stype.numerical
    graph['ORDERS', 'ORDER_SCORE'].dtype = Dtype.float

    # Set timestamp_unit for PyArrow 14 compatibility
    graph['STORES'].time_col.timestamp_unit = 'ns'
    graph['ORDERS'].time_col.timestamp_unit = 'ns'

    graph.infer_links()
    graph.infer_encoders()

    return graph


def _temporal_graph(tmp_path: pathlib.Path) -> Graph:
    num_users = 10
    num_stores = 6
    num_edges = 100
    return _create_temporal_graph(tmp_path, num_users, num_stores, num_edges)


@pytest.fixture()
def irregular_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _irregular_graph_fixed


def _irregular_graph_fixed(tmp_path: pathlib.Path) -> Graph:
    user_df = pd.DataFrame({'USER_ID': np.arange(4)})
    store_df = pd.DataFrame({'STORE_ID': np.arange(2)})
    order_df = pd.DataFrame({
        'USER_ID':
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]),
        'STORE_ID':
        np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]),
        'ORDER_VALUE':
        np.array([2, 1, 6, 2, 1, 4, 3, 2, 4, 2]),
        'TIMESTAMP':
        pd.to_datetime([0, 4, 6, 0, 2, 4, 0, 2, 0, 6], unit='D'),
    })
    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    store_df.to_parquet(osp.join(root_dir, 'STORES.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'STORES': store_df,
            'ORDERS': order_df
        },
        {
            'USERS': 'USER_ID',
            'STORES': 'STORE_ID',
            'ORDERS': None
        },
        {'ORDERS': 'TIMESTAMP'},
    )

    graph = Graph()
    graph.connect(connector)
    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int))
    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.add(
        name='ORDERS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
        cols=['ORDER_VALUE'],
        time_col='TIMESTAMP',
    )
    graph['ORDERS', 'ORDER_VALUE'].stype = Stype.numerical
    graph['ORDERS', 'ORDER_VALUE'].dtype = Dtype.float

    # Set timestamp_unit for PyArrow 14 compatibility
    graph['ORDERS'].time_col.timestamp_unit = 'ns'

    graph.infer_links()
    graph.infer_encoders()

    return graph


def _temporal_multilabel_graph_fixed(tmp_path: pathlib.Path, ) -> Graph:
    num_users = 5
    num_timeframe = 9
    user_df = pd.DataFrame({
        'USER_ID': np.arange(num_users),
    })
    order_df = pd.DataFrame({
        'USER_ID':
        np.repeat(np.arange(0, num_users), num_timeframe),
        'MULTI_LABEL_v1': ['', '', '', 'a', 'a|b', '', 'c', 'a', 'a|b'] +
        ['', '', 'c', '', 'd', 'a', '', 'a', ''] +
        ['', '', '', '', '', '', '', '', ''] +
        ['', '', '', '', '', '', '', 'b', 'b'] +
        ['', '', 'a', 'b|c', 'a|b|c', '', 'a', 'a|b', 'a|c'],
        'TIMESTAMP':
        pd.to_datetime(np.tile(np.arange(num_timeframe), num_users), unit='D'),
        'PRICE':
        np.arange(num_users * num_timeframe),
    })
    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'ORDERS': order_df
        },
        {
            'USERS': 'USER_ID',
            'ORDERS': None
        },
        {'ORDERS': 'TIMESTAMP'},
    )

    # Create graph
    graph = Graph(name='temporal_multilabel_graph_fixed')
    graph.connect(connector)

    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int), cols=[])
    graph['USERS'].stype = Stype.numerical
    graph['USERS'].dtype = Dtype.int64

    graph.add(
        name='ORDERS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
        ],
        cols=['MULTI_LABEL_v1', 'PRICE'],
        time_col='TIMESTAMP',
    )
    graph['ORDERS', 'MULTI_LABEL_v1'].stype = Stype.multicategorical
    graph['ORDERS', 'MULTI_LABEL_v1'].dtype = Dtype.string
    graph['ORDERS', 'MULTI_LABEL_v1'].multicategorical_delimiter = '|'
    graph['ORDERS', 'PRICE'].stype = Stype.numerical
    graph['ORDERS', 'PRICE'].dtype = Dtype.int

    # Set timestamp_unit for PyArrow 14 compatibility
    graph['ORDERS'].time_col.timestamp_unit = 'ns'

    graph.infer_links()
    graph.fully_qualify_all_fkeys()
    graph.infer_encoders()

    return graph


def _temporal_graph_fixed(
    tmp_path: pathlib.Path,
    num_users: int = 10,
    num_stores: int = 8,
    num_edges: int = 80,
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
    name: str = 'temporal_graph_fixed',
    store_order: List[int] = None,
) -> Graph:
    user_df = pd.DataFrame({
        'USER_ID': np.arange(num_users),
        'AGE': np.arange(num_users) % 20 + 20,
    })
    store_df = pd.DataFrame({'STORE_ID': np.arange(num_stores)})
    if store_order is not None:
        store_df = store_df.iloc[store_order]
    if add_na_pkeys:
        user_df.loc[len(user_df)] = [pd.NA, 33]
        user_df.loc[len(user_df)] = [pd.NA, 34]
        store_df.loc[len(store_df)] = [pd.NA]

    length = num_edges - int(num_edges / num_stores) * num_stores
    order_value_length_1 = int(num_edges / 3)
    order_value_length_2 = num_edges - 3 * order_value_length_1
    length_2 = num_edges - int(num_edges / 8) * 4

    order_score_length_1 = int(num_edges / 3)
    order_score_length_2 = num_edges - 3 * order_score_length_1
    order_df = pd.DataFrame({
        'ORDER_ID':
        np.arange(0, num_edges),
        'USER_ID':
        np.repeat(np.arange(0, num_users), int(num_edges / num_users)),
        'STORE_ID':
        np.concatenate(
            (np.tile(np.arange(0, num_stores),
                     int(num_edges / num_stores)), np.arange(0, length))),
        'STORE_ID_NUM':
        np.concatenate(
            (np.tile(np.arange(0, num_stores),
                     int(num_edges / num_stores)), np.arange(0, length))),
        'ORDER_VALUE_2':
        np.concatenate((np.array([1, 2, 1] * order_value_length_1),
                        np.array([2] * order_value_length_2))),
        'ORDER_VALUE':
        np.concatenate((np.repeat(np.arange(int(num_edges / 8)),
                                  4), np.arange(length_2))),
        'ORDER_VALUE_3':
        np.concatenate((np.repeat(np.arange(int(num_edges / 8)),
                                  4), np.arange(length_2))),
        'ORDER_SCORE':
        np.concatenate((np.array([1, 2, 3] * order_score_length_1),
                        np.array([1] * order_score_length_2))),
        'TIMESTAMP':
        pd.to_datetime(np.arange(num_edges), unit='D'),
        'IMAGE_URL': ['image_url'] * num_edges,
    })
    if add_na_pkeys:
        # lhs is NA
        order_df.loc[len(order_df)] = [
            len(order_df), pd.NA, 1, 1, 1, 2, 3, 3,
            pd.to_datetime(2, unit='D'), 'image_url'
        ]
        # rhs is NA
        order_df.loc[len(order_df)] = [
            len(order_df), 1, pd.NA, pd.NA, 1, 2, 3, 3,
            pd.to_datetime(2, unit='D'), 'image_url'
        ]
        # both are NA
        order_df.loc[len(order_df)] = [
            len(order_df), pd.NA, pd.NA, pd.NA, 1, 2, 3, 3,
            pd.to_datetime(2, unit='D'), 'image_url'
        ]
    if add_na_timestamps:
        order_df.loc[len(order_df)] = [
            len(order_df), 1, 1, 1, 1, 2, 3, 3, pd.NA, 'image_url'
        ]
        order_df.loc[len(order_df)] = [
            len(order_df), 2, 2, 2, 1, 2, 3, 3, pd.NA, 'image_url'
        ]
    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    store_df.to_parquet(osp.join(root_dir, 'STORES.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'STORES': store_df,
            'ORDERS': order_df
        },
        {
            'USERS': 'USER_ID',
            'STORES': 'STORE_ID',
            'ORDERS': 'ORDER_ID'
        },
        {'ORDERS': 'TIMESTAMP'},
    )

    # Create graph
    graph = Graph(name=name)
    graph.connect(connector)

    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int), cols=['AGE'])
    graph['USERS', 'AGE'].stype = Stype.numerical
    graph['USERS', 'AGE'].dtype = Dtype.int64

    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.add(
        name='ORDERS',
        pkey=dict(name='ORDER_ID', dtype=Dtype.int),
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
        cols=[
            'ORDER_VALUE',
            'ORDER_SCORE',
            'STORE_ID_NUM',
            'ORDER_VALUE_2',
            'IMAGE_URL',
            'ORDER_VALUE_3',
            'TIMESTAMP',
        ],
        time_col='TIMESTAMP',
    )
    graph['ORDERS', 'STORE_ID_NUM'].stype = Stype.numerical
    graph['ORDERS', 'STORE_ID_NUM'].dtype = Dtype.int64
    graph['ORDERS', 'ORDER_VALUE_2'].stype = Stype.numerical
    graph['ORDERS', 'ORDER_VALUE_2'].dtype = Dtype.int64
    graph['ORDERS', 'ORDER_VALUE'].stype = Stype.categorical
    graph['ORDERS', 'ORDER_VALUE'].dtype = Dtype.string
    graph['ORDERS', 'ORDER_VALUE_3'].stype = Stype.numerical
    graph['ORDERS', 'ORDER_VALUE_3'].dtype = Dtype.int64
    graph['ORDERS', 'ORDER_SCORE'].stype = Stype.numerical
    graph['ORDERS', 'ORDER_SCORE'].dtype = Dtype.float

    graph['ORDERS', 'IMAGE_URL'].stype = Stype.image
    graph['ORDERS', 'IMAGE_URL'].dtype = Dtype.string

    graph['ORDERS', 'TIMESTAMP'].stype = Stype.timestamp
    graph['ORDERS', 'TIMESTAMP'].dtype = Dtype.time

    # Set timestamp_unit for PyArrow 14 compatibility
    graph['ORDERS'].time_col.timestamp_unit = 'ns'

    graph.infer_links()
    graph.fully_qualify_all_fkeys()
    graph.infer_encoders()

    return graph


def _small_temporal_graph_fixed(
    tmp_path: pathlib.Path,
    num_users: int = 5,
    num_stores: int = 3,
    num_edges: int = 15,
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Graph:
    return _temporal_graph_fixed(tmp_path, num_users, num_stores, num_edges,
                                 add_na_pkeys, add_na_timestamps)


def _large_temporal_graph_fixed(
    tmp_path: pathlib.Path,
    num_users: int = 10,
    num_stores: int = 8,
    num_edges: int = 560,
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Graph:
    return _temporal_graph_fixed(tmp_path, num_users, num_stores, num_edges,
                                 add_na_pkeys, add_na_timestamps)


def _large_temporal_graph(tmp_path: pathlib.Path) -> Graph:
    # 10x larger than temporal_graph
    num_users = 100
    num_stores = 60
    num_edges = 1000
    return _create_temporal_graph(tmp_path, num_users, num_stores, num_edges)


def _static_graph(tmp_path: pathlib.Path, key_name_match: bool = True) -> Graph:
    num_users = 10
    num_stores = 6
    num_edges_order = 20
    num_edges_view = 30
    return _create_static_graph(tmp_path, num_users, num_stores,
                                num_edges_order, num_edges_view, key_name_match)


def _fixed_static_graph(tmp_path: pathlib.Path,
                        key_name_match: bool = True) -> Graph:
    num_users = 10
    num_stores = 6
    num_edges_order = 20
    num_edges_view = 30
    return _create_fixed_static_graph(tmp_path, num_users, num_stores,
                                      num_edges_order, num_edges_view,
                                      key_name_match)


def _large_static_graph(tmp_path: pathlib.Path, sparse: bool) -> Graph:
    # 50x larger than static_graph
    num_users = 500
    num_stores = 300
    if sparse:
        num_edges_order = 200
        num_edges_view = 100
    else:
        num_edges_order = 1000
        num_edges_view = 1500
    return _create_static_graph(tmp_path, num_users, num_stores,
                                num_edges_order, num_edges_view)


@pytest.fixture()
def temporal_graph(tmp_path: pathlib.Path) -> Graph:
    return _temporal_graph(tmp_path)


@pytest.fixture()
def temporal_graph_at_path(tmp_path: pathlib.Path) -> Graph:
    return _temporal_graph


@pytest.fixture()
def temporal_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _temporal_graph_fixed


@pytest.fixture()
def temporal_multilabel_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _temporal_multilabel_graph_fixed


@pytest.fixture()
def small_temporal_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _small_temporal_graph_fixed


@pytest.fixture()
def large_temporal_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _large_temporal_graph_fixed


@pytest.fixture()
def large_temporal_graph(tmp_path: pathlib.Path) -> Graph:
    return _large_temporal_graph(tmp_path)


@pytest.fixture()
def static_graph() -> Graph:
    return _static_graph


@pytest.fixture()
def fixed_static_graph() -> Graph:
    return _fixed_static_graph


@pytest.fixture()
def large_static_graph(tmp_path: pathlib.Path, request) -> Graph:
    return _large_static_graph(tmp_path, request.param)


@pytest.fixture()
def temporal_multiclass_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _temporal_multiclass_graph_fixed


@pytest.fixture()
def binary_hyperedge_task(tmp_path: pathlib.Path) -> Task:
    graph = _temporal_graph(tmp_path)
    graph.compute_table_stats()

    train_table_df = pd.DataFrame({
        'USER_ID':
        np.random.randint(0, graph['USERS'].num_rows(), (20, )),
        'STORE_ID':
        np.random.randint(0, graph['STORES'].num_rows(), (20, )),
        'TIMESTAMP':
        pd.to_datetime(np.random.randint(80, 100, (20, )), unit='D'),
        'TARGET':
        np.array([0, 1] * 10),
        'FEAT':
        np.random.uniform(size=(20, )),
    })
    train_table_df[NODE_INDEX_COLUMN] = range(train_table_df.shape[0])

    return Task(
        graph,
        data_source=train_table_df,
        task_type=TaskType.BINARY_CLASSIFICATION,
        train_table_name='TRAIN_TABLE',
        target_col=dict(name='TARGET', stype=Stype.categorical, dtype=Dtype.int),
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
        cols=[
            dict(name='FEAT', stype=Stype.numerical, dtype=Dtype.float,
                 encoder=Numerical())
        ],
        time_col='TIMESTAMP',
        task_name='BINARY_HYPEREDGE_TASK',
    ).fixed_split([0.5, 0.25, 0.25])


@pytest.fixture()
def create_lp_graph() -> Callable[[...], Graph]:
    return lp_test_graph


def write_pd_df(df: pd.DataFrame, path: str, suffix: str):
    if suffix == 'parquet':
        df.to_parquet(path)
    elif suffix == 'csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError


def lp_test_graph(
    root_dir: str = None,
    num_entities: int = 2,
    key_name_match: bool = True,
    set_entity_time_col: bool = True,
    set_trans_time_col: bool = True,
    last_entity_zero_amount: bool = False,
    missing_link: bool = False,
    dangling_entity: bool = False,
    set_extra_time_col: bool = False,
    use_parquet: bool = False,
    rhs_entities_in_future: bool = False,
    duplicated_rhs: bool = False,
) -> Graph:
    """Graph used to test link prediction queries. Graph has
    tables `USERS`, `PRODUCTS` and `TRANS`. `TRANS` has foreign
    keys to `USERS` and `PRODUCTS`.
    Args:
        root_dir: Directory where csv files of tables will be saved.
        num_entities: Number of users in `USERS` table.
        key_name_match: If foreign key name in `TRANS` table matches
            primary key name. Set to `False` to test edge case.
        set_time_col: If `True` will set time_col in `TRANS` table.
        last_entity_zero_amount: Set zero amount in `TRANS` to the
            last entity.
        missing_link: If `False` will remove some links between entity
            table and target table.
        dangling_entity: If `True` will add some entity id not connecting
            with target table.
        duplicated_rhs: If `True`, RHS table has duplicates.
    """
    # TODO: Verify that we are able to drop these tables in static link pred
    tables_dir = 'tables-' + uuid4().hex[:16]
    tables_path = osp.join(root_dir, tables_dir)
    os.makedirs(tables_path)
    if use_parquet:
        suffix = 'parquet'
    else:
        suffix = 'csv'

    entity1_path = osp.join(tables_path, 'USERS.' + suffix)
    entity2_path = osp.join(tables_path, 'PRODUCTS.' + suffix)
    trans_path = osp.join(tables_path, 'TRANS.' + suffix)
    entity2_extra_connect_path = osp.join(tables_path,
                                          'PRODUCT_BRANDS.' + suffix)

    # Create user entity table
    ages = list(range(17, 17 + (num_entities) * 8, 8))
    user_ids = list(range(num_entities))
    dates = ['1974-01-01'] * num_entities
    if dangling_entity:
        user_ids += [max(user_ids) + 1 + x for x in list(range(5))]
        ages += [100] * 5
        dates += ['1974-01-01'] * 5
    # dangling_entity
    user_df = pd.DataFrame({'USER_ID': user_ids, 'AGE': ages, 'DATE': dates})
    write_pd_df(user_df, entity1_path, suffix)

    # Create product entity table
    rhs_df = pd.DataFrame({
        'PRODUCT_ID':
        range(2),
        'BRAND': ['PATAGONIA', 'KUMO'],
        'DATE': ['2970-01-01', '2970-01-01']
        if rhs_entities_in_future else ['1970-01-01', '1970-01-01']
    })
    if duplicated_rhs:
        rhs_df = pd.concat([rhs_df, rhs_df, rhs_df], ignore_index=True)
    write_pd_df(rhs_df, entity2_path, suffix)

    # Create product feature table
    product_brands_df = pd.DataFrame({
        'PRODUCT_ID': [0, 1, 0, 0, 1],
        'BRAND': ['PATAGONIA', 'KUMO', 'PATAGONIA', 'PATAGONIA', 'KUMO'],
        'BRAND_VALUE': [5, 5, 10, 20, 5],
        'DATE':
        ['2018-01-01', '2018-01-01', '2018-01-10', '2018-01-20', '2018-01-15']
    })
    write_pd_df(product_brands_df, entity2_extra_connect_path, suffix)

    # Create Transaction table:
    dates = pd.Series(pd.date_range(start='1/1/2018', end='1/30/2018'))
    dates = pd.concat([dates] * num_entities, axis=0, ignore_index=True)
    user_ids = []
    product_ids = []
    amounts = []
    # each user buys each product 15 times.
    if missing_link:
        for i in range(num_entities):
            user_ids += [i] * 30
            if i % 3 == 0:
                product_ids += [0] * 15 + [1] * 15
            else:
                product_ids += [i % 2] * 30
            amounts += [1.0, 2.0, 3.0] * 10
    else:
        for i in range(num_entities):
            user_ids += [i] * 30
            product_ids += [0] * 15 + [1] * 15
            if not last_entity_zero_amount or i != num_entities - 1:
                amounts += [1.0, 2.0, 3.0] * 10
            else:
                amounts += [0.0] * 30

    if key_name_match:
        fkey_user = 'USER_ID'
        fkey_product = 'PRODUCT_ID'
    else:
        fkey_user = 'USER_ID_FK'
        fkey_product = 'PRODUCT_ID_FK'

    df_len = len(dates)
    if dangling_entity:
        # Add an RHS dangling entity
        df_len += 1
        user_ids = user_ids + [1]
        product_ids = product_ids + [99]  # danling fkey not in products
        amounts = amounts + [4.0]
        dates = pd.concat(
            [dates, pd.Series([pd.to_datetime('01/30/2018')])],
            axis=0,
            ignore_index=True,
        )

    trans_df = pd.DataFrame({
        'TRANS_ID': range(df_len),
        fkey_user: user_ids,
        fkey_product: product_ids,
        'AMOUNT': amounts,
        'DATE': dates,
    })
    write_pd_df(trans_df, trans_path, suffix)

    if use_parquet:
        connector = ParquetConnector('parquet', root_dir=tables_path,
                                     na_values='?')
    else:
        connector = CSVConnector('csv', root_dir=tables_path, na_values='?')
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'PRODUCTS': rhs_df,
            'PRODUCT_BRANDS': product_brands_df,
            'TRANS': trans_df,
        },
        {
            'USERS': 'USER_ID',
            'PRODUCTS': 'PRODUCT_ID',
            'PRODUCT_BRANDS': None,
            'TRANS': None,
        },
        {
            'USERS': 'DATE' if set_entity_time_col else None,
            'PRODUCTS': 'DATE',
            'PRODUCT_BRANDS': 'DATE' if set_extra_time_col else None,
            'TRANS': 'DATE' if set_trans_time_col else None,
        },
    )

    graph = Graph('test')
    graph.connect(connector)

    graph.add(name='USERS', pkey='USER_ID', cols=['AGE'])
    if set_entity_time_col:
        graph['USERS'].set_time_col('DATE')
    graph.add(name='PRODUCTS', pkey='PRODUCT_ID', cols=['BRAND'],
              time_col='DATE')
    graph.add(
        name='PRODUCT_BRANDS', cols=['BRAND_VALUE'],
        fkeys=[dict(name='PRODUCT_ID', dst_table='PRODUCTS', dtype=Dtype.int)])
    graph.add(
        name='TRANS',
        fkeys=[
            dict(name=fkey_user, dst_table='USERS', dtype=Dtype.int),
            dict(name=fkey_product, dst_table='PRODUCTS', dtype=Dtype.int),
        ],
        cols=[fkey_user, fkey_product, 'AMOUNT'],
    )

    graph.infer_links()
    graph = graph.infer_metadata()
    graph['TRANS', 'AMOUNT'].stype = Stype.numerical
    graph['TRANS', 'AMOUNT'].dtype = Dtype.float
    graph = graph.infer_encoders()
    if set_trans_time_col:
        graph['TRANS'].set_time_col('DATE')
    if set_extra_time_col:
        graph['PRODUCT_BRANDS'].set_time_col('DATE')

    return graph


def _temporal_multiclass_graph_fixed(tmp_path: pathlib.Path, ) -> Graph:
    num_users = 5
    num_timeframe = 7
    user_df = pd.DataFrame({
        'USER_ID': np.arange(num_users),
    })
    order_df = pd.DataFrame({
        'USER_ID':
        np.repeat(np.arange(0, num_users), num_timeframe),
        'MULTI_CLASS_v1': ['a', 'b', 'c', 'a', 'a', 'b', 'a'] +
        ['c', 'c', 'c', 'b', 'b', 'c', 'c'] +
        ['a', 'b', 'c', 'a', 'a', 'c', 'a'] +
        ['a', 'a', 'b', 'b', 'b', 'c', 'a'] +
        ['b', 'a', 'a', 'c', 'a', 'b', 'b'],
        'TIMESTAMP':
        pd.to_datetime(np.tile(np.arange(num_timeframe), num_users), unit='D'),
        'PRICE':
        np.arange(num_users * num_timeframe),
    })
    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'ORDERS': order_df
        },
        {
            'USERS': 'USER_ID',
            'ORDERS': None
        },
        {'ORDERS': 'TIMESTAMP'},
    )

    # Create graph
    graph = Graph(name='temporal_multilabel_graph_fixed')
    graph.connect(connector)

    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int), cols=[])
    graph['USERS'].stype = Stype.numerical
    graph['USERS'].dtype = Dtype.int64

    graph.add(
        name='ORDERS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
        ],
        cols=['MULTI_CLASS_v1', 'PRICE'],
        time_col='TIMESTAMP',
    )
    graph['ORDERS', 'MULTI_CLASS_v1'].stype = Stype.categorical
    graph['ORDERS', 'MULTI_CLASS_v1'].dtype = Dtype.string
    graph['ORDERS', 'PRICE'].stype = Stype.numerical
    graph['ORDERS', 'PRICE'].dtype = Dtype.int

    # Set timestamp_unit for PyArrow 14 compatibility
    graph['ORDERS'].time_col.timestamp_unit = 'ns'

    graph.infer_links()
    graph.fully_qualify_all_fkeys()
    graph.infer_encoders()

    return graph


@pytest.fixture
@withDatabricksCredentials
def ucws() -> WorkspaceClient:
    connection_info = DatabricksConnectionInfo.from_environment()
    fsutil.impl_ucv.uc_ctx._connect(connection_info)
    client = fsutil.impl_ucv.uc_ctx.client
    # verify access
    client.catalogs.get('ucv_test')
    return client


@pytest.fixture
def temp_ucv(ucws) -> Generator[UCURI, None, None]:
    schema = f'schema-{uuid.uuid4().hex[:16]}'
    volume = f'volume-{uuid.uuid4().hex[:16]}'

    schema_info = ucws.schemas.create(catalog_name='ucv_test', name=schema)
    assert schema_info.full_name is not None
    volume_info = ucws.volumes.create(catalog_name='ucv_test',
                                      schema_name=schema, name=volume,
                                      volume_type=VolumeType.MANAGED)
    assert volume_info.full_name is not None

    # The SQL warehouse runs in Shared mode as its own service principal, which
    # is not the schema creator. Grant the built-in 'account users' group the
    # minimum privileges needed so SQL queries in these tests can access the
    # dynamically created schema and volume.
    ucws.grants.update(
        SecurableType.SCHEMA.value,
        schema_info.full_name,
        changes=[
            PermissionsChange(principal='account users',
                              add=[Privilege.USE_SCHEMA])
        ],
    )
    ucws.grants.update(
        SecurableType.VOLUME.value,
        volume_info.full_name,
        changes=[
            PermissionsChange(principal='account users',
                              add=[Privilege.READ_VOLUME])
        ],
    )

    yield UCURI(catalog='ucv_test', uc_schema=schema, volume=volume, path="")

    ucws.volumes.delete(volume_info.full_name)
    ucws.schemas.delete(schema_info.full_name)


def _dense_small_temporal_graph_fixed(
    tmp_path: pathlib.Path,
    num_users: int = 5,
    num_stores: int = 3,
    num_edges: int = 15,
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
) -> Graph:
    """
    Deterministic and sparser version:
    Only include an edge if (user_id + t + store_id) % 2 == 0.
    ORDER_VALUE cycles through value_0 to value_4.
    Ensures all 5 values are possible, but the table is sparser and
    deterministic. Uses 12 timestamps per user.
    """
    user_df = pd.DataFrame({
        'USER_ID': np.arange(num_users),
        'AGE': np.arange(num_users) % 20 + 20,
    })
    store_df = pd.DataFrame({'STORE_ID': np.arange(num_stores)})
    if add_na_pkeys:
        user_df.loc[len(user_df)] = [pd.NA, 33]
        user_df.loc[len(user_df)] = [pd.NA, 34]
        store_df.loc[len(store_df)] = [pd.NA]

    user_ids = []
    store_ids = []
    store_id_nums = []
    order_values = []
    timestamps = []

    for user_id in range(num_users):
        for t in range(12):  # 12 timestamps per user
            for store_id in range(num_stores):
                if (user_id + t + store_id) % 2 == 0:
                    user_ids.append(user_id)
                    store_ids.append(store_id)
                    store_id_nums.append(store_id)
                    order_values.append(f"value_{(user_id + t + store_id) % 5}")
                    timestamps.append(pd.to_datetime(t, unit='D'))

    order_df = pd.DataFrame({
        'USER_ID': user_ids,
        'STORE_ID': store_ids,
        'STORE_ID_NUM': store_id_nums,
        'ORDER_VALUE': order_values,
        'TIMESTAMP': timestamps,
    })

    if add_na_pkeys:
        order_df.loc[len(order_df)] = [
            pd.NA, 1, 1, 'value_0',
            pd.to_datetime(2, unit='D')
        ]
    if add_na_timestamps:
        order_df.loc[len(order_df)] = [1, 1, 1, 'value_0', pd.NA]

    root_dir = tmp_path.resolve().as_posix()
    user_df.to_parquet(osp.join(root_dir, 'USERS.parquet'))
    store_df.to_parquet(osp.join(root_dir, 'STORES.parquet'))
    order_df.to_parquet(osp.join(root_dir, 'ORDERS.parquet'))

    connector = ParquetConnector(name='parquet', root_dir=root_dir)
    _seed_parquet_col_stats(
        connector,
        {
            'USERS': user_df,
            'STORES': store_df,
            'ORDERS': order_df
        },
        {
            'USERS': 'USER_ID',
            'STORES': 'STORE_ID',
            'ORDERS': None
        },
        {'ORDERS': 'TIMESTAMP'},
    )

    graph = Graph(name='dense_small_temporal_graph_fixed')
    graph.connect(connector)

    graph.add('USERS', pkey=dict(name='USER_ID', dtype=Dtype.int), cols=['AGE'])
    graph['USERS', 'AGE'].stype = Stype.numerical
    graph['USERS', 'AGE'].dtype = Dtype.int64

    graph.add('STORES', pkey=dict(name='STORE_ID', dtype=Dtype.int))
    graph.add(
        name='ORDERS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='STORE_ID', dst_table='STORES', dtype=Dtype.int),
        ],
        cols=[
            'ORDER_VALUE',
            'STORE_ID_NUM',
            'TIMESTAMP',
        ],
        time_col='TIMESTAMP',
    )
    graph['ORDERS', 'STORE_ID_NUM'].stype = Stype.numerical
    graph['ORDERS', 'STORE_ID_NUM'].dtype = Dtype.int64
    graph['ORDERS', 'ORDER_VALUE'].stype = Stype.categorical
    graph['ORDERS', 'ORDER_VALUE'].dtype = Dtype.string

    graph['ORDERS', 'TIMESTAMP'].stype = Stype.timestamp
    graph['ORDERS', 'TIMESTAMP'].dtype = Dtype.time

    # Set timestamp_unit for PyArrow 14 compatibility
    graph['ORDERS'].time_col.timestamp_unit = 'ns'

    graph.infer_links()
    graph.fully_qualify_all_fkeys()
    graph.infer_encoders()

    return graph


@pytest.fixture()
def dense_small_temporal_graph_fixed(tmp_path: pathlib.Path) -> Callable:
    return _dense_small_temporal_graph_fixed
