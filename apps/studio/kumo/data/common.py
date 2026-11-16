# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from enum import Enum

DATA_BACKEND_THREAD_POOL = ThreadPoolExecutor(
    thread_name_prefix='DataBackendCache-Threadpool')


class TableRefreshStage(str, Enum):
    INGEST_BASE_TABLE = 'ingest_base_table'  # Only applicable for View
    INGEST_TABLE = 'ingest_table'
    COMPUTE_MIN_COL_STATS = 'compute_min_col_stats'
    COMPUTE_FULL_COL_STATS = 'compute_full_col_stats'
    COMPLETED = 'completed'


class MaterializationStage(Enum):
    r"""Tracking the stage at which a Kumo table at a given data version has
    been materialized. A table's data can be materialized through the
    following stages:

    * INGEST: Customer's raw data in its original location/format gets
    ingested by Kumo into Kumo's canonical data format.

    * INDEX: The table is materialized with the index column (as part of table
    materialization)
    """
    INGEST = 'ingest'
    INDEX = 'index'


class EdgeFormatType(str, Enum):
    r"""Different implementations for the edge index materialization format.
        * COO: A typical COO-based representation for materialized
            edges. This is represented in edge parquet files as

        row   col
        0     20
        0     21
        1     24
        ...   ...

        * ROW_AGG_COO: An adjacency list representation for materialized
            edges. This is represented in edge parquet files as

        row   col
        0     [20, 21]
        1     [24, ...]
        ...   ...
    """
    COO = 'COO'
    ROW_AGG_COO = 'ROW_AGG_COO'


class NodeIdxType(str, Enum):
    r"""Different implementations for node index materialization format.
        * GLOBAL_INDEXED: Node indices are represented by global indices,
            where a table with N rows has node indices ranging from 0..N-1.
            This was the original implementation of nodes in the Kumo graph,
            as it is easily amenable to a CSC/CSR-based graph format that is
            used in PyG today.
        * KEY_HASHED: Node indices are represented by hashing primary keys
            (if present), or a unique index on a table produced by
            materialization. This implementation is represented as an
            adjacency-based graph, which pays a small performance penalty
            but enables easy online lookup downstream.
    """
    GLOBAL_INDEXED = 'global_indexed'
    KEY_HASHED = 'key_hashed'
