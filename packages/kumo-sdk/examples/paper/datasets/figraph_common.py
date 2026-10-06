# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared FiGraph constants and helpers for benchmarking datasets."""

from __future__ import annotations

# Parquet filenames under per-year subdirectories of data_root.
LISTED_PARQUET = "listed_companies.parquet"
EDGES_PARQUET = "edges.parquet"

# Node type prefixes (paper: L=listed, U=unlisted, A=audit, H=human, O=other).
PREFIX_L = "L"
PREFIX_U = "U"
PREFIX_A = "A"
PREFIX_H = "H"
PREFIX_O = "O"
PREFIX_ORDER = (PREFIX_L, PREFIX_U, PREFIX_A, PREFIX_H, PREFIX_O)

# Table names for the relational graph: 5 entity tables + directed
# edge tables.
LISTED_TABLE = "listed_companies"
UNLISTED_TABLE = "unlisted_companies"
AUDIT_TABLE = "audit_institutions"
HUMANS_TABLE = "humans"
OTHER_TABLE = "other_entities"

ENTITY_TABLE_BY_PREFIX = {
    PREFIX_L: LISTED_TABLE,
    PREFIX_U: UNLISTED_TABLE,
    PREFIX_A: AUDIT_TABLE,
    PREFIX_H: HUMANS_TABLE,
    PREFIX_O: OTHER_TABLE,
}

FIGRAPH_S3_BASE = "s3://kumo-public-datasets/figraph"

# Years available in the public FiGraph release.
MIN_YEAR = 2014
MAX_YEAR = 2022

# Temporal split for benchmarks.
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021

# Additional table in the relational graph: temporal features for
# listed companies, one row per (listed entity, year).
LISTED_FEATURES_TABLE = "listed_company_features"

# Enterprise PQL: binary classification, predict fraud (Label=1) per
# company-year (feature row). Matches task table and single-table
# benchmarks (train 2014–2020, test 2021–2022 applied via split).
FIGRAPH_ENTERPRISE_QUERY = """
PREDICT listed_company_features.Label = 1
FOR EACH listed_company_features.feature_row_id
WHERE listed_company_features.Label >= 0
"""


def node_prefix(node_id: str) -> str:
    """Return the single-letter type prefix of a FiGraph node ID."""
    if not node_id or not isinstance(node_id, str):
        return ""
    s = str(node_id).strip().upper()
    return s[0] if s and s[0] in "LUAHO" else ""
