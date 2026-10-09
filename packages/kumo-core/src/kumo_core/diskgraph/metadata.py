# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive graph schema from a DiskGraph index's ``index_metadata.json``.

The metadata file is written by ``diskgraph-ingest`` and always lives at
``{index_path}/index_metadata.json``.  This module extracts table names,
primary keys, foreign-key specs, and normalised edge types — everything
that ``compare_model_quality.py`` (and future callers) need without
hard-coding dataset-specific constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EdgeType = Tuple[str, str, str]


@dataclass
class DiskgraphSchema:
    """Graph schema derived from a diskgraph index."""

    table_names: List[str]
    """Sorted node-type names (e.g. ``["account", "card", ...]``)."""

    pkey_dict: Dict[str, str]
    """``{table_name: pk_column}``."""

    fk_specs: List[Tuple[str, str]]
    """``[("src.fk_col", "dst.pk_col"), ...]`` — one per forward FK edge."""

    raw_edge_types: List[EdgeType]
    """All raw edge types as stored in index metadata."""

    edge_types: List[EdgeType]
    """All edge types (forward + reverse) in short normalised form."""

    edge_type_remap: Dict[EdgeType, EdgeType]
    """``{raw_diskgraph_tuple: normalised_tuple}`` for edges that changed."""

    node_types: List[str] = field(default_factory=list)
    """Alias of *table_names* (for call-sites that expect this name)."""


def _parse_fk_relation(
    name: str,
    src_type: str,
    dst_type: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract FK / PK column names from a diskgraph relation name.

    Expected convention: ``{src_type}_{fk_col}_to_{dst_type}_{pk_col}``.

    Returns ``(fk_col, pk_col)`` on success, ``(None, None)`` on failure.
    """
    prefix = f"{src_type}_"
    sep = f"_to_{dst_type}_"

    if not name.startswith(prefix):
        return None, None

    rest = name[len(prefix):]
    idx = rest.find(sep)
    if idx < 0:
        return None, None

    fk_col = rest[:idx]
    pk_col = rest[idx + len(sep):]

    if not fk_col or not pk_col:
        return None, None

    return fk_col, pk_col


def normalize_diskgraph_edge_type(
    src_type: str,
    rel_name: str,
    dst_type: str,
) -> EdgeType:
    """Convert DiskGraph FK relation names to Kumo edge type names."""
    raw_et: EdgeType = (src_type, rel_name, dst_type)

    if rel_name.endswith("_reverse"):
        fwd_name = rel_name[:-len("_reverse")]
        fwd_src = dst_type
        fwd_dst = src_type
        fk_col, _pk_col = _parse_fk_relation(fwd_name, fwd_src, fwd_dst)
        if fk_col is not None:
            return (src_type, f"rev_{fk_col}", dst_type)
        return raw_et

    fk_col, pk_col = _parse_fk_relation(rel_name, src_type, dst_type)
    if fk_col is not None and pk_col is not None:
        return (src_type, fk_col, dst_type)
    return raw_et


def load_diskgraph_schema(index_path: str) -> DiskgraphSchema:
    """Read ``index_metadata.json`` and derive a full :class:`DiskgraphSchema`.

    Parameters
    ----------
    index_path:
        Path to a diskgraph index directory (must contain
        ``index_metadata.json``).

    Raises
    ------
    FileNotFoundError
        If ``index_metadata.json`` does not exist.
    """
    meta_path = Path(index_path) / "index_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # -- node types / primary keys -----------------------------------------
    table_names: List[str] = []
    pkey_dict: Dict[str, str] = {}
    for nt in meta.get("node_types", []):
        name = nt["name"]
        table_names.append(name)
        pk_map = nt.get("primary_key_map")
        if pk_map:
            pkey_dict[name] = pk_map["column"]
    table_names.sort()

    # -- edge relations ----------------------------------------------------
    fk_specs: List[Tuple[str, str]] = []
    raw_edge_types: List[EdgeType] = []
    edge_types: List[EdgeType] = []
    edge_type_remap: Dict[EdgeType, EdgeType] = {}

    for rel in meta.get("edge_relations", []):
        rel_name: str = rel["name"]
        src_type: str = rel["src_type"]
        dst_type: str = rel["dst_type"]
        raw_et: EdgeType = (src_type, rel_name, dst_type)
        raw_edge_types.append(raw_et)

        if rel_name.endswith("_reverse"):
            # Reverse edge — derive short name from the forward name.
            fwd_name = rel_name[:-len("_reverse")]
            fwd_src = dst_type  # forward's src == reverse's dst
            fwd_dst = src_type  # forward's dst == reverse's src
            fk_col, _pk_col = _parse_fk_relation(fwd_name, fwd_src, fwd_dst)
            if fk_col is not None:
                norm_et: EdgeType = (src_type, f"rev_{fk_col}", dst_type)
            else:
                norm_et = raw_et
        else:
            # Forward edge — parse FK / PK columns.
            fk_col, pk_col = _parse_fk_relation(rel_name, src_type, dst_type)
            if fk_col is not None and pk_col is not None:
                fk_specs.append(
                    (f"{src_type}.{fk_col}", f"{dst_type}.{pk_col}"))
                norm_et = (src_type, fk_col, dst_type)
            else:
                norm_et = raw_et

        edge_types.append(norm_et)
        if raw_et != norm_et:
            edge_type_remap[raw_et] = norm_et

    return DiskgraphSchema(
        table_names=table_names,
        pkey_dict=pkey_dict,
        fk_specs=fk_specs,
        raw_edge_types=raw_edge_types,
        edge_types=edge_types,
        edge_type_remap=edge_type_remap,
        node_types=list(table_names),
    )
