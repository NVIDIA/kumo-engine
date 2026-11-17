# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hetero_gnn_gi_degree_t_v1`: nested GNN-GI over transition gates (stub).

Placeholder created with the shared walk foundation; the implementation is
owned by the degree-T pod. Every entry point refuses, so configuring the
method fails before hooks (`raise`) or yields `unsupported` rows (`record`)
and never falls back to another rule.
"""
from kumo_core.explain.walk_attribution import (
    DEGREE_T_METHOD,
    UnsupportedWalkAttribution,
)

METHOD_ID = DEGREE_T_METHOD


def _not_implemented():
    return UnsupportedWalkAttribution(f'{METHOD_ID} is not implemented')


def qualify(module, config):
    raise _not_implemented()


def make_rule(computation, selection, config):
    raise _not_implemented()


def summarize(computation, selection, config, enumeration):
    raise _not_implemented()


def validate_row(row):
    raise _not_implemented()
