# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``_apply_lp_resolution_plan`` (FK-free LP resolution mode).

TT-NG link-prediction TRAINING resolution-mode plan selection (FK-free
redesign, ttng-lp-fkfree-resolution-design.md S1). The helper post-processes
the diskgraph TT plan JSON so that, when ``list_distinct_target_is_array`` is
True (the gate) for a link-prediction plan, the plan declares:

  * ``plan['lp_resolution']`` — a list of ``RefResolveSpec`` dicts the special
    tt-ingest resolves to graph node-id rowpack columns (NO FK edge / NO
    ``seed.edges.bin``):
      - ``ENTITY1`` (scalar) -> ``entity_node_id``   in the entity table
      - ``TARGET1`` (list)   -> ``target_node_ids``   in the RHS LP table

The emitted dict MUST round-trip into the Rust
``DiskGraphTTPlan.lp_resolution: Option<Vec<RefResolveSpec>>`` whose
``RefResolveSpec`` is ``#[serde(rename_all = "snake_case")]`` with fields
``src_col`` / ``dst_table`` / ``kind`` (enum ``"scalar"`` | ``"list"``) /
``node_id_col`` — so the keys + enum strings are pinned exactly below.

When the gate is False the plan JSON must be returned byte-for-byte unchanged
(the exploded path stays untouched), and no ``KUMO_LP_DELIVERY`` / ``list_fk``
machinery may appear.
"""
from __future__ import annotations

import json

import pytest

# The helper only needs the stdlib ``json``, but importing the module triggers
# ``kumo/__init__.py`` which requires torch. Skip cleanly where torch is absent
# (infra), so the test is real in CI but does not false-fail in lean envs.
try:
    from kumo.data.generic.diskgraph_train_table import (
        _apply_lp_resolution_plan,
    )
except Exception as exc:  # pragma: no cover - env dependent
    pytest.skip(f"kumo import unavailable ({exc!r})", allow_module_level=True)


def _lp_plan_json(*, explode: bool) -> str:
    """A minimal LP plan JSON shaped like the diskgraph compiler output."""
    return json.dumps({
        'entity': {
            'table': 'users',
            'column': 'user_id',
            'output_time_col_name': 'TIMESTAMP',
        },
        'prediction_type': {
            'link': {
                'explode': explode,
                'include_frequency': True,
            },
        },
    })


def test_resolution_plan_emits_two_specs_with_exact_serde_shape() -> None:
    out = _apply_lp_resolution_plan(
        _lp_plan_json(explode=True),
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
    )
    plan = json.loads(out)

    # ``explode`` is no longer emitted/modified: DG dropped the field from
    # ``PredictionType::Link`` (array/FK-free is the only LP mode), so the
    # helper leaves whatever the compiler produced untouched.
    assert plan['prediction_type']['link']['explode'] is True

    specs = plan['lp_resolution']
    assert isinstance(specs, list) and len(specs) == 2

    entity_spec, target_spec = specs

    # ENTITY1 -> entity_node_id, scalar, resolved against the entity table.
    # Exact keys + enum string the Rust RefResolveSpec expects.
    assert entity_spec == {
        'src_col': 'ENTITY1',
        'dst_table': 'users',
        'kind': 'scalar',
        'node_id_col': 'entity_node_id',
    }
    # TARGET1 -> target_node_ids, list (array target), RHS LP table.
    assert target_spec == {
        'src_col': 'TARGET1',
        'dst_table': 'products',
        'kind': 'list',
        'node_id_col': 'target_node_ids',
    }
    # Exact key set (no extra fields that would break serde deny/round-trip).
    assert set(entity_spec) == {'src_col', 'dst_table', 'kind', 'node_id_col'}
    assert set(target_spec) == {'src_col', 'dst_table', 'kind', 'node_id_col'}
    # Enum strings are snake_case scalars only.
    assert entity_spec['kind'] in ('scalar', 'list')
    assert target_spec['kind'] in ('scalar', 'list')

    # No legacy delivery toggle / list-FK machinery leaks into the plan.
    assert 'KUMO_LP_DELIVERY' not in out
    assert 'list_fk' not in out
    assert 'materialize_edges' not in out
    assert 'fkey_mapping' not in plan


def test_lp_rhs_is_temporal_threaded_into_plan_true() -> None:
    """Ingest-gate data condition: ``lp_rhs_is_temporal=True`` is written to the
    plan so DG builds ``bucket_index`` (gate = ``lp_resolution.is_some() &&
    lp_rhs_is_temporal``)."""
    out = _apply_lp_resolution_plan(
        _lp_plan_json(explode=True),
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
        lp_rhs_is_temporal=True,
    )
    plan = json.loads(out)
    assert plan['lp_rhs_is_temporal'] is True


def test_lp_rhs_is_temporal_defaults_false_atemporal_rhs() -> None:
    """Ingest-gate data condition: an atemporal RHS (e.g. amazon ``products``)
    leaves ``lp_rhs_is_temporal=False`` -> DG builds NO ``bucket_index`` -> the
    load gate also stays flat. Default (unset) is False."""
    out = _apply_lp_resolution_plan(
        _lp_plan_json(explode=True),
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
    )
    plan = json.loads(out)
    assert plan['lp_rhs_is_temporal'] is False


def test_explicit_entity_table_overrides_plan_lookup() -> None:
    out = _apply_lp_resolution_plan(
        _lp_plan_json(explode=True),
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='accounts',
        lp_rhs_entity_table='products',
    )
    plan = json.loads(out)
    assert plan['lp_resolution'][0]['dst_table'] == 'accounts'


def test_link_resolution_requires_rhs_table() -> None:
    with pytest.raises(RuntimeError, match='rhs_node_type must be resolved'):
        _apply_lp_resolution_plan(
            _lp_plan_json(explode=True),
            list_distinct_target_is_array=True,
            is_link_pred=True,
            entity_table='users',
            lp_rhs_entity_table=None,
        )


def test_gate_off_non_lp_returns_plan_byte_for_byte_unchanged() -> None:
    # Not link prediction (array flag irrelevant): plan returned unchanged.
    src = _lp_plan_json(explode=True)
    assert _apply_lp_resolution_plan(
        src,
        list_distinct_target_is_array=True,
        is_link_pred=False,
        entity_table='users',
        lp_rhs_entity_table='products',
    ) == src
    # Does not emit an lp_resolution / leave the toggle behind.
    assert 'lp_resolution' not in src


def test_lp_flag_off_returns_plan_unchanged() -> None:
    # Array-mode is now the single source of truth decided upstream by
    # ``allow_array_targets()`` (always True under TT-NG), so the helper simply
    # honors the passed value: with the flag OFF (non-TT-NG, non-array) it is a
    # no-op and returns the plan byte-for-byte unchanged.
    src = _lp_plan_json(explode=True)
    assert _apply_lp_resolution_plan(
        src,
        list_distinct_target_is_array=False,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
    ) == src
    assert 'lp_resolution' not in src


def test_lp_flag_on_emits_full_resolution() -> None:
    # With array-mode True (the value TT-NG supplies from config), the helper
    # emits the full FK-free ENTITY1/TARGET1 node-id resolution.
    src = _lp_plan_json(explode=True)
    out = _apply_lp_resolution_plan(
        src,
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
    )
    specs = json.loads(out)['lp_resolution']
    assert len(specs) == 2
    assert specs[0]['node_id_col'] == 'entity_node_id'
    assert specs[1] == {
        'src_col': 'TARGET1',
        'dst_table': 'products',
        'kind': 'list',
        'node_id_col': 'target_node_ids',
    }


def test_static_lp_emits_exclude_spec() -> None:
    """Static LP appends a third RefResolveSpec resolving gen's
    ``exclude_target_keys`` (prior-stage RHS keys) into ``exclude_node_ids``,
    against the SAME RHS table as the positives."""
    out = _apply_lp_resolution_plan(
        _lp_plan_json(explode=True),
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
        is_static_lp=True,
    )
    specs = json.loads(out)['lp_resolution']
    assert len(specs) == 3
    assert specs[2] == {
        'src_col': 'exclude_target_keys',
        'dst_table': 'products',
        'kind': 'list',
        'node_id_col': 'exclude_node_ids',
    }
    assert set(specs[2]) == {'src_col', 'dst_table', 'kind', 'node_id_col'}


def test_temporal_lp_no_exclude_spec() -> None:
    """Temporal LP (is_static_lp=False) never emits the exclude spec —
    temporal past-visit is unported."""
    out = _apply_lp_resolution_plan(
        _lp_plan_json(explode=True),
        list_distinct_target_is_array=True,
        is_link_pred=True,
        entity_table='users',
        lp_rhs_entity_table='products',
        is_static_lp=False,
    )
    specs = json.loads(out)['lp_resolution']
    assert len(specs) == 2
    assert 'exclude_target_keys' not in out


if __name__ == '__main__':  # pragma: no cover
    # Allow running the pure-logic assertions without pytest/torch by calling
    # the helper directly (it only depends on stdlib json).
    raise SystemExit(pytest.main([__file__, '-v']))
