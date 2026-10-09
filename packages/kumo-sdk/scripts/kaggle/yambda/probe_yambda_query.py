# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What matrix rows 6, 7, 10, 13 and 15 compile to, before any data exists.

Five rows, one fixture, one graph contract:

    row 6   PREDICT LIST_DISTINCT(interactions.event_type)
            CLASSIFY TOP 5   FOR EACH users.user_id
    row 7   PREDICT LIST_DISTINCT(interactions.event_type, 0, 30, days)
            RANK TOP 5       FOR EACH users.user_id
    row 10  PREDICT interactions.played_ratio_pct
            FOR EACH interactions.interaction_id
    row 13  PREDICT LIST_DISTINCT(interactions.item_id)
            RANK TOP 100     FOR EACH users.user_id
    row 15  PREDICT LIST_DISTINCT(interactions.item_id, 0, 30, days)
            RANK TOP 100     FOR EACH users.user_id

Three questions decide expressibility, and all three are answerable from the
compiler in a second with no data:

1.  **Row 6.**  ``kumopql/validator/time_range_validator.py`` allows a static
    (unwindowed) ``LIST_DISTINCT`` in exactly one cell: ``RANK`` over a foreign
    key.  NOAA rows 5/8 hit the non-foreign-key side; BCCC row 14 hit the
    ``CLASSIFY``-over-a-genuine-foreign-key side.  ``event_type`` is a plain
    categorical column *and* row 6 asks for ``CLASSIFY``, so it should be
    rejected twice over.  Measured, not assumed.

2.  **Rows 13 and 15 against one graph.**  BCCC row 14 S2 established that
    "static link prediction" is a property of the *graph*: with a time column on
    the fact table a static ``LIST_DISTINCT`` is rejected ("time range
    required"), and without one a windowed target is rejected ("Missing a
    'Create date' column").  The Yambda graph contract says ``event_time`` *is*
    the event time, so row 13 and row 15 may not be jointly expressible.  The
    2x2 is compiled here.

3.  **The anchor edge.**  ``interactions`` carries two foreign keys and
    ``join_validator.py::_find_keys`` does ``keys = curr_keys`` -- the last
    declared foreign key wins, silently, with no PQL syntax to override it.
    Unlike BCCC (where both keys pointed at *the same* table) Yambda's two keys
    point at different tables, so the ambiguity may not bite; both orders are
    compiled to find out.

Nothing here reads the fixture.  ``kumo_core.pquery.normalize_graph`` takes a
fully resolved portable graph, so the graph is synthesized from the schema
``probe_yambda_fixture.py`` read off the stage.  The plans printed are the real
ones: ``run_yambda_local.py`` compiles the same queries through the same
functions on the way to training.
"""
import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

#: The five published rows, verbatim modulo Snowflake's upper-cased names.
QUERIES: Dict[str, str] = {
    'row 6  multilabel classification (as published)':
    ('PREDICT LIST_DISTINCT(interactions.EVENT_TYPE)\n'
     'CLASSIFY TOP 5\n'
     'FOR EACH users.USER_ID'),
    'row 7  multilabel ranking (as published)':
    ('PREDICT LIST_DISTINCT(interactions.EVENT_TYPE, 0, 30, days)\n'
     'RANK TOP 5\n'
     'FOR EACH users.USER_ID'),
    'row 10 regression (as published)':
    ('PREDICT interactions.PLAYED_RATIO_PCT\n'
     'FOR EACH interactions.INTERACTION_ID'),
    'row 13 static link prediction (as published)':
    ('PREDICT LIST_DISTINCT(interactions.ITEM_ID)\n'
     'RANK TOP 100\n'
     'FOR EACH users.USER_ID'),
    'row 15 temporal link prediction (as published)':
    ('PREDICT LIST_DISTINCT(interactions.ITEM_ID, 0, 30, days)\n'
     'RANK TOP 100\n'
     'FOR EACH users.USER_ID'),
}

#: Neighbouring shapes that say *why* each row compiles the way it does.
VARIANTS: Dict[str, str] = {
    'row 6 rescued: windowed CLASSIFY (the NOAA row 5 shape)':
    ('PREDICT LIST_DISTINCT(interactions.EVENT_TYPE, 0, 30, days)\n'
     'CLASSIFY TOP 5\n'
     'FOR EACH users.USER_ID'),
    'row 6 rescued: static RANK over the categorical column':
    ('PREDICT LIST_DISTINCT(interactions.EVENT_TYPE)\n'
     'RANK TOP 5\n'
     'FOR EACH users.USER_ID'),
    'row 7 with CLASSIFY instead of RANK':
    ('PREDICT LIST_DISTINCT(interactions.EVENT_TYPE, 0, 30, days)\n'
     'CLASSIFY TOP 5\n'
     'FOR EACH users.USER_ID'),
    'row 13 with CLASSIFY instead of RANK (expected: rejected)':
    ('PREDICT LIST_DISTINCT(interactions.ITEM_ID)\n'
     'CLASSIFY TOP 100\n'
     'FOR EACH users.USER_ID'),
    'row 13/15 anchored on items instead of users':
    ('PREDICT LIST_DISTINCT(interactions.USER_ID, 0, 30, days)\n'
     'RANK TOP 100\n'
     'FOR EACH items.ITEM_ID'),
    'row 15 with a 7-day window':
    ('PREDICT LIST_DISTINCT(interactions.ITEM_ID, 0, 7, days)\n'
     'RANK TOP 100\n'
     'FOR EACH users.USER_ID'),
    'row 15 anchored on the fact table (expected: no fkey)':
    ('PREDICT LIST_DISTINCT(interactions.ITEM_ID, 0, 30, days)\n'
     'RANK TOP 100\n'
     'FOR EACH interactions.INTERACTION_ID'),
    'row 10 anchored on users instead (an aggregate)':
    ('PREDICT AVG(interactions.PLAYED_RATIO_PCT, 0, 30, days)\n'
     'FOR EACH users.USER_ID'),
}


def col(name: str, dtype: str, stype: str) -> Dict[str, str]:
    return {'name': name, 'dtype': dtype, 'stype': stype}


def portable_graph(user_last: bool = True,
                   with_time: bool = True) -> Dict[str, Any]:
    r"""The three-table Yambda graph as a resolved portable-graph mapping.

    ``user_last`` controls the **order** the two foreign keys are declared in
    and nothing else; ``with_time`` drops ``EVENT_TIME`` from ``interactions``
    so question 2 above can be answered.

    Column list is the fixture's, verbatim from ``INFER_SCHEMA`` on the stage:
    ``interaction_id, user_id, item_id, event_time, event_type, is_organic,
    played_ratio_pct, track_length_seconds``.  ``users`` and ``items`` are one
    column wide -- both are bare id lists in the published fixture, which is
    itself load-bearing for the link-prediction rows (S1).
    """
    fkeys = [
        {
            'name': 'USER_ID',
            'dtype': 'int',
            'stype': 'ID',
            'dst_table': 'users',
            'dst_key': 'USER_ID'
        },
        {
            'name': 'ITEM_ID',
            'dtype': 'int',
            'stype': 'ID',
            'dst_table': 'items',
            'dst_key': 'ITEM_ID'
        },
    ]
    if user_last:
        fkeys = list(reversed(fkeys))

    cols = [
        col('INTERACTION_ID', 'int', 'ID'),
        col('USER_ID', 'int', 'ID'),
        col('ITEM_ID', 'int', 'ID'),
        col('EVENT_TYPE', 'string', 'categorical'),
        col('IS_ORGANIC', 'int', 'categorical'),
        col('PLAYED_RATIO_PCT', 'float', 'numerical'),
        col('TRACK_LENGTH_SECONDS', 'float', 'numerical'),
    ]
    time_col = col('EVENT_TIME', 'date', 'timestamp') if with_time else None
    if time_col is not None:
        cols.insert(3, time_col)

    return {
        'tables': [
            {
                'table_name': 'users',
                'source_table_name': 'HYU.YAMBDA.USERS_SMOKE',
                'pkey': col('USER_ID', 'int', 'ID'),
                'time_col': None,
                'cols': [col('USER_ID', 'int', 'ID')],
                'fkeys': [],
            },
            {
                'table_name': 'items',
                'source_table_name': 'HYU.YAMBDA.ITEMS_SMOKE',
                'pkey': col('ITEM_ID', 'int', 'ID'),
                'time_col': None,
                'cols': [col('ITEM_ID', 'int', 'ID')],
                'fkeys': [],
            },
            {
                'table_name': 'interactions',
                'source_table_name': 'HYU.YAMBDA.INTERACTIONS_SMOKE',
                'pkey': col('INTERACTION_ID', 'int', 'ID'),
                'time_col': time_col,
                'cols': cols,
                'fkeys': fkeys,
            },
        ],
    }


def _filters(payload: Any) -> List[Any]:
    r"""Every ``filter`` node anywhere in a compiled plan."""
    found: List[Any] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == 'filter':
                found.append(value)
            else:
                found.extend(_filters(value))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_filters(value))
    return found


def describe(label: str, query: str, graph: Any, anchors: List[datetime],
             dump_plan: bool) -> None:
    r"""Parse, validate and compile one query; print what came back."""
    from kumo_core.pquery import parse_and_validate_query

    print(f'\n{"=" * 74}\n{label}\n{"=" * 74}')
    print(query)
    try:
        parsed = parse_and_validate_query(query, graph)
    except Exception as error:  # the interesting failure mode, so report it
        print(f'\nREJECTED  {type(error).__name__}: {error}')
        return

    print(f'\ntask_type    : {parsed.task_type}')
    print(f'task_family  : {parsed.task_family}')
    print(f'query_type   : {getattr(parsed, "query_type", None)}')
    print(f'entity_table : {parsed.entity_table}')
    print(f'rhs_table    : {getattr(parsed, "rhs_node_type", None)}')
    print(f'target_stype : {parsed.target_stype}')
    for field in ('is_temporal', 'num_classes', 'top_k', 'target_table',
                  'target_column', 'forecast_length'):
        if hasattr(parsed, field):
            print(f'{field:<13}: {getattr(parsed, field)}')
    target_ast = getattr(parsed.query, 'target_ast', None)
    for field in ('rhs_key', 'lhs_key'):
        if hasattr(target_ast, field):
            print(f'target_ast.{field:<7}: {getattr(target_ast, field)}')

    from kumo_core.pquery import _default_split
    try:
        split_config, split_family = _default_split(parsed, graph)
        print(f'split        : {split_family} {split_config}')
    except Exception as error:
        print(f'split        : FAILED {type(error).__name__}: {error}')

    import diskgraph
    from kumo_core.training.table import apply_reference_resolution

    entity_time = graph.table(parsed.entity_table).time_column
    print(f'entity time  : {entity_time}')

    is_lp = parsed.task_family == 'link_prediction'
    is_static_link = parsed.task_type == 'static_link_prediction'
    for anchor in anchors:
        nanos = int(anchor.timestamp()) * 1_000_000_000
        for is_prediction in (False, True):
            mode = 'prediction' if is_prediction else 'training'
            options: Dict[str,
                          Any] = dict(query=parsed.query, timestamps=[nanos],
                                      task_type=parsed.task_type,
                                      input_timestamp_unit='nanoseconds',
                                      is_prediction=is_prediction,
                                      target_stype=parsed.target_stype,
                                      entity_time_col=entity_time)
            if is_lp:
                options.update(list_distinct_target_is_array=True)
            if is_lp and is_prediction:
                options.update(lp_prediction_mode='LHS')
            if is_lp and not is_prediction:
                options.update(
                    lp_rhs_entity_table=getattr(parsed, 'rhs_node_type', None))
            try:
                plan = diskgraph.DiskGraphPlanCompiler().compile(**options)
                plan = apply_reference_resolution(
                    plan, entity_node_type=parsed.entity_table,
                    rhs_node_type=getattr(parsed, 'rhs_node_type', None),
                    resolve_link_targets=is_lp, is_static_link=is_static_link
                    and not is_prediction, rhs_is_temporal=False)
            except Exception as error:
                print(f'\n{anchor:%Y-%m-%d} {mode:<10} COMPILE FAILED '
                      f'{type(error).__name__}: {str(error)[:400]}')
                continue
            payload = json.loads(plan) if isinstance(plan, str) else plan
            print(f'\n{anchor:%Y-%m-%d} {mode:<10} '
                  f'filters={json.dumps(_filters(payload))[:500]}')
            if dump_plan and is_prediction:
                print(json.dumps(payload, indent=1)[:6000])


def report_default_plans() -> None:
    r"""The default model plan for every task type these five rows reach."""
    from kumo_core.model_plan import ModelPlan
    from kumoapi.model_plan import QueryType, RunMode
    from kumoapi.task import TaskType

    for name, query_type, has_time in (('STATIC_LINK_PREDICTION',
                                        QueryType.STATIC,
                                        False), ('TEMPORAL_LINK_PREDICTION',
                                                 QueryType.TEMPORAL, True),
                                       ('MULTILABEL_CLASSIFICATION',
                                        QueryType.TEMPORAL,
                                        True), ('MULTILABEL_RANKING',
                                                QueryType.TEMPORAL, True),
                                       ('REGRESSION', QueryType.STATIC,
                                        False)):
        task = getattr(TaskType, name, None)
        if task is None:
            print(f'\n{name}: no such TaskType')
            continue
        try:
            plan = ModelPlan.from_default(task_type=task,
                                          query_type=query_type,
                                          has_train_table_time_col=has_time,
                                          run_mode=RunMode.FAST)
        except Exception as error:
            print(f'\n{name}: from_default failed '
                  f'{type(error).__name__}: {error}')
            continue
        job = plan.training_job
        print(f'\n{name}')
        print(f'  tune_metric     {getattr(job, "tune_metric", None)}')
        print(f'  metrics         {getattr(job, "metrics", None)}')
        for section in ('training_table_generation', 'training_job',
                        'neighbor_sampling', 'optimization'):
            block = getattr(plan, section, None)
            if block is None:
                continue
            names = getattr(type(block), 'model_fields', None) or vars(block)
            for field in names:
                print(f'  {section}.{field} = {getattr(block, field)}')

    from kumo_core.resolved_task import default_metrics
    from kumoapi.task import TaskType as _TaskType
    for name in ('STATIC_LINK_PREDICTION', 'TEMPORAL_LINK_PREDICTION',
                 'MULTILABEL_CLASSIFICATION', 'MULTILABEL_RANKING',
                 'REGRESSION'):
        task = getattr(_TaskType, name, None)
        if task is not None:
            print(f'\ndefault_metrics({name}) = {default_metrics(task)}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-plan', action='store_true')
    parser.add_argument('--variants', action='store_true')
    parser.add_argument(
        '--fkey-order', choices=['user', 'item', 'both'], default='both',
        help='which interactions edge is declared last, and so '
        'which one _find_keys keeps')
    parser.add_argument(
        '--no-time-column', action='store_true',
        help='also compile against a graph with no EVENT_TIME '
        'on interactions')
    parser.add_argument('--only', default=None,
                        help='substring filter over query labels')
    args = parser.parse_args()

    from kumo_core.pquery import normalize_graph

    # One anchor inside the data and one past its end.
    anchors = [
        datetime(2022, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 17, tzinfo=timezone.utc)
    ]

    orders = (['user', 'item']
              if args.fkey_order == 'both' else [args.fkey_order])
    queries = dict(QUERIES)
    if args.variants:
        queries.update(VARIANTS)
    if args.only:
        queries = {k: v for k, v in queries.items() if args.only in k}

    time_variants = [True] + ([False] if args.no_time_column else [])
    for with_time in time_variants:
        for order in orders:
            graph = normalize_graph(
                portable_graph(user_last=order == 'user', with_time=with_time))
            print(f'\n\n{"#" * 74}\n# interactions fkey kept by _find_keys: '
                  f'{order.upper()}_ID   '
                  f'(interactions time column: '
                  f'{"EVENT_TIME" if with_time else "NONE"})\n{"#" * 74}')
            for label, query in queries.items():
                describe(label, query, graph, anchors, args.dump_plan)

    print(f'\n{"=" * 74}\ndefault model plans\n{"=" * 74}')
    report_default_plans()


if __name__ == '__main__':
    main()
