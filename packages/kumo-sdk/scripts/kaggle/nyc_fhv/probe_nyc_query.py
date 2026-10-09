# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What the three NYC FHV queries compile to, before any data exists.

Rows 9, 12 and 16 raise four questions that are far cheaper to answer from the
compiler than from a finished run, and one of them decides whether two of the
three rows are expressible at all:

1.  **Which of the two zone edges does ``FOR EACH zones.zone_id`` traverse?**
    The graph contract says to connect ``pickup_zone_id`` and
    ``dropoff_zone_id`` to ``zones.zone_id`` *separately*, and then to
    "explicitly select ``pickup_zone_id`` as the ``FOR EACH`` anchor" -- but the
    PQL grammar (``PQLGrammar.g4``) has **no** syntax for naming an edge. So
    either the choice is made somewhere else or it is not made at all. This
    probe builds the same graph with the two foreign keys declared in both
    orders and compiles the same query against each.

2.  Is row 9's ``PREDICT trips.trip_seconds FOR EACH trips.trip_id`` a *static*
    query -- entity and target on the same table, which is rows 1-4's shape --
    even though ``trips`` carries a time column? That decides whether it needs
    a ``--holdout-pct`` (row 1 S1) and it is visible in the compiled prediction
    plan's filter nodes.

3.  Does row 16's target survive the loop check? ``trips.dropoff_zone_id`` is a
    foreign key **into the entity table itself**, and
    ``predictive_query_validator.py:429`` rejects a ``LIST_DISTINCT`` target
    whose join key is the target column. NOAA S2 hit that message from the
    static side; here the query is temporal, so whether it fires depends
    entirely on question 1.

4.  Where does ``forecast_length`` come from, and does ``FORECAST n
    TIMEFRAMES`` before ``FOR EACH`` compile to ``forecasting``? NOAA S3
    established the clause and its position; this re-checks it on a
    ``COUNT(trips.*)`` target, which is a different aggregation.

None of it needs the fixture. ``kumo_core.pquery.normalize_graph`` takes a fully
resolved *portable graph* -- a plain mapping of table and column descriptions --
so the graph is synthesized here from the schema ``probe_nyc_fixture.py`` read
off the stage. The plans printed are the real ones: ``run_nyc_local.py``
compiles the same queries through the same functions on the way to training.
"""
import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

QUERIES: Dict[str, str] = {
    'row 9 regression (as published)': ('PREDICT trips.TRIP_SECONDS\n'
                                        'FOR EACH trips.TRIP_ID'),
    'row 12 forecasting (as published)':
    ('PREDICT COUNT(trips.*, 0, 1, days)\n'
     'FOR EACH zones.ZONE_ID'),
    'row 12 forecasting (+ FORECAST 7 TIMEFRAMES, what the matrix means)':
    ('PREDICT COUNT(trips.*, 0, 1, days)\n'
     'FORECAST 7 TIMEFRAMES\n'
     'FOR EACH zones.ZONE_ID'),
    'row 16 temporal link prediction (as published)':
    ('PREDICT LIST_DISTINCT(trips.DROPOFF_ZONE_ID, 0, 7, days)\n'
     'RANK TOP 20\n'
     'FOR EACH zones.ZONE_ID'),
}

VARIANTS: Dict[str, str] = {
    'row 12 with FORECAST 1 TIMEFRAMES':
    ('PREDICT COUNT(trips.*, 0, 1, days)\n'
     'FORECAST 1 TIMEFRAMES\n'
     'FOR EACH zones.ZONE_ID'),
    'row 12 with FORECAST 2 TIMEFRAMES':
    ('PREDICT COUNT(trips.*, 0, 1, days)\n'
     'FORECAST 2 TIMEFRAMES\n'
     'FOR EACH zones.ZONE_ID'),
    'row 16 with a 1-day window':
    ('PREDICT LIST_DISTINCT(trips.DROPOFF_ZONE_ID, 0, 1, days)\n'
     'RANK TOP 20\n'
     'FOR EACH zones.ZONE_ID'),
    'row 16 targeting the pickup key instead':
    ('PREDICT LIST_DISTINCT(trips.PICKUP_ZONE_ID, 0, 7, days)\n'
     'RANK TOP 20\n'
     'FOR EACH zones.ZONE_ID'),
    'row 9 without a time column on trips is a different question; see '
    '--fkey-order': ('PREDICT trips.TRIP_SECONDS\n'
                     'FOR EACH trips.TRIP_ID'),
}


def col(name: str, dtype: str, stype: str) -> Dict[str, str]:
    return {'name': name, 'dtype': dtype, 'stype': stype}


def portable_graph(pickup_first: bool = True) -> Dict[str, Any]:
    r"""The three-table NYC FHV graph as a resolved portable-graph mapping.

    ``pickup_first`` controls the **order** the two zone foreign keys are
    declared in, and nothing else. That is not cosmetic: ``to_api_graph``
    (``kumo_core/pquery.py:370``) emits one ``ColumnKeyGroup`` per foreign key,
    so ``zones.ZONE_ID`` appears in two groups, and ``JoinValidator._find_keys``
    (``kumopql/validator/join_validator.py:149``) **assigns** rather than
    accumulates as it walks them::

        if is_keys_group:
            keys = curr_keys      # <- overwrite, not extend

    so the last matching group wins. Declaration order is therefore the only
    lever there is over which edge ``FOR EACH zones.ZONE_ID`` traverses. This
    function exists to make that testable rather than argued.

    Kept deliberately narrow: only the columns the three queries need plus the
    ones the leakage question in ``probe_nyc_target.py`` turns on. The run
    script builds the real, wide graph.
    """
    zone_fkeys = [
        {
            'name': 'PICKUP_ZONE_ID',
            'dtype': 'int',
            'stype': 'ID',
            'dst_table': 'zones',
            'dst_key': 'ZONE_ID'
        },
        {
            'name': 'DROPOFF_ZONE_ID',
            'dtype': 'int',
            'stype': 'ID',
            'dst_table': 'zones',
            'dst_key': 'ZONE_ID'
        },
    ]
    if pickup_first:
        # `_find_keys` keeps the LAST matching group, so "pickup is the anchor"
        # means pickup is declared LAST.
        zone_fkeys = list(reversed(zone_fkeys))

    return {
        'tables': [
            {
                'table_name': 'zones',
                'source_table_name': 'HYU.NYC_FHV.ZONES_RAW',
                'pkey': col('ZONE_ID', 'int', 'ID'),
                'time_col': None,
                'cols': [col('ZONE_ID', 'int', 'ID')],
                'fkeys': [],
            },
            {
                'table_name': 'providers',
                'source_table_name': 'HYU.NYC_FHV.PROVIDERS_RAW',
                'pkey': col('PROVIDER_ID', 'string', 'ID'),
                'time_col': None,
                'cols': [col('PROVIDER_ID', 'string', 'ID')],
                'fkeys': [],
            },
            {
                'table_name':
                'trips',
                'source_table_name':
                'HYU.NYC_FHV.TRIPS_R9TINY',
                'pkey':
                col('TRIP_ID', 'string', 'ID'),
                'time_col':
                col('PICKUP_DATETIME', 'date', 'timestamp'),
                'cols': [
                    col('TRIP_ID', 'string', 'ID'),
                    col('PROVIDER_ID', 'string', 'ID'),
                    col('PICKUP_DATETIME', 'date', 'timestamp'),
                    col('PICKUP_ZONE_ID', 'int', 'ID'),
                    col('DROPOFF_ZONE_ID', 'int', 'ID'),
                    col('TRIP_MILES', 'float', 'numerical'),
                    col('TRIP_SECONDS', 'int', 'numerical'),
                    col('BASE_PASSENGER_FARE', 'float', 'numerical'),
                ],
                'fkeys': [
                    {
                        'name': 'PROVIDER_ID',
                        'dtype': 'string',
                        'stype': 'ID',
                        'dst_table': 'providers',
                        'dst_key': 'PROVIDER_ID'
                    },
                    *zone_fkeys,
                ],
            },
        ],
    }


def describe(label: str, query: str, graph: Any, anchors: List[datetime],
             dump_plan: bool) -> None:
    r"""Parse, validate and compile one query; print what came back.

    The parse settles the task type and, for row 16, whether the loop check
    fires. The compile settles the prediction-mode filter -- the node that made
    rows 1-4 return nothing without a holdout, and which NOAA S2 measured to be
    absent for temporal aggregate targets.
    """
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
    print(f'entity_table : {parsed.entity_table}')
    print(f'target_stype : {parsed.target_stype}')
    for field in ('is_temporal', 'num_classes', 'top_k', 'target_table',
                  'target_column', 'forecast_length'):
        if hasattr(parsed, field):
            print(f'{field:<13}: {getattr(parsed, field)}')
    # The join key is the whole question for rows 12 and 16: it is the edge the
    # traversal from `zones` actually took.
    target_ast = getattr(parsed.query, 'target_ast', None)
    for field in ('rhs_key', 'lhs_key'):
        if hasattr(target_ast, field):
            print(f'target_ast.{field:<3}: {getattr(target_ast, field)}')

    import diskgraph
    from kumo_core.training.table import apply_reference_resolution

    entity_time = graph.table(parsed.entity_table).time_column
    print(f'entity time  : {entity_time}')

    is_lp = parsed.task_family == 'link_prediction'
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
            if is_lp and is_prediction:
                # What `kumo_core/prediction.py:58` adds for link prediction,
                # and the reason NOAA S9 found forecasting broken: that special
                # case has no forecasting sibling.
                options.update(lp_prediction_mode='LHS',
                               list_distinct_target_is_array=True)
            try:
                plan = diskgraph.DiskGraphPlanCompiler().compile(**options)
                plan = apply_reference_resolution(
                    plan, entity_node_type=parsed.entity_table,
                    rhs_node_type=None, resolve_link_targets=False,
                    is_static_link=False, rhs_is_temporal=False)
            except Exception as error:
                print(f'\n{anchor:%Y-%m-%d} {mode:<10} COMPILE FAILED '
                      f'{type(error).__name__}: {str(error)[:300]}')
                continue
            payload = json.loads(plan) if isinstance(plan, str) else plan
            print(f'\n{anchor:%Y-%m-%d} {mode:<10} '
                  f'filters={json.dumps(_filters(payload))[:400]}')
            if dump_plan and is_prediction:
                print(json.dumps(payload, indent=1)[:6000])


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


def report_default_plans() -> None:
    r"""The default model plan for each task type these three rows reach.

    The model-plan boundary was fixed in the working tree during the NOAA block
    (NOAA S7), so overrides now arrive -- but the default is still what an
    unconfigured run gets, and for ``FORECASTING`` it is where ``forecast_length``
    would live if the matrix were right about it.
    """
    from kumo_core.model_plan import ModelPlan
    from kumoapi.model_plan import QueryType, RunMode
    from kumoapi.task import TaskType

    for name in ('REGRESSION', 'FORECASTING', 'LINK_PREDICTION',
                 'TEMPORAL_LINK_PREDICTION', 'MULTILABEL_RANKING'):
        task = getattr(TaskType, name, None)
        if task is None:
            print(f'\n{name}: no such TaskType')
            continue
        try:
            plan = ModelPlan.from_default(task_type=task,
                                          query_type=QueryType.TEMPORAL,
                                          has_train_table_time_col=True,
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
                        'column_processing', 'neighbor_sampling',
                        'optimization', 'model_architecture'):
            block = getattr(plan, section, None)
            if block is None:
                continue
            names = getattr(type(block), 'model_fields', None) or vars(block)
            for field in names:
                if 'forecast' in field or 'horizon' in field:
                    print(f'  {section}.{field} = {getattr(block, field)}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-plan', action='store_true')
    parser.add_argument('--variants', action='store_true')
    parser.add_argument(
        '--fkey-order', choices=['pickup', 'dropoff', 'both'], default='both',
        help='which zone edge is declared last, and so which '
        'one _find_keys keeps')
    args = parser.parse_args()

    from kumo_core.pquery import normalize_graph

    # One anchor inside the data and one past its end. Row 1 S1 needed four
    # anchors to establish that an empty prediction table was not an anchor
    # problem, and the same doubt attaches to any empty result here.
    anchors = [
        datetime(2022, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 17, tzinfo=timezone.utc)
    ]

    orders = (['pickup', 'dropoff']
              if args.fkey_order == 'both' else [args.fkey_order])
    queries = dict(QUERIES)
    if args.variants:
        queries.update(VARIANTS)

    for order in orders:
        graph = normalize_graph(portable_graph(pickup_first=order == 'pickup'))
        print(f'\n\n{"#" * 74}\n# zone edge kept by _find_keys: '
              f'{order.upper()}_ZONE_ID\n{"#" * 74}')
        for label, query in queries.items():
            describe(label, query, graph, anchors, args.dump_plan)

    print(f'\n{"=" * 74}\ndefault model plans\n{"=" * 74}')
    report_default_plans()


if __name__ == '__main__':
    main()
