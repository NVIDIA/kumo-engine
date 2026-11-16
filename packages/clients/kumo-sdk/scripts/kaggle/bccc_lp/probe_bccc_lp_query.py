# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What matrix row 14 compiles to, before any data exists.

Row 14 is ``PREDICT LIST_DISTINCT(network_flows.dst_ip) / RANK TOP 100 /
FOR EACH hosts.host_id`` -- **static** link prediction, with no time window on
the target. Three questions decide whether it is expressible at all, and all
three are answerable from the compiler in a second:

1.  **Is a static ``LIST_DISTINCT`` legal here?** NOAA rows 5 and 8 were
    rejected by ``kumopql/validator/time_range_validator.py``, which enumerates
    four cases and allows exactly one: ``RANK`` over a **foreign key**. NOAA's
    target was a categorical column and landed in a rejected cell.
    ``network_flows.dst_ip`` *is* a foreign key, so row 14 should land in the
    legal one -- verified here rather than assumed.

2.  **Which of the two ``hosts`` edges does ``FOR EACH hosts.host_id``
    traverse?** ``src_ip`` and ``dst_ip`` both point at ``hosts.host_id``, and
    the matrix says to "explicitly select the keyed ``src_ip`` edge as the
    ``FOR EACH`` anchor". There is no PQL syntax for that (``PQLGrammar.g4`` has
    no ``VIA``/``USING`` rule). ``kumo_core.pquery.to_api_graph`` emits one
    ``ColumnKeyGroup`` per foreign key and ``kumopql/validator/
    join_validator.py::_find_keys`` **assigns** rather than accumulates
    (``keys = curr_keys``), so the *last declared* foreign key wins silently.
    This probe builds the same graph both ways round and compiles the same
    query against each. NYC row 16 established the rule on a temporal LP
    target; row 14 is the static form of the same shape, so whether the rule
    transfers is a measurement.

3.  **Does the time column on ``network_flows`` block the static query?**
    ``static_query_validations`` emits a *warning* -- "Entity table hosts does
    not have a Create date, but several other columns in the query do" -- and
    NOAA S2 read that as a possible rejection. Compiling the query against a
    graph with and without ``EVENT_TIME`` settles whether it is fatal.

Nothing here needs the fixture. ``kumo_core.pquery.normalize_graph`` takes a
fully resolved *portable graph*, so the graph is synthesized from the schema
``../bccc/probe_bccc_fixture.py`` read off the stage. The plans printed are the
real ones: ``run_bccc_lp_local.py`` compiles the same query through the same
functions on the way to training.
"""
import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

#: The published row 14 query, plus the neighbouring shapes that say *why* it
#: compiles the way it does.
QUERIES: Dict[str, str] = {
    'row 14 static link prediction (as published)':
    ('PREDICT LIST_DISTINCT(network_flows.DST_IP)\n'
     'RANK TOP 100\n'
     'FOR EACH hosts.HOST_ID'),
}

VARIANTS: Dict[str, str] = {
    'row 14 with CLASSIFY instead of RANK (expected: rejected)':
    ('PREDICT LIST_DISTINCT(network_flows.DST_IP)\n'
     'CLASSIFY TOP 100\n'
     'FOR EACH hosts.HOST_ID'),
    'row 14 targeting SRC_IP instead of DST_IP':
    ('PREDICT LIST_DISTINCT(network_flows.SRC_IP)\n'
     'RANK TOP 100\n'
     'FOR EACH hosts.HOST_ID'),
    'row 14 made temporal with a 1-day window':
    ('PREDICT LIST_DISTINCT(network_flows.DST_IP, 0, 1, days)\n'
     'RANK TOP 100\n'
     'FOR EACH hosts.HOST_ID'),
    'row 14 made temporal with a 7-day window':
    ('PREDICT LIST_DISTINCT(network_flows.DST_IP, 0, 7, days)\n'
     'RANK TOP 100\n'
     'FOR EACH hosts.HOST_ID'),
    'row 14 anchored on the flow table instead':
    ('PREDICT LIST_DISTINCT(network_flows.DST_IP)\n'
     'RANK TOP 100\n'
     'FOR EACH network_flows.FLOW_ID'),
}


def col(name: str, dtype: str, stype: str) -> Dict[str, str]:
    return {'name': name, 'dtype': dtype, 'stype': stype}


def portable_graph(src_last: bool = True,
                   with_time: bool = True) -> Dict[str, Any]:
    r"""The two-table BCCC graph as a resolved portable-graph mapping.

    ``src_last`` controls the **order** the two host foreign keys are declared
    in, and nothing else; ``with_time`` drops ``EVENT_TIME`` from
    ``network_flows`` so the third question above can be answered.

    Deliberately narrow: the key, the two foreign keys, the time column and two
    representative feature columns. The run script builds the real, wide graph.
    """
    host_fkeys = [
        {
            'name': 'SRC_IP',
            'dtype': 'string',
            'stype': 'ID',
            'dst_table': 'hosts',
            'dst_key': 'HOST_ID'
        },
        {
            'name': 'DST_IP',
            'dtype': 'string',
            'stype': 'ID',
            'dst_table': 'hosts',
            'dst_key': 'HOST_ID'
        },
    ]
    if src_last:
        # `_find_keys` keeps the LAST matching group, so "src_ip is the anchor"
        # means src_ip is declared LAST.
        host_fkeys = list(reversed(host_fkeys))

    cols = [
        col('FLOW_ID', 'string', 'ID'),
        col('SRC_IP', 'string', 'ID'),
        col('DST_IP', 'string', 'ID'),
        col('DURATION', 'float', 'numerical'),
        col('PACKETS_CNT', 'int', 'numerical'),
    ]
    time_col = col('EVENT_TIME', 'date', 'timestamp') if with_time else None
    if time_col is not None:
        cols.insert(3, time_col)

    return {
        'tables': [
            {
                'table_name': 'hosts',
                'source_table_name': 'EXAMPLE_DB.BCCC.HOSTS_LP_SMOKE',
                'pkey': col('HOST_ID', 'string', 'ID'),
                'time_col': None,
                'cols': [col('HOST_ID', 'string', 'ID')],
                'fkeys': [],
            },
            {
                'table_name': 'network_flows',
                'source_table_name': 'EXAMPLE_DB.BCCC.NETWORK_FLOWS_LP_SMOKE',
                'pkey': col('FLOW_ID', 'string', 'ID'),
                'time_col': time_col,
                'cols': cols,
                'fkeys': host_fkeys,
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
    r"""Parse, validate and compile one query; print what came back.

    The parse settles the task type, the resolved anchor edge and whether the
    loop check fires. The compile settles the prediction-mode filter -- the node
    that made rows 1-4 return nothing without a holdout, and which NOAA S2
    measured to be *present but inverted* for static aggregate targets
    (``has_edges = False``, i.e. only entities with no related rows). Row 14 is
    the first static aggregate target in the series that actually compiles, so
    that branch is finally reachable and this is where it shows up.
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
    print(f'query_type   : {getattr(parsed, "query_type", None)}')
    print(f'entity_table : {parsed.entity_table}')
    print(f'rhs_table    : {getattr(parsed, "rhs_node_type", None)}')
    print(f'target_stype : {parsed.target_stype}')
    for field in ('is_temporal', 'num_classes', 'top_k', 'target_table',
                  'target_column', 'forecast_length'):
        if hasattr(parsed, field):
            print(f'{field:<13}: {getattr(parsed, field)}')
    # The join key is the whole question: it is the edge the traversal from
    # `hosts` actually took.
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
                # What `kumo_core/prediction.py:58` adds for link prediction.
                options.update(lp_prediction_mode='LHS')
            if is_lp and not is_prediction:
                options.update(
                    lp_rhs_entity_table=getattr(parsed, 'rhs_node_type', None))
            try:
                plan = diskgraph.DiskGraphPlanCompiler().compile(**options)
                plan = apply_reference_resolution(
                    plan,
                    entity_node_type=parsed.entity_table,
                    rhs_node_type=getattr(parsed, 'rhs_node_type', None),
                    resolve_link_targets=is_lp,
                    # Training passes `task_type == 'static_link_prediction'`
                    # here; batch prediction hardcodes `False`
                    # (`kumo_core/prediction.py:69`). Both are reproduced.
                    is_static_link=is_static_link and not is_prediction,
                    rhs_is_temporal=False)
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
    r"""The default model plan for the two link-prediction task types."""
    from kumo_core.model_plan import ModelPlan
    from kumoapi.model_plan import QueryType, RunMode
    from kumoapi.task import TaskType

    for name, query_type, has_time in (('STATIC_LINK_PREDICTION',
                                        QueryType.STATIC, False),
                                       ('TEMPORAL_LINK_PREDICTION',
                                        QueryType.TEMPORAL, True)):
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
    for name in ('STATIC_LINK_PREDICTION', 'TEMPORAL_LINK_PREDICTION'):
        task = getattr(_TaskType, name, None)
        if task is not None:
            print(f'\ndefault_metrics({name}) = {default_metrics(task)}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-plan', action='store_true')
    parser.add_argument('--variants', action='store_true')
    parser.add_argument(
        '--fkey-order', choices=['src', 'dst', 'both'], default='both',
        help='which host edge is declared last, and so which '
        'one _find_keys keeps')
    parser.add_argument(
        '--no-time-column', action='store_true',
        help='also compile against a graph with no EVENT_TIME '
        'on network_flows')
    args = parser.parse_args()

    from kumo_core.pquery import normalize_graph

    # One anchor inside the data and one past its end. Row 1 S1 needed four
    # anchors to establish that an empty prediction table was not an anchor
    # problem, and the same doubt attaches to any empty result here.
    anchors = [
        datetime(2018, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 17, tzinfo=timezone.utc)
    ]

    orders = (['src', 'dst']
              if args.fkey_order == 'both' else [args.fkey_order])
    queries = dict(QUERIES)
    if args.variants:
        queries.update(VARIANTS)

    time_variants = [True] + ([False] if args.no_time_column else [])
    for with_time in time_variants:
        for order in orders:
            graph = normalize_graph(
                portable_graph(src_last=order == 'src', with_time=with_time))
            print(f'\n\n{"#" * 74}\n# host edge kept by _find_keys: '
                  f'{order.upper()}_IP   '
                  f'(network_flows time column: '
                  f'{"EVENT_TIME" if with_time else "NONE"})\n{"#" * 74}')
            for label, query in queries.items():
                describe(label, query, graph, anchors, args.dump_plan)

    print(f'\n{"=" * 74}\ndefault model plans\n{"=" * 74}')
    report_default_plans()


if __name__ == '__main__':
    main()
