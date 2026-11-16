# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What the three NOAA queries compile to, before any data exists.

Rows 5, 8 and 11 introduce three task families no previous row exercised --
``multilabel_classification``, ``multilabel_ranking`` and ``forecasting`` -- and
each raises a question that is cheaper to answer from the compiler than from a
finished run:

1.  Is ``LIST_DISTINCT(observations.element)`` legal with **no time window**?
    Every previous multilabel example in the matrix carries one
    (``, 0, 30, days`` for yambda); rows 5 and 8 do not.
2.  What does the prediction-mode filter look like for an **aggregate** target?
    Rows 1-4 all had static *column* targets, which compile to
    ``WHERE <target> IS NULL`` and select nothing on a fully-labeled fixture
    (row 1 S1). An aggregate target has no column to blank, so either the
    filter is different or there is none.
3.  Does ``RANK TOP 25`` differ from ``CLASSIFY TOP 25`` anywhere other than the
    task type -- in the plan, the target, or the metric set?
4.  Where does ``forecast_length`` live, and what does the default plan for
    ``forecasting`` actually resolve to?

None of that needs the fixture. ``kumo_core.pquery.normalize_graph`` takes a
fully resolved *portable graph* -- a plain mapping of table and column
descriptions -- so the graph is synthesized here from the schema
``probe_noaa_fixture.py`` read off the stage, and the queries are parsed and
compiled against it. Running this costs a second and needs no Snowflake session,
which is the point: it front-runs a build that takes the better part of an hour.

The plans it prints are the real ones. ``run_noaa_local.py`` compiles the same
queries through the same functions on the way to training.
"""
import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

QUERIES: Dict[str, str] = {
    'row 5 multilabel classification':
    ('PREDICT LIST_DISTINCT(observations.ELEMENT)\n'
     'CLASSIFY TOP 25\n'
     'FOR EACH stations.STATION_ID'),
    'row 8 multilabel ranking':
    ('PREDICT LIST_DISTINCT(observations.ELEMENT)\n'
     'RANK TOP 25\n'
     'FOR EACH stations.STATION_ID'),
    'row 11 forecasting':
    ("PREDICT AVG(observations.VALUE WHERE observations.ELEMENT = 'TMAX', "
     '0, 7, days)\n'
     'FOR EACH stations.STATION_ID'),
}

#: Variants worth compiling beside the published three, because each isolates
#: one thing the published query leaves ambiguous.
VARIANTS: Dict[str, str] = {
    'row 5 with a time window (what yambda rows 6/7 look like)':
    ('PREDICT LIST_DISTINCT(observations.ELEMENT, 0, 30, days)\n'
     'CLASSIFY TOP 25\n'
     'FOR EACH stations.STATION_ID'),
    'row 11 without the element filter':
    ('PREDICT AVG(observations.VALUE, 0, 7, days)\n'
     'FOR EACH stations.STATION_ID'),
}


def portable_graph() -> Dict[str, Any]:
    r"""The two-table NOAA graph as a fully resolved portable-graph mapping.

    Written by hand rather than exported from a run so that this probe has no
    dependency on a build. The column names, dtypes and stypes are the ones
    ``run_noaa_local.py`` produces: ``STATION_ID`` is an ID on both sides (it is
    the primary key of ``stations`` and a foreign key on ``observations``),
    ``ELEMENT`` is categorical because that is what makes the multilabel target
    a classification target at all, and ``EVENT_TIME`` is the fact table's time
    column.

    ``stations`` deliberately has **no** ``time_col``: the published table is
    one column wide and carries nothing else. That is the interesting half of
    rows 5 and 8 -- a static entity table with a temporal fact table.
    """
    def col(name: str, dtype: str, stype: str) -> Dict[str, str]:
        return {'name': name, 'dtype': dtype, 'stype': stype}

    return {
        'tables': [
            {
                'table_name': 'stations',
                'source_table_name': 'EXAMPLE_DB.NOAA.STATIONS_SMOKE',
                'pkey': col('STATION_ID', 'string', 'ID'),
                'time_col': None,
                'cols': [col('STATION_ID', 'string', 'ID')],
                'fkeys': [],
            },
            {
                'table_name':
                'observations',
                'source_table_name':
                'EXAMPLE_DB.NOAA.OBSERVATIONS_SMOKE',
                'pkey':
                None,
                'time_col':
                col('EVENT_TIME', 'date', 'timestamp'),
                'cols': [
                    col('STATION_ID', 'string', 'ID'),
                    col('EVENT_TIME', 'date', 'timestamp'),
                    col('ELEMENT', 'string', 'categorical'),
                    col('VALUE', 'int', 'numerical'),
                ],
                'fkeys': [{
                    'name': 'STATION_ID',
                    'dtype': 'string',
                    'stype': 'ID',
                    'dst_table': 'stations',
                    'dst_key': 'STATION_ID',
                }],
            },
        ],
    }


def timeless_graph() -> Dict[str, Any]:
    r"""The same graph with ``observations`` stripped of its time column.

    This exists to make one thing *measurable* that would otherwise only be
    readable in the compiler source. Rows 1-4 established that a static
    **column** target compiles its prediction table to ``WHERE target IS NULL``
    and therefore selects nothing on a fully-labeled fixture. The question this
    block raises is what a static **aggregate** target does instead, and the
    answer matters for the whole matrix, not just for NOAA.

    It cannot be asked of the real graph: with a time column on the fact table,
    every static aggregation is rejected before compilation
    (``Time range required for LIST_DISTINCT with time column``). Removing the
    time column removes *that* objection, so what is left is whatever else
    stands in the way.

    **The answer turns out to be that nothing gets through.** Measured, both
    surviving candidates are still rejected, for two different reasons:

    * ``AVG(...)`` -- *"static usage aggregations is only supported for
      LIST_DISTINCT combined with RANK"*;
    * ``LIST_DISTINCT(observations.STATION_ID) RANK`` -- the only shape the
      first message admits -- *"Target observations.STATION_ID creates a loop by
      pointing to the entity table"*.

    ``observations`` has exactly one foreign key and it points at ``stations``,
    which is the entity. So on a two-table dimension/fact graph there is **no
    expressible static aggregate target at all**, and the compiler's
    ``has_edges == False`` prediction filter for static aggregates
    (``diskgraph/compiler.py``) is unreachable from here. That is worth stating
    as a measured rejection rather than left as a code reading, which is what
    this function is for.
    """
    graph = portable_graph()
    for table in graph['tables']:
        if table['table_name'] == 'observations':
            table['time_col'] = None
            table['cols'] = [
                c for c in table['cols'] if c['name'] != 'EVENT_TIME'
            ]
    return graph


#: Queries compiled against ``timeless_graph()`` rather than the real one.
TIMELESS: Dict[str, str] = {
    'static RANK over a foreign key (the one legal static LIST_DISTINCT)':
    ('PREDICT LIST_DISTINCT(observations.STATION_ID)\n'
     'RANK TOP 25\n'
     'FOR EACH stations.STATION_ID'),
    'static aggregate target (row 11 with no window)':
    ("PREDICT AVG(observations.VALUE WHERE observations.ELEMENT = 'TMAX')\n"
     'FOR EACH stations.STATION_ID'),
}


def describe(label: str, query: str, graph: Any, anchors: List[datetime],
             dump_plan: bool) -> None:
    r"""Parse, validate and compile one query; print what came back.

    The parse is the cheap half and settles the task type. The compile is the
    half that answers the prediction-table question, because the filter that
    selected nothing on rows 1-4 is a node of the compiled plan rather than
    anything the parser reports.
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
                  'target_column'):
        if hasattr(parsed, field):
            print(f'{field:<13}: {getattr(parsed, field)}')

    import diskgraph
    from kumo_core.training.table import apply_reference_resolution

    entity_time = graph.table(parsed.entity_table).time_column
    print(f'entity time  : {entity_time}')

    for anchor in anchors:
        nanos = int(anchor.timestamp()) * 1_000_000_000
        for is_prediction in (False, True):
            mode = 'prediction' if is_prediction else 'training'
            try:
                plan = diskgraph.DiskGraphPlanCompiler().compile(
                    query=parsed.query,
                    timestamps=[nanos],
                    task_type=parsed.task_type,
                    input_timestamp_unit='nanoseconds',
                    is_prediction=is_prediction,
                    target_stype=parsed.target_stype,
                    entity_time_col=entity_time,
                )
                plan = apply_reference_resolution(
                    plan, entity_node_type=parsed.entity_table,
                    rhs_node_type=None, resolve_link_targets=False,
                    is_static_link=False, rhs_is_temporal=False)
            except Exception as error:
                print(f'\n{anchor:%Y-%m-%d} {mode:<10} COMPILE FAILED '
                      f'{type(error).__name__}: {str(error)[:300]}')
                continue
            payload = json.loads(plan) if isinstance(plan, str) else plan
            # The filter is the whole point: on rows 1-4 it was
            # `{"compare": {"op": "is_null", ...}}` and it is why batch
            # prediction returned nothing.
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
    r"""The default model plan for each of the three task types.

    Row 3 S7 and row 4 S7 measured user overrides being silently dropped, so the
    default is what a run actually gets. For rows 5 and 8 that decides which
    metrics exist at all; for row 11 it decides ``forecast_length``, which the
    matrix says should be 4 and which is model-plan configuration rather than
    PQL.
    """
    from kumo_core.model_plan import ModelPlan
    from kumoapi.model_plan import QueryType, RunMode
    from kumoapi.task import TaskType

    for name in ('MULTILABEL_CLASSIFICATION', 'MULTILABEL_RANKING',
                 'FORECASTING', 'MULTICLASS_CLASSIFICATION'):
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
        # Anything horizon-shaped, wherever it lives. The matrix says row 11
        # needs ``forecast_length=4`` and calls it model-plan configuration
        # without saying which section owns it, so it is searched for.
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
    parser.add_argument(
        '--dump-plan', action='store_true',
        help='print the whole compiled prediction plan, not '
        'just its filter nodes')
    parser.add_argument('--variants', action='store_true',
                        help='also compile the contrast queries')
    args = parser.parse_args()

    from kumo_core.pquery import normalize_graph
    graph = normalize_graph(portable_graph())

    # Two anchors: one inside the data and one past its end. Row 1 S1 needed
    # four anchors to establish that an empty prediction table was not an
    # anchor problem, and the same doubt will attach to any empty result here.
    anchors = [
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 16, tzinfo=timezone.utc)
    ]

    queries = dict(QUERIES)
    if args.variants:
        queries.update(VARIANTS)
    for label, query in queries.items():
        describe(label, query, graph, anchors, args.dump_plan)

    if args.variants:
        # Compiled against the time-less graph, where a static aggregate is
        # legal, purely so the prediction-mode filter can be *seen*. Look for
        # `has_edges` in the filter list: the compiler keeps only entities that
        # traversed **zero** edges into the target table, which on a fixture
        # where every station has observations selects nothing at all -- a
        # different failure from rows 1-4's `is_null`, with the same outcome.
        timeless = normalize_graph(timeless_graph())
        for label, query in TIMELESS.items():
            describe(f'[time-less graph] {label}', query, timeless, anchors,
                     args.dump_plan)

    print(f'\n{"=" * 74}\ndefault model plans\n{"=" * 74}')
    report_default_plans()


if __name__ == '__main__':
    main()
