# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Why the ethereum prediction table comes back empty.

Training on this fixture succeeds and the task is static
(``is_temporal: false``), yet ``generate_prediction_table`` returns zero rows.
This dumps the compiled prediction plan and re-runs TT generation against the
already-ingested snapshot at several anchors, so the cause is read off the plan
rather than inferred from the failure.
"""
import argparse
import json
from datetime import datetime, timezone

import diskgraph
from kumo_core.pquery import normalize_graph, parse_and_validate_query
from kumo_core.training.table import apply_reference_resolution

QUERY = 'PREDICT addresses.IS_SCAM\nFOR EACH addresses.ADDRESS'

ANCHORS = {
    'max day in slice (2024-11-22)': datetime(2024, 11, 22,
                                              tzinfo=timezone.utc),
    'mid history (2020-01-01)': datetime(2020, 1, 1, tzinfo=timezone.utc),
    'before all data (2014-01-01)': datetime(2014, 1, 1, tzinfo=timezone.utc),
    'now (2026)': datetime(2026, 8, 15, tzinfo=timezone.utc),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-config', required=True,
                        help='published-model/manifests/graph.json')
    parser.add_argument('--index-path', required=True)
    args = parser.parse_args()

    graph_config = json.load(open(args.graph_config))
    native_graph = normalize_graph(graph_config)
    parsed = parse_and_validate_query(QUERY, native_graph)
    print('task_type  :', parsed.task_type)
    print('task_family:', parsed.task_family)
    print('entity     :', parsed.entity_table)
    print('entity time:', native_graph.table(parsed.entity_table).time_column)

    compiler = diskgraph.DiskGraphPlanCompiler()
    for label, moment in ANCHORS.items():
        nanos = int(moment.timestamp()) * 1_000_000_000
        plan_json = compiler.compile(
            query=parsed.query,
            timestamps=[nanos],
            task_type=parsed.task_type,
            input_timestamp_unit='nanoseconds',
            is_prediction=True,
            target_stype=parsed.target_stype,
            entity_time_col=native_graph.table(
                parsed.entity_table).time_column,
        )
        plan_json = apply_reference_resolution(
            plan_json, entity_node_type=parsed.entity_table,
            rhs_node_type=None, resolve_link_targets=False,
            is_static_link=False, rhs_is_temporal=False)
        if label == 'max day in slice (2024-11-22)':
            print('\n--- compiled prediction plan ---')
            print(
                json.dumps(
                    json.loads(plan_json) if isinstance(plan_json, str) else
                    plan_json, indent=1)[:4000])
        try:
            generated = diskgraph.execute_tt_gen(plan_json, args.index_path,
                                                 f'probe_{abs(hash(label))}')
            print(f'\n{label:<32} rows={generated.get("row_count")}')
        except Exception as error:
            print(f'\n{label:<32} ERROR {type(error).__name__}: '
                  f'{str(error)[:200]}')


if __name__ == '__main__':
    main()
