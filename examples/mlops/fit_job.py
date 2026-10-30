# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train the users-churn model: ``python fit_job.py --artifact-root ROOT
--source SOURCE [...]``; see README.md."""

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(
    globals().get('__file__') or sys.argv[0])))

import jobs  # noqa: E402

import kumo.engine as kumo  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> kumo.TrainingResult:
    args = jobs.parse_args(argv, score=False)
    connector = jobs.source(args)
    graph = jobs.build_graph(connector, args.table_prefix)
    with kumo.mlops(jobs.mlops(args, connector)):
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=args.query)
        table = query.generate_training_table(snapshot)
        plan = query.suggest_model_plan(snapshot)
        if args.num_experiments:
            plan.training_job.num_experiments = args.num_experiments
        result = kumo.Trainer(plan).fit(
            table, snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(
                binary_classification_threshold=args.threshold))
    print(f'model: models/{result.id}')
    if args.model_path_file:
        Path(args.model_path_file).write_text(f'models/{result.id}')
    return result


if __name__ == '__main__':
    main()
