# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score with the champion model, or the one ``--model-path`` names:
``python score_job.py --artifact-root ROOT --source SOURCE [...]``; see
README.md."""

import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(
    globals().get('__file__') or sys.argv[0])))

import jobs  # noqa: E402

import kumo.engine as kumo  # noqa: E402
from kumo.engine.connector import connector_for_root  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> kumo.PredictionResult:
    args = jobs.parse_args(argv, score=True)
    connector = jobs.source(args)
    graph = jobs.build_graph(connector, args.table_prefix)
    with kumo.mlops(jobs.mlops(args, connector)):
        trainer = jobs.load_model(args, connector)
        snapshot = (kumo.GraphSnapshot.load_from_connector(
            connector_for_root(args.artifact_root), path=args.snapshot,
            graph=graph) if args.snapshot else graph.snapshot())
        defaults = trainer.result.scoring_defaults
        result = trainer.predict(
            snapshot, output_config=jobs.prediction_output(args, connector),
            binary_classification_threshold=None
            if defaults.binary_classification_threshold is not None else 0.5)
    print(f'predictions: {result.num_rows} rows')
    return result


if __name__ == '__main__':
    main()
