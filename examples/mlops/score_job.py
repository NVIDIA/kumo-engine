# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score with the champion or ``--model-path`` model; see README.md."""

import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(
    globals().get('__file__') or sys.argv[0])))

import jobs  # noqa: E402

import kumo.engine as kumo  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> kumo.PredictionResult:
    args = jobs.parse_args(argv, score=True)
    connector = jobs.source(args)
    graph = jobs.build_graph(connector, args.table_prefix)
    cfg = jobs.mlops(args, connector)
    with kumo.mlops(cfg):
        trainer = jobs.load_model(args, cfg)
        snapshot = (kumo.GraphSnapshot.load_from_connector(
            cfg, id=args.snapshot, graph=graph)
                    if args.snapshot else graph.snapshot())
        result = trainer.predict(
            snapshot, output_config=jobs.prediction_output(args, connector))
    print(f'predictions: {result.num_rows} rows')
    return result


if __name__ == '__main__':
    main()
