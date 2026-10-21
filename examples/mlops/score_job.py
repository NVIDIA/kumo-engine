# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score with the champion model: ``python score_job.py [--job-id ID
--job-run-id ID]``; see README.md. ``KUMO_SNAPSHOT`` (a path under
``KUMO_ARTIFACT_ROOT``) scores an exported snapshot instead of fresh data."""

import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402

import kumo.engine as kumo  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> None:
    job_id, job_run_id = jobs.start(argv)
    with kumo.job(job_id=job_id, job_run_id=job_run_id) as run:
        connector = jobs.source()
        trainer = jobs.load_champion(connector)
        run.record(bundle_id=trainer.bundle_id)
        graph = jobs.build_graph(connector)
        exported = os.environ.get('KUMO_SNAPSHOT')
        snapshot = (kumo.GraphSnapshot.load_from_connector(
            jobs.artifacts(), path=exported, graph=graph)
                    if exported else graph.snapshot())
        run.record(snapshot_id=snapshot.id)

        output = jobs.prediction_output(connector)
        defaults = trainer.scoring_defaults.binary_classification_threshold
        result = trainer.predict_snapshot(
            snapshot, output_config=output,
            binary_classification_threshold=None
            if defaults is not None else 0.5)
        run.record(run_id=result.selection.job_id,
                   outputs=output.destinations())


if __name__ == '__main__':
    main()
