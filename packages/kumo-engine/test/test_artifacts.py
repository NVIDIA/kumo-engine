# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional

from kumo.engine._manifest import (
    Artifact,
    Manifest,
    ModelManifest,
    ModelParents,
    Record,
    SnapshotManifest,
    SnapshotParents,
    SourceTableEntry,
    TrainingTableManifest,
    artifacts,
)


@dataclass(frozen=True, kw_only=True)
class ReportParents(Record):
    snapshot: SnapshotManifest
    model: ModelManifest


@dataclass(frozen=True, kw_only=True)
class ReportManifest(Manifest):
    kind: str = 'report'
    parents: ReportParents
    pages: Optional[List[SnapshotManifest]] = None

    IS_DATASET = True


def test_a_new_kind_names_its_artifacts_without_other_changes():
    orders = SourceTableEntry(connector='S3', location='s3://data/orders',
                              version='7', version_kind='exact')
    snapshot = SnapshotManifest(
        id='g@d', location='s3://root/snapshots/g/d',
        parents=SnapshotParents(source_tables={'orders': orders}))
    model = ModelManifest(
        id='m', location='s3://root/models/m', parents=ModelParents(
            training_table=TrainingTableManifest(id='t')))
    report = ReportManifest(
        id='r', location='s3://root/reports/r',
        parents=ReportParents(snapshot=snapshot, model=model),
        pages=[snapshot, SnapshotManifest(id='p', reconstructed=True)])

    assert artifacts(report) == [
        Artifact('report', 'r', 's3://root/reports/r', True,
                 'kumo.report_uri', version='r', version_kind='exact'),
        Artifact('snapshot', 'g@d', 's3://root/snapshots/g/d', True,
                 'kumo.snapshot_uri', version='g@d', version_kind='exact'),
        Artifact('source_table', 'orders', 's3://data/orders', True,
                 connector='S3', version='7', version_kind='exact'),
        Artifact('model', 'm', 's3://root/models/m', False,
                 'kumo.bundle_uri', version='m', version_kind='exact'),
        Artifact('snapshot', 'p', None, True, 'kumo.snapshot_uri',
                 version='p', version_kind='exact'),
    ]
