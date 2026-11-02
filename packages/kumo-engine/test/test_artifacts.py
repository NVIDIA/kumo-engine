# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional

from kumo_core.artifacts import bundle_id_from_manifest
from kumo_core.summaries import Dataset, Table, Tag

import kumo.engine as kumo
from kumo.engine._deps import Dependencies, FileAsset
from kumo.engine._manifest import (
    FIT_ROLES,
    Artifact,
    Dependency,
    Manifest,
    ModelManifest,
    ModelParents,
    OutputDestination,
    PredictionManifest,
    PredictionOutput,
    PredictionParents,
    PredictionTableRecord,
    Record,
    RunScoring,
    SnapshotManifest,
    SnapshotParents,
    SourceTableEntry,
    TrainingTableManifest,
    TrainingTableParents,
    artifact_summaries,
    artifacts,
    read_manifest,
)
from kumo.engine._snapshot_io import store
from kumo.engine.trainer.trainer import _store_and_register


@dataclass(frozen=True, kw_only=True)
class ReportParents(Record):
    snapshot: SnapshotManifest
    model: ModelManifest


@dataclass(frozen=True, kw_only=True)
class ReportManifest(Manifest):
    kind: str = 'report'
    parents: ReportParents
    pages: Optional[List[SnapshotManifest]] = None
    dependencies: Optional[List[Dependency]] = None

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
                 version='r', version_kind='exact', label='report r'),
        Artifact('snapshot', 'g@d', 's3://root/snapshots/g/d', True,
                 version='g@d', version_kind='exact',
                 label='graph snapshot g@d'),
        Artifact('source_table', 'orders', 's3://data/orders', True,
                 connector='S3', version='7', version_kind='exact'),
        Artifact('model', 'm', 's3://root/models/m', False,
                 'kumo.bundle_uri', version='m', version_kind='exact',
                 label='model m'),
        Artifact('snapshot', 'p', None, True, version='p',
                 version_kind='exact', label='graph snapshot p'),
    ]


def test_each_store_records_its_location_and_keeps_the_ids(fitted, tmp_path):
    trainer, _, snapshot = fitted
    bundle = trainer.result.bundle_dir
    ids = snapshot.id, trainer.result.id, bundle_id_from_manifest(bundle)
    for root in (tmp_path / 'a', tmp_path / 'b'):
        settings = kumo.MLOps(str(root))
        store(snapshot, settings)
        _store_and_register(bundle, trainer.result.manifest, settings)
        assert snapshot.manifest.location == settings.snapshot_uri(snapshot)
        assert trainer.result.manifest.location == f'{root}/models/{ids[1]}'
        assert read_manifest(root / 'models' / ids[1] / '_provenance.json',
                             ModelManifest).location == f'{root}/models/{ids[1]}'
        assert (snapshot.id, trainer.result.id,
                bundle_id_from_manifest(bundle)) == ids


def test_a_new_kind_and_asset_reach_tags_and_datasets(tmp_path):
    settings = kumo.MLOps(str(tmp_path / 'root'))
    (tmp_path / 'notes.txt').write_text('notes')
    asset = FileAsset('dummy', 'notes', tmp_path / 'notes.txt')
    dependency = Dependencies(settings=settings).require(asset)
    root = f"{tmp_path / 'root'}"
    orders = SourceTableEntry(connector='S3', location='s3://data/orders',
                              version='7', version_kind='exact')
    report = ReportManifest(
        id='r', location=f'{root}/reports/r',
        parents=ReportParents(
            snapshot=SnapshotManifest(
                id='g@d', location=f'{root}/snapshots/g/d',
                parents=SnapshotParents(source_tables={'orders': orders})),
            model=ModelManifest(id='m', location=f'{root}/models/m')),
        dependencies=[dependency])

    assert dependency.location == f'{root}/assets/dummy/{dependency.sha256}'
    assert ReportManifest.from_dict(report.to_dict()) == report
    assert artifact_summaries(report, FIT_ROLES, {
        'orders': ({'ID': {'stype': 'ID', 'dtype': 'int64'}}, {'rows': 3})
    }) == [
        Tag('kumo.bundle_uri', f'{root}/models/m'),
        Table('locations', ('kind', 'id', 'location'), (
            ('report', 'r', f'{root}/reports/r'),
            ('snapshot', 'g@d', f'{root}/snapshots/g/d'),
            ('source_table', 'orders', 's3://data/orders'),
            ('model', 'm', f'{root}/models/m'),
            ('asset', dependency.sha256, dependency.location)), brief=True),
        Dataset('graph snapshot g@d', {
            'connector': 'LOCAL_FILE', 'location': f'{root}/snapshots/g/d',
            'version': 'g@d', 'version_kind': 'exact'}, {}, {},
            'graph context'),
        Dataset('orders', {'connector': 'S3', 'location': 's3://data/orders',
                           'version': '7', 'version_kind': 'exact'},
                {'ID': {'stype': 'ID', 'dtype': 'int64'}}, {'rows': 3},
                'source table'),
        Dataset('report r', {'connector': 'LOCAL_FILE',
                             'location': f'{root}/reports/r', 'version': 'r',
                             'version_kind': 'exact'}, {}, {}, 'report'),
    ]


def test_a_fits_datasets_are_named_and_labelled_by_role():
    users = SourceTableEntry(connector='S3', location='s3://data/users',
                             version='3', version_kind='exact')
    snapshot = SnapshotManifest(
        id='g@5ac245', created_at='2026-11-01T09:00:00+00:00',
        location='s3://root/snapshots/g/5ac245',
        parents=SnapshotParents(source_tables={'users': users}))
    model = ModelManifest(id='m', parents=ModelParents(
        training_table=TrainingTableManifest(
            id='tt-fadf04', created_at='2026-11-01T10:00:00+00:00',
            location='s3://root/training_tables/tt-fadf04',
            parents=TrainingTableParents(snapshot=snapshot))))

    assert [(s.name, s.context, s.source['version'])
            for s in artifact_summaries(model, FIT_ROLES)
            if isinstance(s, Dataset)] == [
        ('graph snapshot 2026-11-01-5ac245', 'graph context', 'g@5ac245'),
        ('training table 2026-11-01-fadf04', 'training examples',
         'tt-fadf04'),
        ('users', 'source table', '3'),
    ]


def test_a_prediction_run_reads_its_snapshot_and_writes_its_outputs():
    orders = SourceTableEntry(connector='S3', location='s3://data/orders',
                              version='7', version_kind='exact')
    run = PredictionManifest(
        id='p', location='s3://root/predictions/p',
        parents=PredictionParents(
            model=ModelManifest(id='m', location='s3://root/models/m'),
            snapshot=SnapshotManifest(
                id='g@d', parents=SnapshotParents(
                    source_tables={'orders': orders}))),
        prediction_table=PredictionTableRecord(
            query_sha256='q', source='pql', anchors=[], rows=2),
        scoring=RunScoring(
            binary_classification_threshold=None, threshold_source='none',
            num_classes_to_return=None, count_source='none',
            num_workers=None),
        outputs=[
            PredictionOutput(artifact='predictions',
                             destination=OutputDestination(
                                 connector='S3', location='s3://out/p')),
            PredictionOutput(artifact='embeddings',
                             destination=OutputDestination(
                                 connector='SNOWFLAKE',
                                 location='"DB"."S"."EMB"')),
        ], status='succeeded', timings=None, stats=None)

    assert artifact_summaries(run, {'snapshot': 'scoring',
                                    'source_table': 'scoring'}) == [
        Tag('kumo.bundle_uri', 's3://root/models/m'),
        Tag('kumo.output_uri', 's3://out/p'),
        Tag('kumo.output_table', '"DB"."S"."EMB"'),
        Table('locations', ('kind', 'id', 'location'), (
            ('prediction', 'p', 's3://root/predictions/p'),
            ('model', 'm', 's3://root/models/m'),
            ('source_table', 'orders', 's3://data/orders'),
            ('prediction_output', 'predictions', 's3://out/p'),
            ('prediction_output', 'embeddings', '"DB"."S"."EMB"')),
              brief=True),
        Dataset('orders', {'connector': 'S3', 'location': 's3://data/orders',
                           'version': '7', 'version_kind': 'exact'}, {}, {},
                'scoring'),
        Dataset('predictions', {'connector': 'S3', 'location': 's3://out/p',
                                'version': None, 'version_kind': None},
                {}, {}, 'output'),
        Dataset('embeddings', {'connector': 'SNOWFLAKE',
                               'location': '"DB"."S"."EMB"', 'version': None,
                               'version_kind': None}, {}, {}, 'output'),
    ]
