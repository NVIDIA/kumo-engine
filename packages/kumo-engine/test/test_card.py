# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.summaries import Note, Table

from kumo.engine import _card

MODEL = {'kumo.model_name': 'kumo-demo-churn', 'kumo.model_version': '6'}
QUERY = '`PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.USER_ID`'


def test_a_fit_heading_leads_with_the_tuned_holdout_metric():
    tags = {**MODEL, 'kumo.tune_metric': 'auroc', 'kumo.data_warnings': '2'}

    assert _card.heading('fit', {'test/auroc': 0.7871, 'test/ap': 0.5},
                         tags) == [
        'holdout auroc 0.787, 2 data problems', '']
    assert _card.heading('fit', {'test/loss': 0.25}, {
        'kumo.tune_metric': 'loss', 'kumo.data_warnings': '0',
        'kumo.error': 'ValueError:\n  stop'})[0] == (
            'failed: ValueError: stop, holdout loss 0.25')


def test_a_prediction_heading_leads_with_its_watch_numbers():
    metrics = {'drift/psi': 0.031, 'pred/rows': 10000,
               'pred/skipped_share': 0.0125, 'realized/auroc': 0.7,
               'realized/labelled_share': 1.0}
    tags = {**MODEL, 'kumo.as_of': '2026-10-29', 'kumo.data_warnings': '1'}

    assert _card.heading('prediction', metrics, tags)[0] == (
        'realized auroc 0.7, psi 0.031 vs holdout (small shift), 10,000 rows '
        'as of 2026-10-29, 1.2% skipped, 1 data problem')
    unlabelled = {name: value for name, value in metrics.items()
                  if not name.startswith('realized/')}
    assert _card.heading('prediction', {**unlabelled, 'drift/psi': 0.3}, {
        **tags, 'kumo.entity': 'users',
        'kumo.labels_mature': '2026-11-05'})[0] == (
        'psi 0.3 vs holdout (large shift), 10,000 users as of 2026-10-29, '
        'labels mature 2026-11-05, 1.2% skipped, 1 data problem')
    assert _card.heading('prediction', {}, {}) == ['prediction', '']
    long = _card.heading('prediction', {}, {'kumo.error': 'x' * 200})[0]
    assert len(long) == _card.HEADLINE_CHARS and long.endswith('x…')


def test_a_fit_says_what_it_trained_and_where_its_numbers_come_from():
    tags = {**MODEL, 'kumo.task_type': 'binary_classification',
            'kumo.tune_metric': 'auroc', 'kumo.best_trial': '2',
            'kumo.best_epoch': '4'}

    assert _card.intro('fit', tags, {'training_job.num_experiments': '3'},
                       QUERY) == (
        'Fit of `kumo-demo-churn` v6 for binary classification of users; its '
        'curves and holdout metrics are from trial 2 of 3, epoch 4, the best '
        f'by val auroc.\n\n{QUERY}')
    assert _card.intro('fit', {'kumo.best_epoch': '1'}, {}, None) == (
        'Fit; its holdout metrics are from epoch 1, the best by val loss.')


def test_the_body_holds_problems_tables_notes_and_locations():
    summaries = (
        *_card.holdout('fit', {'test/ap': 0.5, 'test/auroc': 0.75,
                               'val/auroc': 0.7},
                       {'kumo.tune_metric': 'auroc'}),
        Note('problems', 'table `users`: 12 duplicate keys'),
        Table('data/splits', ('split', 'rows', 'positive rate'),
              (('train', 26000, 0.498), ('test', 2000, None)), brief=True),
        Note('notes', 'bins are the test target quantiles'),
        Table('locations', ('kind', 'id', 'location'), (
            ('model', 'a' * 64, f"s3://bucket/root/models/{'a' * 64}"),
            ('source_table', 'orders', '/data/orders')), brief=True),
    )

    assert '\n'.join(_card.body('Fit.', summaries, [
        '_provenance.json', 'data/health.json'])) == '\n'.join([
        '', 'Fit.', '', '**Problems**', '',
        '- table `users`: 12 duplicate keys', '', '**Holdout metrics**', '',
        '| metric | value |', '| --- | ---: |', '| test/auroc | 0.75 |',
        '| test/ap (average precision) | 0.5 |', '', '**Training data**', '',
        '| split | rows | positive rate |', '| --- | ---: | ---: |',
        '| train | 26,000 | 0.498 |', '| test | 2,000 |  |', '',
        '**Notes**', '', '- bins are the test target quantiles', '',
        '**Where things are**', '', '| Artifact | Location |',
        '| --- | --- |',
        f"| model bundle `aaaaaaaa` | [`s3://bucket/root/models/{'a' * 64}`]"
        '(https://s3.console.aws.amazon.com/s3/buckets/bucket?prefix=root/'
        f"models/{'a' * 64}/) |",
        '| source table `orders` | `/data/orders` |', '',
        'Full lineage: `_provenance.json`. Tables: `data/health.json`.'])
    assert _card.holdout('prediction', {'test/ap': 0.5}, {}) == []


def test_a_link_prediction_fit_shows_its_holdout_metrics_at_the_query_k():
    metrics = {'test/map_at_1': 0.25, 'test/map_at_10': 0.125,
               'test/recall_at_10': 0.5, 'test/locality_score': 0.75}

    table, note = _card.holdout('fit', metrics, {'kumo.tune_metric': 'map@10'})
    assert table.rows == (('test/map_at_10', 0.125),
                          ('test/locality_score', 0.75),
                          ('test/recall_at_10', 0.5))
    assert note == Note('notes',
                        'Holdout metrics at other k: `holdout/by_k.json`.')


def test_a_prediction_names_the_model_and_its_fit_run():
    assert _card.model_line(MODEL, 'v6-churn', '#/experiments/1/runs/r') == (
        'Scores of `kumo-demo-churn` v6, trained by '
        '[v6-churn](#/experiments/1/runs/r).')
    assert _card.model_line({'kumo.kind': 'monitoring',
                             'kumo.monitor': 'drift',
                             'kumo.bundle_id': 'b' * 64}) == (
        'Drift check of the scores of bundle `bbbbbbbb`.')


def test_a_version_headline_leads_with_the_tuned_metric():
    metrics = {'test/mae': 0.25, 'before_refit/mae': 0.3, 'test/loss': 0.5,
               'test/mse': 1.0, 'test/rmse': 1.0, 'test/map_at_10': 0.125}
    assert _card.version_headline(metrics, 'loss') == (
        'test/loss 0.5 · test/mae 0.25 · test/mse 1 · test/rmse 1')
    assert _card.version_headline(metrics, 'map@10') == (
        'test/map_at_10 0.125 · test/mae 0.25 · test/loss 0.5 · test/mse 1')


def test_a_prediction_heading_leads_with_the_tuned_realized_metric():
    metrics = {'realized/ap': 0.71, 'realized/auroc': 0.78}

    assert _card.heading('prediction', metrics,
                         {'kumo.tune_metric': 'auroc'})[0] == (
        'realized auroc 0.78')
    assert _card.heading('prediction', metrics, {})[0] == 'realized ap 0.71'


def test_data_preparation_marks_a_snapshot_built_by_another_process(
        monkeypatch):
    from kumo.engine import _prep_cost
    from kumo.engine._manifest import (
        BuildCost,
        SnapshotManifest,
        TrainingTableManifest,
    )

    cost = BuildCost(started_at='2026-10-29T06:00:00+00:00',
                     finished_at='2026-10-29T06:01:30+00:00', seconds=90.0,
                     disk_bytes=3 * 1024**3)
    monkeypatch.setattr(_prep_cost, '_built_here', {'tt'})
    table, = _prep_cost.prep_summaries(
        SnapshotManifest(id='snap', build=cost),
        TrainingTableManifest(id='tt', build=BuildCost(
            started_at='', finished_at='', seconds=12.5,
            disk_bytes=1024**3 // 2)))

    assert _card.table_text(table) == '\n'.join([
        '**Data preparation**', '', '| step | time (s) | disk (GiB) |',
        '| --- | ---: | ---: |',
        '| Graph snapshot (built earlier, reused) | 90 | 3 |',
        '| Training table | 12.5 | 0.5 |'])
    assert _prep_cost.prep_summaries(SnapshotManifest(id='old')) == []
