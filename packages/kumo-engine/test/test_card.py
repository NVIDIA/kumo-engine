# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.summaries import Note, Table

from kumo.engine import _card

MODEL = {'kumo.model_name': 'kumo-demo-churn', 'kumo.model_version': '6'}
QUERY = '`PREDICT COUNT(orders.*, 0, 7, days) > 0 FOR EACH users.USER_ID`'


def test_a_fit_headline_leads_with_the_tuned_holdout_metric():
    tags = {**MODEL, 'kumo.tune_metric': 'auroc', 'kumo.data_warnings': '2',
            'kumo.task_type': 'binary_classification'}
    params = {'training_job.num_experiments': '3'}

    assert _card.headline('fit', {'auroc': 0.7871, 'ap': 0.5}, tags, params,
                          QUERY) == (
        'auroc 0.787 · 2 data problems · v6 kumo-demo-churn · best of 3 '
        'trials · binary classification of users')
    assert _card.headline('fit', {'holdout_loss': 0.25}, {
        'kumo.tune_metric': 'loss', 'kumo.data_warnings': '0',
        'kumo.error': 'ValueError: stop'}, {}) == (
            'failed: ValueError: stop · holdout_loss 0.25')


def test_a_prediction_headline_leads_with_its_watch_numbers():
    metrics = {'drift/psi': 0.031, 'pred/rows': 10000,
               'pred/skipped_share': 0.0125, 'realized/auroc': 0.7,
               'realized/labelled_share': 1.0}
    tags = {**MODEL, 'kumo.as_of': '2026-10-29', 'kumo.data_warnings': '1'}

    assert _card.headline('prediction', metrics, tags, {}) == (
        'realized auroc 0.7 · psi 0.031 vs training · 10,000 rows as of '
        '2026-10-29 · 1.2% skipped · 1 data problem · v6 kumo-demo-churn')
    assert _card.headline('prediction', {}, {}, {}) == 'prediction'
    long = _card.headline('prediction', {}, {'kumo.error': 'x' * 200}, {})
    assert len(long) == _card.HEADLINE_CHARS and long.endswith('x…')


def test_the_body_holds_problems_tables_notes_and_locations():
    summaries = (
        Note('problems', 'link `orders.ITEM_ID → items`: 14 % match'),
        Note('problems', 'table `users`: 12 duplicate keys'),
        Table('data/splits', ('split', 'rows', 'positive rate'),
              (('train', 26000, 0.498), ('val', 2000, 0.491),
               ('test', 2000, None)), brief=True),
        Table('pred/summary', ('column', 'mean', 'min'),
              (('True_PROB', 0.4834, 0.012),), brief=True),
        Note('notes', 'bins are the test target quantiles'),
        Note('locations', 'root `s3://bucket/root/`'),
        Note('locations', 'bundle `models/abc/`'),
    )

    assert _card.body(QUERY, summaries, [
        '_provenance.json', 'data/health.json']) == [
        QUERY, '',
        '**Problems:** link `orders.ITEM_ID → items`: 14 % match · table '
        '`users`: 12 duplicate keys', '',
        '**Training data:**\n\n| split | rows | positive rate |\n'
        '| --- | ---: | ---: |\n| train | 26,000 | 0.498 |\n'
        '| val | 2,000 | 0.491 |\n| test | 2,000 |  |', '',
        '**Scores:** True_PROB: mean 0.483 · min 0.012', '',
        '**Notes:** bins are the test target quantiles', '',
        '**Where things are**', '- root `s3://bucket/root/`',
        '- bundle `models/abc/`', '',
        'Full lineage: `_provenance.json`. Tables: `data/health.json`.']


def test_a_prediction_names_the_model_and_its_fit_run():
    assert _card.model_line('churn', '6', 'b' * 64, 'v6-churn',
                            '#/experiments/1/runs/r') == (
        'Model: `churn` v6, trained by [v6-churn](#/experiments/1/runs/r).')
    assert _card.model_line(None, None, 'b' * 64) == 'Model: bundle `bbbbbbbb`.'


def test_a_version_headline_leads_with_the_tuned_metric():
    metrics = {'mae': 0.25, 'before_refit/mae': 0.3, 'holdout_loss': 0.5,
               'mse': 1.0, 'rmse': 1.0, 'map@10': 0.125}
    assert _card.version_headline(metrics, 'loss') == (
        'holdout_loss 0.5 · mae 0.25 · mse 1 · rmse 1')
    assert _card.version_headline(metrics, 'map@10') == (
        'map_at_10 0.125 · mae 0.25 · holdout_loss 0.5 · mse 1')
