# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
from dataclasses import replace

import matplotlib
import pytest
from kumo_core.io.csv import save_csv
from kumoapi.explain.column_analysis import ColumnCohorts
from kumoapi.explain.gradient import GraphGradientScore, TableGradientScore
from kumoapi.typing import Stype

from kumo.engine.explain.plotting import plot_cohort, plot_gradient
from kumo.engine.explain.results import TrainingExplanations

matplotlib.use('Agg', force=True)


@pytest.fixture
def cohort():
    return ColumnCohorts(
        stype=Stype.numerical, cohorts=['Low', 'High'], populations=[.25, .75],
        predictions=[3., 7.], targets=[2., 8.], prediction_score=2.,
        target_score=3., table_name='users', hop=1, col_name='age')


@pytest.fixture(autouse=True)
def close_plots():
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


def test_gradient_preserves_ranked_raw_scores(monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: pytest.fail('unexpected display'))
    scores = GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': 4., 'shared': 2.}),
        'orders': TableGradientScore('orders', {'shared': 4., 'total': 1.}),
    })
    figure, axis = plot_gradient(scores, top_k=3)
    assert axis.figure is figure
    assert [bar.get_width() for bar in axis.patches] == [4., 4., 2.]
    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        'orders.shared', 'users.age', 'users.shared']
    assert axis.yaxis_inverted()
    assert axis.get_xlabel() == 'Gradient score'
    assert scores.tables['users'].columns['age'] == 4.


def test_zero_gradient_scores_remain_zero():
    scores = GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': 0.}),
    })
    _, axis = plot_gradient(scores)
    assert [bar.get_width() for bar in axis.patches] == [0.]


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5])
def test_invalid_gradient_limit(top_k):
    with pytest.raises(ValueError, match='positive integer'):
        plot_gradient(GraphGradientScore(), top_k=top_k)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf')])
def test_invalid_gradient_scores(score):
    scores = GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': score}),
    })
    with pytest.raises(ValueError, match='finite'):
        plot_gradient(scores)


def test_signed_recorded_scores_are_preserved():
    scores = GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': -1.}),
    })
    _, axis = plot_gradient(scores)
    assert [bar.get_width() for bar in axis.patches] == [-1.]


def test_empty_gradient_scores():
    with pytest.raises(ValueError, match='No gradient scores'):
        plot_gradient(GraphGradientScore())


def test_cohort_separates_population(cohort, monkeypatch):
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: pytest.fail('unexpected display'))
    figure, (prediction, population) = plot_cohort(cohort)
    assert list(figure.axes) == [prediction, population]
    assert prediction.lines[0].get_ydata().tolist() == [3., 7.]
    assert prediction.lines[1].get_ydata().tolist() == [2., 8.]
    assert [bar.get_height() for bar in population.patches] == [.25, .75]
    assert population.get_ylabel() == 'Population share'
    assert [tick.get_text() for tick in population.get_xticklabels()] == ['Low', 'High']
    assert prediction.get_title() == 'users[hop=1].age'


def test_cohort_without_targets(cohort):
    _, (prediction, _) = plot_cohort(replace(cohort, targets=None, target_score=None))
    assert len(prediction.lines) == 1


def test_class_selection_preserves_values(cohort):
    classification = replace(
        cohort, predictions={'no': [.9, .2], 'yes': [.1, .8]},
        targets={'no': [1., .1], 'yes': [0., .9]})
    with pytest.raises(ValueError, match='class_name is required'):
        plot_cohort(classification)
    with pytest.raises(ValueError, match='Unknown class'):
        plot_cohort(classification, class_name='missing')
    _, (prediction, _) = plot_cohort(classification, class_name='yes')
    assert prediction.lines[0].get_ydata().tolist() == [.1, .8]
    assert prediction.lines[1].get_ydata().tolist() == [0., .9]
    assert 'yes' in prediction.get_title()


def test_class_requires_class_predictions(cohort):
    with pytest.raises(ValueError, match='class-keyed'):
        plot_cohort(cohort, class_name='yes')


@pytest.mark.parametrize('changes', [
    {'populations': [.25]},
    {'predictions': [float('nan'), 1.]},
    {'targets': [1.]},
    {'populations': [.2, .2]},
    {'populations': [-.1, 1.1]},
    {'table_name': '???'},
    {'predictions': {'yes': [.1, .8]}, 'targets': {'no': [.9, .2]}},
])
def test_invalid_cohort_data(cohort, changes):
    with pytest.raises(ValueError):
        plot_cohort(replace(cohort, **changes), class_name=None)


def test_missing_plot_dependency(cohort, monkeypatch, tmp_path):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'matplotlib.pyplot':
            raise ModuleNotFoundError("No module named 'matplotlib'", name='matplotlib')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(ImportError, match=r'kumo-engine\[plots\]'):
        plot_cohort(cohort)
    scores = GraphGradientScore(tables={
        'users': TableGradientScore('users', {'age': 1.}),
    })
    save_csv(scores.to_pandas(), str(tmp_path / 'explain_test/scores.csv'))
    result = TrainingExplanations.from_trial(tmp_path, tmp_path)
    assert result.gradient_scores() == scores
    assert result.availability['gradient'].available
    with pytest.raises(ImportError, match=r'kumo-engine\[plots\]'):
        result.plot_gradient()
