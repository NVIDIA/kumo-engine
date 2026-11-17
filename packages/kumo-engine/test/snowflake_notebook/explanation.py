# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real binary and class-selected refit explanations in the managed notebook."""
import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from xml.etree import ElementTree

PLOTS = ('gradient', 'cohort', 'cohort-class-0', 'cohort-class-1',
         'cohort-multilabel-0', 'cohort-multilabel-1', 'cohort-regression',
         'gradient-layout', 'gradient-zero', 'cohort-layout')
COHORT_KEY = ('events', 1, 'COUNT(*)')
EXPORTS = tuple(f'explanation-{kind}.{suffix}' for kind in PLOTS
                for suffix in ('png', 'svg', 'pdf'))
FILES = (*EXPORTS, 'explanation-result.json', 'explanation-run.tar.gz')


def fit(root):
    import torch
    from wheel_mlops_lifecycle import (
        QUERY,
        TRIALS,
        graph,
        selected,
        small_plan,
    )

    import kumo.engine as kumo

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(dir=root, prefix='fit-') as scratch:
        scratch = Path(scratch)
        retained = scratch / 'retained'
        retained.mkdir()
        for task in ('binary', 'multiclass', 'multilabel', 'regression'):
            work = scratch / task
            work.mkdir()
            kumo.configure(scratch_dir=work / 'scratch')
            training_graph = graph(
                work, '2024-03-01', categorical=task
                in {'multiclass', 'multilabel'})
            snapshot = training_graph.snapshot()
            query = {
                'binary':
                QUERY,
                'multiclass':
                'PREDICT FIRST(events.KIND, 0, 7, days) FOR EACH users.ID',
                'multilabel': ('PREDICT LIST_DISTINCT(events.KIND, 0, 7, days) '
                               'CLASSIFY TOP 3 FOR EACH users.ID'),
                'regression':
                'PREDICT COUNT(events.*, 0, 7, days) FOR EACH users.ID',
            }[task]
            table = kumo.PredictiveQuery(
                graph=training_graph,
                query=query).generate_training_table(snapshot)
            plan = small_plan(1, TRIALS if task == 'binary' else 1, task=task)
            plan.training_job.disable_explain = False
            plan.training_job.refit_trainval = True
            result = kumo.Trainer(plan, accelerator='cpu').fit(
                table, snapshot=snapshot)
            explained = result.explanations
            assert result.manifest.fit_mode == 'refit_trainval'
            assert all(a.available and a.generation_failure is None for a in
                       explained.availability.values()), explained.availability
            trial = selected(
                explained,
                os.path.relpath(result.holdout_path.parent, explained.run_root))
            destination = retained if task == 'binary' else retained / task
            destination.mkdir(exist_ok=True)
            shutil.copytree(explained.run_root, destination / 'training-run')
            result.export(
                kumo.ModelOutputConfig(
                    kumo.FileConnector(root_dir=str(destination)),
                    path='untracked-model'))
            state = dict(
                bundle=result.id, explanation_trial=trial,
                gradient=explained.gradient_scores().to_pandas().to_dict(
                    'records'))
            (destination / 'fit.json').write_text(
                json.dumps({'result': {
                    'untracked': state
                }}))
        with tarfile.open(root / 'explanation-run.tar.gz', 'w:gz') as archive:
            for name in ('training-run', 'untracked-model', 'fit.json',
                         'multiclass', 'multilabel', 'regression'):
                archive.add(retained / name, arcname=name)
    # The fit process exits with its scratch and original run removed.


def render(root):
    import matplotlib.pyplot as plt
    from wheel_mlops_lifecycle import explain, selected

    import kumo.engine as kumo

    retained = root / 'reopened-run'
    retained.mkdir()
    with tarfile.open(root / 'explanation-run.tar.gz') as archive:
        for entry in archive.getmembers():
            assert (entry.isfile()
                    or entry.isdir()) and not Path(entry.name).is_absolute()
            assert '..' not in Path(entry.name).parts
        archive.extractall(retained, filter='data')
    # Reuse the lifecycle's actual-refit attribution, sampled-graph and negative checks.
    proof = explain(retained, check_local_permissions=False)
    plt.close('all')
    state = json.loads(
        (retained / 'fit.json').read_text())['result']['untracked']
    assert kumo.Trainer.load_from_connector(
        kumo.FileConnector(root_dir=str(retained)), path='untracked-model',
        accelerator='cpu').result.id == state['bundle']
    proof['bundle'] = state['bundle']
    tasks = {}
    for task in ('multiclass', 'multilabel', 'regression'):
        directory = retained / task
        task_state = json.loads(
            (directory / 'fit.json').read_text())['result']['untracked']
        explanations = kumo.TrainingExplanations.load(directory /
                                                      'training-run')
        selected(explanations, task_state['explanation_trial'])
        assert kumo.Trainer.load_from_connector(
            kumo.FileConnector(root_dir=str(directory)), path='untracked-model',
            accelerator='cpu').result.id == task_state['bundle']
        classes = explanations.classes(COHORT_KEY)
        assert (set(classes) == {'morning', 'afternoon', 'evening'}
                if task != 'regression' else classes == ()), classes
        tasks[task] = dict(status='passed', fresh_process='passed',
                           bundle=task_state['bundle'], classes=classes)
    report = dict(status='passed', fresh_process='passed', refit=proof,
                  bundle=state['bundle'], plots={}, **tasks)
    (root / 'explanation-result.json').write_text(json.dumps(report, indent=2))
    return report


def check_layout(figure):
    """Check every visible title/axis/tick label fits, including vector exports."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    labels = []
    for axis in figure.axes:
        ticks = []
        for positions, texts, limits in ((axis.get_xticks(),
                                          axis.get_xticklabels(),
                                          axis.get_xlim()),
                                         (axis.get_yticks(),
                                          axis.get_yticklabels(),
                                          axis.get_ylim())):
            low, high = sorted(limits)
            ticks.append([
                text for value, text in zip(positions, texts) if
                low <= value <= high and text.get_visible() and text.get_text()
            ])
        texts = [
            axis.title, axis.xaxis.label, axis.yaxis.label, *ticks[0], *ticks[1]
        ]
        legend = axis.get_legend()
        if legend is not None:
            texts.extend(legend.get_texts())
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= -1 and bounds.y0 >= -1, text.get_text()
            assert bounds.x1 <= figure.bbox.width + 1, text.get_text()
            assert bounds.y1 <= figure.bbox.height + 1, text.get_text()
            labels.append(text.get_text())
        for tick_labels in ticks:
            bounds = [text.get_window_extent(renderer) for text in tick_labels]
            assert all(not left.overlaps(right)
                       for left, right in zip(bounds, bounds[1:]))
    return labels


def layout_examples(report):
    """Supplement real runs with explicit plot-boundary inputs, never model evidence."""
    from kumoapi.explain.column_analysis import ColumnCohorts
    from kumoapi.explain.gradient import GraphGradientScore, TableGradientScore
    from kumoapi.typing import Stype

    from kumo.engine.explain.plotting import plot_cohort, plot_gradient

    table = 'customer_purchase_history'
    columns = {
        f'long_feature_name_{index:02d}_with_suffix': float(index % 3)
        for index in range(24)
    }
    scores = GraphGradientScore(
        tables={table: TableGradientScore(table, columns)})
    gradient, axis = plot_gradient(scores, top_k=24)
    gradient.set_size_inches(12, 10)
    gradient.tight_layout()
    expected = sorted(columns, key=lambda col: (-columns[col], col))
    assert [t.get_text() for t in axis.get_yticklabels()
            ] == [f'{table}.{col}' for col in expected]
    assert [bar.get_width()
            for bar in axis.patches] == [columns[col] for col in expected]
    zeros, zero_axis = plot_gradient(
        GraphGradientScore(
            tables={
                table:
                TableGradientScore(table, {
                    'first': 0.,
                    'second': 0.,
                    'third': 0.
                })
            }))
    assert [bar.get_width() for bar in zero_axis.patches] == [0., 0., 0.]
    cohort = ColumnCohorts(
        stype=Stype.categorical,
        cohorts=[f'Long cohort label number {i:02d}' for i in range(12)],
        populations=[1 / 12] * 12, predictions=[0., .5, .5] * 4, targets=None,
        prediction_score=0., target_score=None, table_name=table, hop=1,
        col_name='long_categorical_feature_name')
    cohort_figure, (prediction, population) = plot_cohort(cohort)
    cohort_figure.set_size_inches(24, 8)
    cohort_figure.tight_layout()
    assert len(prediction.lines) == 1
    assert list(prediction.lines[0].get_ydata()) == cohort.predictions
    assert [bar.get_height()
            for bar in population.patches] == cohort.populations
    for kind in ('gradient-layout', 'gradient-zero', 'cohort-layout'):
        report['plots'][kind] = dict(source='deterministic-plot-input')
    report['layout_examples'] = dict(status='passed', columns=24, cohorts=12,
                                     tied_and_zero_scores='passed',
                                     absent_targets='passed')
    return [('gradient-layout', gradient, 'Training column importance'),
            ('gradient-zero', zeros, 'Training column importance'),
            ('cohort-layout', cohort_figure, 'Population share')]


def display_outputs(root, report, display):
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    from PIL import Image
    from pypdf import PdfReader
    from wheel_mlops_lifecycle import expect

    import kumo.engine as kumo

    shell = get_ipython()
    assert shell is not None, 'Explanation UI checks require a real notebook kernel'
    shell.run_line_magic('matplotlib', 'inline')
    explained = kumo.TrainingExplanations.load(root /
                                               'reopened-run/training-run')
    gradient, axis = explained.plot_gradient(top_k=5)
    labels = [text.get_text() for text in axis.get_yticklabels()]
    scores = explained.gradient_scores().to_pandas().sort_values(
        ['score', 'table', 'column'], ascending=[False, True, True]).head(5)
    assert labels == [
        f'{row.table}.{row.column}' for row in scores.itertuples()
    ]
    assert [bar.get_width() for bar in axis.patches] == scores.score.tolist()
    cohort, (prediction, population) = explained.plot_cohort(
        ('events', 1, 'COUNT(*)'))
    recorded = explained.cohort(('events', 1, 'COUNT(*)'))
    assert prediction.get_ylabel() == 'Average value'
    assert population.get_ylabel() == 'Population share'
    assert list(prediction.lines[0].get_ydata()) == recorded.predictions
    assert list(prediction.lines[1].get_ydata()) == recorded.targets
    assert [bar.get_height()
            for bar in population.patches] == recorded.populations
    figures = [('gradient', gradient, 'Training column importance'),
               ('cohort', cohort, 'Population share')]
    for task, prefix in (('multiclass', 'class'), ('multilabel', 'multilabel')):
        classified = kumo.TrainingExplanations.load(
            root / f'reopened-run/{task}/training-run')
        for method in (classified.cohort, classified.plot_cohort):
            expect(ValueError, lambda: method(COHORT_KEY), 'class')
            expect(KeyError,
                   lambda: method(COHORT_KEY, class_name='missing-class'),
                   'missing-class')
        selected_classes = sorted(classified.classes(COHORT_KEY))[:2]
        class_predictions = []
        for index, class_name in enumerate(selected_classes):
            recorded = classified.cohort(COHORT_KEY, class_name=class_name)
            figure, (class_axis,
                     shares) = classified.plot_cohort(COHORT_KEY,
                                                      class_name=class_name)
            assert list(class_axis.lines[0].get_ydata()
                        ) == recorded.predictions[class_name]
            assert list(
                class_axis.lines[1].get_ydata()) == recorded.targets[class_name]
            assert [bar.get_height()
                    for bar in shares.patches] == recorded.populations
            assert class_axis.get_title().endswith(' — ' + class_name)
            kind = f'cohort-{prefix}-{index}'
            report['plots'][kind] = dict(
                class_name=class_name, title=class_axis.get_title(),
                labels=[class_name, 'Average value', 'Population share'])
            class_predictions.append(recorded.predictions[class_name])
            figures.append((kind, figure, class_name))
        assert class_predictions[0] != class_predictions[
            1], 'Class selection must change predictions'
        report[task]['selected_classes'] = selected_classes
        report[task]['invalid_selection'] = 'passed'
    regression = kumo.TrainingExplanations.load(
        root / 'reopened-run/regression/training-run')
    recorded = regression.cohort(COHORT_KEY)
    figure, (regression_axis, shares) = regression.plot_cohort(COHORT_KEY)
    assert list(regression_axis.lines[0].get_ydata()) == recorded.predictions
    assert list(regression_axis.lines[1].get_ydata()) == recorded.targets
    assert [bar.get_height() for bar in shares.patches] == recorded.populations
    figures.append(('cohort-regression', figure, 'Population share'))
    figures.extend(layout_examples(report))
    for kind, figure, label in figures:
        labels_to_check = check_layout(figure)
        native, metadata = shell.display_formatter.format(figure)
        assert 'image/png' in native, 'Notebook must format the actual Matplotlib figure'
        files = {}
        for suffix in ('png', 'svg', 'pdf'):
            file = root / f'explanation-{kind}.{suffix}'
            if suffix == 'png':
                file.write_bytes(base64.b64decode(native['image/png']))
            else:
                figure.savefig(file)
            if suffix == 'png':
                with Image.open(file) as image:
                    image.verify()
            elif suffix == 'svg':
                assert ElementTree.parse(file).getroot().tag.endswith('svg')
                assert all(text in file.read_text()
                           for text in [label, *labels_to_check])
            else:
                pages = PdfReader(file).pages
                assert len(pages) == 1
                content = ' '.join((pages[0].extract_text() or '').split())
                assert all(' '.join(text.split()) in content
                           for text in [label, *labels_to_check])
                ratio = float(pages[0].mediabox.width /
                              pages[0].mediabox.height)
                assert abs(ratio -
                           figure.get_figwidth() / figure.get_figheight()) < .01
            files[suffix] = hashlib.sha256(file.read_bytes()).hexdigest()
        plot = report['plots'].setdefault(kind, {})
        plot.setdefault('source', 'retained-training-run')
        plot.update(title=figure.axes[0].get_title(), sha256=files,
                    layout='passed')
        display(native, metadata=metadata, raw=True)
        plt.close(figure)
    report['plots']['gradient']['labels'] = labels
    report['plots']['cohort']['labels'] = ['Average value', 'Population share']
    (root / 'explanation-result.json').write_text(json.dumps(report, indent=2))


def exercise(root, display):
    from wheel_mlops_lifecycle import execute

    for phase in ('fit', 'render'):
        with (root / f'explanation-{phase}.log').open('w') as log:
            code = execute([
                sys.executable,
                str(Path(__file__).resolve()), phase,
                str(root)
            ], root, log, 1200)
        if code:
            print((root / f'explanation-{phase}.log').read_text()[-12000:],
                  flush=True)
            raise AssertionError(
                f'Explanation {phase} failed; inspect explanation-{phase}.log')
    report = json.loads((root / 'explanation-result.json').read_text())
    display_outputs(root, report, display)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'render'))
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    (fit if args.phase == 'fit' else render)(args.root)
