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

PLOTS = ('gradient', 'cohort', 'cohort-class-0', 'cohort-class-1')
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
        for task in ('binary', 'multiclass'):
            work = scratch / task
            work.mkdir()
            kumo.configure(scratch_dir=work / 'scratch')
            training_graph = graph(work, '2024-03-01',
                                   categorical=task == 'multiclass')
            snapshot = training_graph.snapshot()
            query = (QUERY if task == 'binary' else
                     'PREDICT FIRST(events.KIND, 0, 7, days) FOR EACH users.ID')
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
                         'multiclass'):
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
    multiclass = retained / 'multiclass'
    class_state = json.loads(
        (multiclass / 'fit.json').read_text())['result']['untracked']
    classified = kumo.TrainingExplanations.load(multiclass / 'training-run')
    selected(classified, class_state['explanation_trial'])
    assert kumo.Trainer.load_from_connector(
        kumo.FileConnector(root_dir=str(multiclass)), path='untracked-model',
        accelerator='cpu').result.id == class_state['bundle']
    classes = classified.classes(COHORT_KEY)
    assert set(classes) == {'morning', 'afternoon', 'evening'}, classes
    report = dict(
        status='passed', fresh_process='passed', refit=proof,
        bundle=state['bundle'], plots={},
        multiclass=dict(status='passed', fresh_process='passed',
                        bundle=class_state['bundle'], classes=classes))
    (root / 'explanation-result.json').write_text(json.dumps(report, indent=2))
    return report


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
    classified = kumo.TrainingExplanations.load(
        root / 'reopened-run/multiclass/training-run')
    for method in (classified.cohort, classified.plot_cohort):
        expect(ValueError, lambda: method(COHORT_KEY), 'class')
        expect(KeyError, lambda: method(COHORT_KEY, class_name='missing-class'),
               'missing-class')
    selected_classes = sorted(classified.classes(COHORT_KEY))[:2]
    class_predictions = []
    for index, class_name in enumerate(selected_classes):
        recorded = classified.cohort(COHORT_KEY, class_name=class_name)
        figure, (class_axis,
                 shares) = classified.plot_cohort(COHORT_KEY,
                                                  class_name=class_name)
        assert list(
            class_axis.lines[0].get_ydata()) == recorded.predictions[class_name]
        assert list(
            class_axis.lines[1].get_ydata()) == recorded.targets[class_name]
        assert [bar.get_height()
                for bar in shares.patches] == recorded.populations
        assert class_axis.get_title().endswith(' — ' + class_name)
        kind = f'cohort-class-{index}'
        report['plots'][kind] = dict(
            class_name=class_name, title=class_axis.get_title(),
            labels=[class_name, 'Average value', 'Population share'])
        class_predictions.append(recorded.predictions[class_name])
        figures.append((kind, figure, class_name))
    assert class_predictions[0] != class_predictions[
        1], 'Class selection must change predictions'
    report['multiclass']['selected_classes'] = selected_classes
    report['multiclass']['invalid_selection'] = 'passed'
    for kind, figure, label in figures:
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
                assert label in file.read_text()
            else:
                assert label in '\n'.join(page.extract_text() or ''
                                          for page in PdfReader(file).pages)
            files[suffix] = hashlib.sha256(file.read_bytes()).hexdigest()
        plot = report['plots'].setdefault(kind, {})
        plot.update(title=figure.axes[0].get_title(), sha256=files)
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
