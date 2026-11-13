# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real refit explanations rendered inside the managed graph notebook."""
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

EXPORTS = tuple(f'explanation-{kind}.{suffix}'
                for kind in ('gradient', 'cohort')
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
        kumo.configure(scratch_dir=scratch / 'scratch')
        training_graph = graph(scratch, '2024-03-01')
        snapshot = training_graph.snapshot()
        table = kumo.PredictiveQuery(
            graph=training_graph, query=QUERY).generate_training_table(snapshot)
        plan = small_plan(1, TRIALS)
        plan.training_job.disable_explain = False
        plan.training_job.refit_trainval = True
        result = kumo.Trainer(plan,
                              accelerator='cpu').fit(table, snapshot=snapshot)
        explained = result.explanations
        assert result.manifest.fit_mode == 'refit_trainval'
        assert all(
            a.available and a.generation_failure is None
            for a in explained.availability.values()), explained.availability
        trial = selected(
            explained,
            os.path.relpath(result.holdout_path.parent, explained.run_root))
        retained = scratch / 'retained'
        retained.mkdir()
        shutil.copytree(explained.run_root, retained / 'training-run')
        result.export(
            kumo.ModelOutputConfig(kumo.FileConnector(root_dir=str(retained)),
                                   path='untracked-model'))
        state = dict(
            bundle=result.id, explanation_trial=trial,
            gradient=explained.gradient_scores().to_pandas().to_dict('records'))
        (retained / 'fit.json').write_text(
            json.dumps({'result': {
                'untracked': state
            }}))
        with tarfile.open(root / 'explanation-run.tar.gz', 'w:gz') as archive:
            for name in ('training-run', 'untracked-model', 'fit.json'):
                archive.add(retained / name, arcname=name)
    # The fit process exits with its scratch and original run removed.


def render(root):
    import matplotlib.pyplot as plt
    from wheel_mlops_lifecycle import explain

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
    report = dict(status='passed', fresh_process='passed', refit=proof,
                  bundle=state['bundle'], plots={})
    (root / 'explanation-result.json').write_text(json.dumps(report, indent=2))
    return report


def display_outputs(root, report, display):
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    from PIL import Image
    from pypdf import PdfReader

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
    for kind, figure, label in (('gradient', gradient,
                                 'Training column importance'),
                                ('cohort', cohort, 'Population share')):
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
        report['plots'][kind] = dict(
            title=axis.get_title()
            if kind == 'gradient' else prediction.get_title(), sha256=files)
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
