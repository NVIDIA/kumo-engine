# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local Jupyter explanation checks; does not submit to Snowflake or train."""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

FIXTURE = '''
from pathlib import Path
from kumo_core.io.csv import save_csv
from kumo_core.io.json import save_json
from kumoapi.explain.gradient import GraphGradientScore, TableGradientScore
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.task import TaskType
from kumoapi.typing import Stype

run = Path('retained-run').resolve()
trial = run / 'models/users/automl/version_0'
trial.mkdir(parents=True)
save_json({'best_trainer': str(trial)}, str(trial.parent / 'best_trainer.json'))
save_json({'refit_cfg': {'refitted_trainval': False}}, str(trial / 'config.json'))
scores = GraphGradientScore(tables={
    'users': TableGradientScore('users', {'age': .75, 'zero': 0.}),
    'events': TableGradientScore('events', {'age': .25}),
})
save_csv(scores.to_pandas(), str(trial / 'explain_test/scores.csv'))
cohort = ColumnCohorts(
    stype=Stype.numerical, cohorts=['Low', 'High'], populations=[.25, .75],
    predictions={'no': [.9, .2], 'yes': [.1, .8]},
    targets={'no': [1., 0.], 'yes': [0., 1.]},
    prediction_score=.35, target_score=.5,
    table_name='users', hop=0, col_name='age')
save_json(ColumnAnalysisOutput(TaskType.MULTICLASS_CLASSIFICATION, [cohort]),
          str(trial / 'column_analysis/test/column_analysis.json'))
'''

DISPLAY = '''
%matplotlib inline
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from kumo.engine import TrainingExplanations, ExplanationUnavailableError

explanations = TrainingExplanations.load(Path('retained-run'))
assert explanations.availability['gradient'].available
assert explanations.availability['cohorts'].available
assert not explanations.availability['samples'].available
assert explanations.classes(('users', 0, 'age')) == ('no', 'yes')
gradient, axis = explanations.plot_gradient()
assert [t.get_text() for t in axis.get_yticklabels()] == [
    'users.age', 'events.age', 'users.zero']
assert [bar.get_width() for bar in axis.patches] == [.75, .25, 0.]
cohort, (prediction, population) = explanations.plot_cohort(
    ('users', 0, 'age'), class_name='yes')
assert list(prediction.lines[0].get_ydata()) == [.1, .8]
assert list(prediction.lines[1].get_ydata()) == [0., 1.]
assert [bar.get_height() for bar in population.patches] == [.25, .75]
assert prediction.get_ylabel() == 'Average value'
assert population.get_ylabel() == 'Population share'
for name, figure in [('gradient', gradient), ('cohort', cohort)]:
    display(figure)
    for suffix in ('png', 'svg', 'pdf'):
        figure.savefig(f'{name}.{suffix}')
    plt.close(figure)
try:
    TrainingExplanations.unavailable().plot_gradient()
except ExplanationUnavailableError:
    pass
else:
    raise AssertionError('Model-only load fabricated an explanation')
assert not plt.get_fignums()
'''


def png_outputs(notebook):
    return [
        output['data']['image/png'] for cell in notebook.cells
        if cell.cell_type == 'code' for output in cell.get('outputs', [])
        if output.output_type in ('display_data', 'execute_result')
        and 'image/png' in output.get('data', {})
    ]


def exercise(destination):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from PIL import Image
    from pypdf import PdfReader

    destination = Path(destination).resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Keep kernel discovery, caches and history inside this attempt.
    for variable, directory in (('JUPYTER_DATA_DIR', 'jupyter-data'),
                                ('JUPYTER_RUNTIME_DIR', 'jupyter-runtime'),
                                ('JUPYTER_CONFIG_DIR', 'jupyter-config'),
                                ('IPYTHONDIR', 'ipython'), ('MPLCONFIGDIR',
                                                            'matplotlib')):
        os.environ[variable] = str(destination / directory)
    kernel = destination / 'jupyter-data/kernels/kumo-explanation-local'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(
        json.dumps({
            'argv': [
                sys.executable, '-m', 'ipykernel_launcher', '-f',
                '{connection_file}'
            ],
            'display_name':
            'Local explanation check',
            'language':
            'python',
        }))
    retained = None
    for name, sources in [('initial', [FIXTURE, DISPLAY]),
                          ('fresh-kernel', [DISPLAY])]:
        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell(source) for source in sources])
        client = NotebookClient(
            notebook, timeout=180, kernel_name='kumo-explanation-local',
            resources={'metadata': {
                'path': str(destination)
            }})
        client.execute()  # Shuts down this kernel before saved-output checks.
        path = destination / (name + '.ipynb')
        nbformat.write(notebook, path)
        saved = nbformat.read(path, as_version=4)
        nbformat.validate(saved)
        images = png_outputs(saved)
        assert len(images) == 2, 'Expected two retained explanation displays'
        assert images == png_outputs(notebook), 'Saved MIME output changed'
        for content in images:
            with Image.open(io.BytesIO(base64.b64decode(content))) as image:
                image.verify()
        if retained is not None:
            assert images == retained, 'Fresh-kernel plots differ'
        retained = images
        html, _ = HTMLExporter().from_notebook_node(saved)
        (destination / (name + '.html')).write_text(html)

    exports = {}
    for name, label in [('gradient', 'users.age'),
                        ('cohort', 'Population share')]:
        for suffix in ('png', 'svg', 'pdf'):
            path = destination / (name + '.' + suffix)
            if suffix == 'png':
                with Image.open(path) as image:
                    image.verify()
            elif suffix == 'svg':
                assert ElementTree.parse(path).getroot().tag.endswith('svg')
                assert label in path.read_text()
            else:
                text = '\n'.join(page.extract_text() or ''
                                 for page in PdfReader(path).pages)
                assert label in text, 'PDF is missing explanation labels'
            exports[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(status='passed', profile='local-jupyter-schema-fixture',
                  training='not_run', snowflake='not_run', saved_displays=2,
                  fresh_kernel='passed', exports=exports)
    (destination /
     'explanation-result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(exercise(arguments.output_dir), indent=2))
