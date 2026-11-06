# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel cells for the shared graph scenario; CONFIG is supplied by the runner."""


def execute(config, dbutils):
    import hashlib
    import importlib.metadata
    import json
    import os
    import shutil
    import sys
    import tarfile
    import tempfile
    import zipfile
    from pathlib import Path
    ROOT = Path(tempfile.mkdtemp(prefix='egu-databricks-graph-'))
    PAYLOAD = ROOT / 'payload'
    PAYLOAD.mkdir()
    for name, asset in config['assets'].items():
        target = PAYLOAD / name
        shutil.copyfile(asset['path'], target)
        assert target.stat().st_size == asset['size']
        assert hashlib.sha256(
            target.read_bytes()).hexdigest() == asset['sha256'], name
    wheel = PAYLOAD / config['candidate']['wheel_file']
    assert importlib.metadata.version(
        'kumo-engine') == config['candidate']['expected_version']
    import diskgraph
    import diskgraph._c
    import kumo_core

    import kumo.engine as kumo
    installed = Path(
        importlib.metadata.distribution('kumo-engine').locate_file(
            '')).resolve()
    with zipfile.ZipFile(wheel) as archive:
        for module in (kumo, kumo_core, diskgraph, diskgraph._c):
            target = Path(module.__file__).resolve()
            assert target.is_relative_to(installed)
            relative = str(target.relative_to(installed))
            assert archive.read(relative) == target.read_bytes(), relative
        for name in ('graph.py', 'table.py', '_widget.py', '_visualization.py',
                     '_visualization_viewer.py', '_visualization_pdf.py',
                     '_visualization_layout.py'):
            relative = 'kumo/engine/' + name
            assert archive.read(relative) == (installed /
                                              relative).read_bytes(), relative
            assert hashlib.sha256(archive.read(
                relative)).hexdigest() == config['modules'][name], name
    FIXTURE = ROOT / 'fixture'
    FIXTURE.mkdir()
    with tarfile.open(PAYLOAD / 'pr172-inputs-v1.tar.gz') as archive:
        for member in archive.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(
                member.name).parts
            assert member.isfile() or member.isdir(), member.name
        archive.extractall(FIXTURE, filter='data')
    sys.path.insert(0, str(PAYLOAD))
    print('KUMO_DATABRICKS_SETUP=' + config['attempt'], flush=True)

    from uuid import uuid4

    from graph_fixture import exercise
    from IPython.display import display
    execution_id = uuid4().hex[:12]
    print('KUMO_DATABRICKS_GRAPH_STARTED=' + execution_id, flush=True)
    context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = context.notebookPath().get()
    assert notebook_path == config['notebook_path'], notebook_path
    runtime_version = os.environ.get('DATABRICKS_RUNTIME_VERSION')
    assert runtime_version, 'Expected Databricks notebook runtime'
    assert os.environ.get('KUMO_DATABRICKS_TEST_PAT'
                          ) is None, 'Runner PAT must not enter the notebook'
    report = exercise(ROOT / 'results', FIXTURE, display=display)
    report.update(
        execution_id=execution_id, attempt=config['attempt'],
        wheel_sha256=config['candidate']['wheel_sha256'],
        source_commit=config['source_commit'],
        runtime=dict(platform='databricks', notebook_path=notebook_path,
                     version=runtime_version, python=sys.version,
                     workspace=True, display_path='IPython.display.display'),
        payload=config['assets'])
    (ROOT / 'results' / 'graph-result.json').write_text(
        json.dumps(report, indent=2))
    destination = Path(config['output'])
    destination.mkdir(parents=True, exist_ok=False)
    for file in (ROOT / 'results').iterdir():
        if file.is_file():
            shutil.copyfile(file, destination / file.name)
    receipt = dict(
        status='passed', attempt=config['attempt'], execution_id=execution_id,
        output=config['output'], files={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in destination.iterdir() if f.is_file()
        })
    print('KUMO_DATABRICKS_GRAPH_RECEIPT=' + json.dumps(receipt), flush=True)
