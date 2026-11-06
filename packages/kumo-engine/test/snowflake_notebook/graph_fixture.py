# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph UI assertions using An's unchanged, checksum-pinned PR172 fixture."""
import hashlib
import inspect
import json
import re
from html import unescape
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree


def retain(session, root, stage):
    uploaded = {}
    for file in sorted(Path(root).iterdir()):
        if not file.is_file():
            continue
        transfers = session.file.put(str(file), stage, auto_compress=False,
                                     overwrite=False)
        if len(transfers) != 1 or transfers[0].status != 'UPLOADED':
            raise RuntimeError('Evidence upload failed: ' + file.name)
        with file.open('rb') as stream:
            uploaded[file.name] = hashlib.file_digest(stream,
                                                      'sha256').hexdigest()
    if set(uploaded) != {
            'graph.svg', 'graph.pdf', 'graph.md', 'default.html',
            'sampled.html', 'graph-result.json'
    }:
        raise RuntimeError('Incomplete graph evidence')
    return uploaded


def exercise(root, fixture, display=None):
    import pandas as pd
    from kumo_core.sources.preview import sample_dataframe
    from pypdf import PdfReader
    from wheel_gpu_smoke import build_graph

    from kumo.engine import FileConnector

    root, fixture = Path(root), Path(fixture)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    spec = json.loads((fixture / 'graphs.json').read_text())['adtech']
    graph, snapshot = build_graph(spec, fixture / 'adtech/pr',
                                  materialize=False)
    assert snapshot is None
    tables = graph.tables
    connector = next(iter(tables.values())).connector
    assert all(table.connector is connector for table in tables.values())
    backend = connector.source_backend
    descriptor = inspect.getattr_static(FileConnector, 'source_backend')
    get_sample = type(backend).get_sample
    accesses, previews, frames = [], [], {}

    def source(instance):
        accesses.append(instance.id)
        return descriptor.__get__(instance, type(instance))

    def sample(instance, *args, **kwargs):
        previews.append(
            dict(table=args[1], limit=args[2], timeout=kwargs.get('timeout')))
        actual = get_sample(instance, *args, **kwargs)
        frames[args[1]] = sample_dataframe(actual.sample_rows)
        return actual

    expected_edges = sorted((edge['src_table'], edge['fkey'], edge['dst_table'],
                             spec['tables'][edge['dst_table']]['pkey'])
                            for edge in spec['edges'])
    report = dict(status='failed', fixture='pr172-v1/adtech/pr', schema=spec,
                  default_reads={}, opt_in_reads={}, samples=[], canaries=[],
                  exports={})
    with patch.object(FileConnector, 'source_backend',
                      property(source)), patch.object(type(backend),
                                                      'get_sample', sample):
        default = graph.visualize(show_cols=True)
        if display is not None:
            display(default)
        baseline = root / 'metadata-only'
        baseline.mkdir()
        for suffix in ('svg', 'pdf', 'md'):
            graph.visualize(baseline / ('graph.' + suffix), show_cols=True)
        report['default_reads'] = dict(backend_accesses=len(accesses),
                                       sample_calls=len(previews))
        assert not accesses and not previews, 'Default graph rendering accessed source data'
        root_svg = ElementTree.fromstring(str(default))
        nodes = [
            node for node in root_svg.iter() if node.get('data-kumo-table')
        ]
        assert {node.get('data-kumo-table') for node in nodes} == set(tables)
        assert all('Sample values:' not in node.get('data-kumo-tip', '')
                   for node in nodes)
        actual_edges = sorted(
            tuple(
                node.get(key) for key in ('data-src-table', 'data-src-column',
                                          'data-dst-table', 'data-dst-column'))
            for node in root_svg.findall(
                './/{http://www.w3.org/2000/svg}path[@class="kumo-edge"]'))
        assert actual_edges == expected_edges, 'Rendered relationships differ from An fixture'
        for name, table in spec['tables'].items():
            rows = {
                node.get('data-kumo-column'): node
                for node in nodes if node.get('data-kumo-table') == name
                and node.get('data-kumo-column')
            }
            assert set(rows) == {column['name']
                                 for column in table['cols']}, name
            if table.get('pkey'):
                assert 'Role: PK' in rows[table['pkey']].get('data-kumo-tip')
            for src, column, dst, primary in expected_edges:
                if src == name:
                    tip = rows[column].get('data-kumo-tip')
                    assert 'FK' in tip and dst + '.' + primary in tip
        sampled = graph.visualize(show_cols=True, show_samples=True)
        if display is not None:
            display(sampled)
        assert len(previews) == len(tables) and {
            call['table']
            for call in previews
        } == {table.table_name
              for table in tables.values()}
        assert all(call['limit'] == 30 and call['timeout'] > 0
                   for call in previews)
        for name, table in tables.items():
            frame = frames[table.table_name]
            assert frame is not None and len(frame) > 0, name
            columns = spec['tables'][name]['cols']
            selected = next(
                (column['name']
                 for column in columns if column['dtype'] == 'date'), None)
            if selected is None:
                selected = next(column['name'] for column in columns
                                if column['dtype'] == 'string')
            values = frame[selected].iloc[:30].dropna().iloc[:6].tolist()
            labels = [
                str(value) if isinstance(value, pd.Timestamp)
                or columns[[column['name']
                            for column in columns].index(selected)]['dtype']
                == 'date' else repr(value) for value in values
            ]
            assert len(labels) == 6 and all(
                value in unescape(sampled._repr_html_())
                for value in labels), name
            report['samples'].append(
                dict(table=name, column=selected, values=labels))
        report['opt_in_reads'] = dict(backend_accesses=len(accesses),
                                      sample_calls=len(previews),
                                      calls=list(previews))
        before_export = (len(accesses), len(previews))
        baseline_text = (baseline / 'graph.svg').read_text() + (
            baseline / 'graph.md').read_text() + '\n'.join(
                page.extract_text() or ''
                for page in PdfReader(baseline / 'graph.pdf').pages)
        report['canaries'] = sorted({
            value
            for item in report['samples']
            for value in item['values']
            if len(value) >= 8 and value not in baseline_text
        })
        assert len(report['canaries']
                   ) >= 6, 'Need distinctive existing fixture sample values'
        for suffix in ('svg', 'pdf', 'md'):
            destination = root / ('graph.' + suffix)
            graph.visualize(destination, show_cols=True, show_samples=True)
            content = destination.read_bytes()
            text = '\n'.join(page.extract_text() or '' for page in PdfReader(
                destination).pages) if suffix == 'pdf' else content.decode()
            normalized = re.sub(r'\s+', '', text)
            assert all(
                re.sub(r'\s+', '', value) not in normalized
                and value.encode() not in content for value in
                report['canaries']), 'Diagram export contains sample values'
            original = baseline / ('graph.' + suffix)
            if suffix == 'pdf':
                original_text = '\n'.join(page.extract_text() or ''
                                          for page in PdfReader(original).pages)
                assert re.sub(r'\s+', '', original_text) == normalized
            elif suffix == 'svg':
                normalize = lambda value: re.sub(r'kumo-[0-9a-f]{32}',
                                                 'kumo-ID', value)
                assert normalize(text) == normalize(original.read_text())
                assert 'Sample values:' not in text
                assert ElementTree.fromstring(
                    text).tag == '{http://www.w3.org/2000/svg}svg'
            else:
                assert text == original.read_text() and 'flowchart LR' in text
            report['exports'][suffix] = dict(
                path=destination.name,
                sha256=hashlib.sha256(content).hexdigest(),
                sample_values='absent')
        report['export_reads'] = dict(
            backend_accesses=len(accesses) - before_export[0],
            sample_calls=len(previews) - before_export[1])
    (root / 'default.html').write_text(default._repr_html_())
    (root / 'sampled.html').write_text(sampled._repr_html_())
    report['status'] = 'passed'
    (root / 'graph-result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
