# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A run's card: a short Markdown summary of a fit or prediction run,
from its metrics, tags, params, notes and brief tables, for any tracker."""

import re
from typing import Any, List, Mapping, Optional, Sequence

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.summaries import Note, Summary, Table

HEADLINE_CHARS = 140
_TITLES = {'data/splits': 'Training data', 'pred/summary': 'Scores'}


def entity(query: Optional[str]) -> Optional[str]:
    r"""The entity table a predictive query predicts for."""
    found = re.search(r'FOR\s+EACH\s+(\w+)', query or '', re.IGNORECASE)
    return found and found.group(1)


def headline(kind: str, metrics: Mapping[str, float], tags: Mapping[str, str],
             params: Mapping[str, str], query: Optional[str] = None) -> str:
    r"""Line 1: the run's status and watch numbers, plain text, at most
    `HEADLINE_CHARS` characters, zero counts left out."""
    parts = [f"failed: {tags['kumo.error']}"] if 'kumo.error' in tags else []
    if kind == 'fit':
        tune = tags.get('kumo.tune_metric')
        name = 'holdout_loss' if tune == 'loss' else _key(tune or '')
        if name in metrics:
            parts.append(f'{name} {number(metrics[name])}')
    else:
        realized = sorted(name for name in metrics
                          if name.startswith('realized/')
                          and name != 'realized/labelled_share')
        if realized:
            parts.append(f"realized {realized[0].partition('/')[2]} "
                         f'{number(metrics[realized[0]])}')
        for key, label in (('drift/psi', 'psi'), ('drift/psi_approx', 'psi≈')):
            if key in metrics:
                parts.append(f'{label} {number(metrics[key])} vs training')
                break
        if 'pred/rows' in metrics:
            as_of = tags.get('kumo.as_of')
            parts.append(f"{number(metrics['pred/rows'])} rows"
                         + (f' as of {as_of}' if as_of else ''))
        if metrics.get('pred/skipped_share'):
            parts.append(f"{metrics['pred/skipped_share']:.1%} skipped")
    problems = int(tags.get('kumo.data_warnings') or 0)
    if problems:
        parts.append(f"{problems} data problem{'s' * (problems > 1)}")
    if tags.get('kumo.model_name') and tags.get('kumo.model_version'):
        parts.append(f"v{tags['kumo.model_version']} {tags['kumo.model_name']}")
    if kind == 'fit':
        trials = int(params.get('training_job.num_experiments') or 1)
        if trials > 1:
            parts.append(f'best of {trials} trials')
        task = tags.get('kumo.task_type', '').replace('_', ' ')
        of = entity(query)
        if task:
            parts.append(f'{task} of {of}' if of else task)
    line = ' · '.join(parts) or kind
    return (line if len(line) <= HEADLINE_CHARS else
            line[:HEADLINE_CHARS - 1] + '…')


def version_headline(metrics: Mapping[str, float],
                     tune: Optional[str]) -> str:
    r"""A registered version's metrics line: the tuned metric, then the
    next three holdout metrics."""
    tuned = 'holdout_loss' if tune == 'loss' else tune
    names = [name for name in metrics if '/' not in name]
    names.sort(key=lambda name: name != tuned)
    return ' · '.join(f'{_key(name)} {number(metrics[name])}'
                      for name in names[:4])


def model_line(name: Optional[str], version: Optional[str],
               bundle_id: Optional[str], fit_run: Optional[str] = None,
               link: Optional[str] = None) -> str:
    r"""A prediction's line 2: the model that scored and the fit run that
    trained it, linked when ``link`` is given."""
    model = (f'`{name}` v{version}' if name and version else
             f'bundle `{(bundle_id or "")[:8]}`')
    if fit_run is None:
        return f'Model: {model}.'
    fit = f'[{fit_run}]({link})' if link else f'`{fit_run}`'
    return f'Model: {model}, trained by {fit}.'


def body(intro: Optional[str], summaries: Sequence[Summary],
         files: Sequence[str]) -> List[str]:
    r"""The card after line 1: ``intro`` (line 2), problems, brief tables,
    notes, where things are, and the run's files."""
    notes = {
        section: [s.text for s in summaries
                  if isinstance(s, Note) and s.section == section]
        for section in ('problems', 'notes', 'locations')
    }
    lines = [] if intro is None else [intro]
    if notes['problems']:
        lines += ['', f"**Problems:** {' · '.join(notes['problems'])}"]
    for table in summaries:
        if isinstance(table, Table):
            lines += ['', table_text(table)]
    if notes['notes']:
        lines += ['', f"**Notes:** {' · '.join(notes['notes'])}"]
    if notes['locations']:
        lines += ['', '**Where things are**',
                  *(f'- {text}' for text in notes['locations'])]
    return [*lines, '', files_line(files)]


def block(summaries: Sequence[Summary]) -> str:
    r"""A part of the card a later step adds, such as drift: its notes,
    then its brief tables."""
    return '\n'.join([
        *(s.text for s in summaries if isinstance(s, Note)),
        *(table_text(s) for s in summaries if isinstance(s, Table))])


def files_line(files: Sequence[str]) -> str:
    listed = ', '.join(f'`{name}`' for name in files
                       if name != PROVENANCE_FILE)
    return (f'Full lineage: `{PROVENANCE_FILE}`.'
            + (f' Tables: {listed}.' if listed else ''))


def table_text(table: Table) -> str:
    r"""``**Title:**`` and the table: one line for one or two rows, else a
    Markdown table."""
    title = f'**{_title(table.name)}:**'
    if len(table.rows) <= 2:
        return f"{title} {'; '.join(_inline(table.columns, row) for row in table.rows)}"
    numeric = [all(isinstance(row[i], (int, float)) or row[i] is None
                   for row in table.rows) for i in range(len(table.columns))]
    return '\n'.join([
        title, '', _row(table.columns),
        _row(['---:' if right else '---' for right in numeric]),
        *(_row([_cell(value) for value in row]) for row in table.rows)
    ])


def number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return f'{int(value):,}'
    return f'{value:,.0f}' if abs(value) >= 1000 else f'{value:.3g}'


def _inline(columns: Sequence[str], row: Sequence[Any]) -> str:
    pairs = [(column, value) for column, value in zip(columns, row)
             if value is not None]
    label = ''
    if pairs and isinstance(pairs[0][1], str):
        label, pairs = f'{pairs[0][1]}: ', pairs[1:]
    return label + ' · '.join(f'{column} {number(value)}'
                              for column, value in pairs)


def _title(name: str) -> str:
    title = _TITLES.get(name, name.rpartition('/')[2].replace('_', ' '))
    return title[:1].upper() + title[1:]


def _row(cells: Sequence[str]) -> str:
    return '| ' + ' | '.join(cells) + ' |'


def _cell(value: Any) -> str:
    return '' if value is None else number(value).replace('|', '\\|')


def _key(name: str) -> str:
    return name.replace('@', '_at_')
