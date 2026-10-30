# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A run's card: a short Markdown summary of a fit or prediction run,
from its metrics, tags, params, notes and brief tables, for any tracker."""

import math
import re
from typing import Any, List, Mapping, Optional, Sequence
from urllib.parse import quote

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.score_drift import psi_shift
from kumo_core.summaries import Note, Summary, Table

HEADLINE_CHARS = 140
NOTES_LINE = ("*kumo rewrites everything above this line; editing it stops "
              "kumo's updates.*")
TAIL = ('', NOTES_LINE, '', '---')
_TITLES = {'data/splits': 'Training data', 'pred/summary': 'Scores',
           'holdout/metrics': 'Holdout metrics', 'drift/score': 'Score drift',
           'realized/metrics': 'Realized vs holdout'}
_MEANINGS = {'ap': 'average precision',
             'auprc': 'area under the precision-recall curve, trapezoidal',
             'loss': 'the training objective',
             'logloss': 'cross-entropy of the scores', 'bias': 'pred − actual'}
_LEADS = {'prediction': 'Scores of', 'drift': 'Drift check of the scores of',
          'evaluate': 'Realized quality of the scores of'}
_ARTIFACTS = {'model': 'model bundle', 'prediction': 'prediction folder',
              'prediction_output': 'output'}
_NAMED = ('source_table', 'prediction_output')
_CONSOLES = {'s3': 'https://s3.console.aws.amazon.com/s3/buckets/{}?prefix={}/',
             'gs': 'https://console.cloud.google.com/storage/browser/{}/{}'}


def entity(query: Optional[str]) -> Optional[str]:
    r"""The entity table a predictive query predicts for."""
    found = re.search(r'FOR\s+EACH\s+(\w+)', query or '', re.IGNORECASE)
    return found and found.group(1)


def heading(kind: str, metrics: Mapping[str, float],
            tags: Mapping[str, str]) -> List[str]:
    r"""The card's heading: the run's status and watch numbers, plain text
    of at most `HEADLINE_CHARS` characters, zero counts left out."""
    parts = [f"failed: {tags['kumo.error']}"] if 'kumo.error' in tags else []
    if kind == 'fit':
        tune = _key(tags.get('kumo.tune_metric') or '')
        if f'test/{tune}' in metrics:
            parts.append(f"holdout {tune} {number(metrics[f'test/{tune}'])}")
    else:
        tuned = f"realized/{_key(tags.get('kumo.tune_metric') or '')}"
        realized = sorted((name for name in metrics
                           if name.startswith('realized/')
                           and name != 'realized/labelled_share'),
                          key=lambda name: (name != tuned, name))
        if realized:
            parts.append(f"realized {realized[0].partition('/')[2]} "
                         f'{number(metrics[realized[0]])}')
        if 'drift/psi' in metrics:
            psi = metrics['drift/psi']
            link = tags.get('kumo.task_type', '').endswith('link_prediction')
            parts.append(f'psi {number(psi)} vs holdout (' + (
                'link prediction: compare with the previous run' if link
                else f'{psi_shift(psi)} shift') + ')')
        as_of = tags.get('kumo.as_of')
        if 'pred/rows' in metrics:
            parts.append(f"{number(metrics['pred/rows'])} "
                         f"{tags.get('kumo.entity') or 'rows'}"
                         + (f' as of {as_of}' if as_of else ''))
        elif as_of:
            parts.append(f'as of {as_of}')
        if tags.get('kumo.labels_mature') and not realized:
            parts.append(f"labels mature {tags['kumo.labels_mature']}")
        if metrics.get('pred/skipped_share'):
            parts.append(f"{metrics['pred/skipped_share']:.1%} skipped")
    problems = int(tags.get('kumo.data_warnings') or 0)
    if problems:
        parts.append(f"{problems} data problem{'s' * (problems > 1)}")
    line = ' '.join(', '.join(parts).split()) or kind
    return [line if len(line) <= HEADLINE_CHARS else
            line[:HEADLINE_CHARS - 1] + '…', '']


def intro(kind: str, tags: Mapping[str, str], params: Mapping[str, str],
          query: Optional[str], fit_run: Optional[str] = None,
          link: Optional[str] = None) -> str:
    r"""The sentence under the heading, and a fit's ``query``."""
    if kind != 'fit':
        return model_line(tags, fit_run, link)
    return fit_line(tags, params, query) + (f'\n\n{query}' if query else '')


def fit_line(tags: Mapping[str, str], params: Mapping[str, str],
             query: Optional[str]) -> str:
    r"""A fit's sentence: the model it registered, its task, and the trial
    and epoch its curves and holdout metrics come from."""
    task = tags.get('kumo.task_type', '').replace('_', ' ')
    of = entity(query)
    line = 'Fit' + (f' of {_model(tags)}' if _model(tags) else '') + (
        f" for {task}{f' of {of}' if of else ''}" if task else '')
    tune = f"val {_key(tags.get('kumo.tune_metric') or 'loss')}"
    trials = int(params.get('training_job.num_experiments') or 1)
    trial = tags.get('kumo.best_trial') if trials > 1 else None
    epoch = tags.get('kumo.best_epoch')
    if trial or epoch:
        what = 'curves and holdout metrics' if trial else 'holdout metrics'
        at = ', '.join([*([f'trial {trial} of {trials}'] if trial else []),
                        *([f'epoch {epoch}'] if epoch else [])])
        line += f'; its {what} are from {at}, the best by {tune}'
    return line + '.'


def holdout(kind: str, metrics: Mapping[str, float],
            tags: Mapping[str, str]) -> List[Summary]:
    r"""A fit's brief table ``holdout/metrics``: its holdout metrics, the
    tuned one first, each with what it means where its name is unclear;
    of the metrics at k, only those at the tuned metric's k, and a note
    pointing to the others."""
    if kind != 'fit':
        return []
    tuned = f"test/{_key(tags.get('kumo.tune_metric') or '')}"
    tested = {_key(name): value for name, value in metrics.items()
              if name.startswith('test/')}
    at = _at(tuned)
    names = sorted((name for name in tested if _at(name) in (None, at)),
                   key=lambda name: (name != tuned, name))
    if not names:
        return []
    others = len(names) < len(tested)
    return [Table('holdout/metrics', ('metric', 'value'), tuple(
        (_meaning(name), tested[name]) for name in names), brief=True),
        *([Note('notes', 'Holdout metrics at other k: `holdout/by_k.json`.')]
          if others else [])]


def version_headline(metrics: Mapping[str, float],
                     tune: Optional[str]) -> str:
    r"""A registered version's metrics line: the tuned metric, then the
    next three holdout metrics."""
    tuned = f"test/{_key(tune or '')}"
    names = [name for name in metrics if name.startswith('test/')]
    names.sort(key=lambda name: name != tuned)
    return ' · '.join(f'{_key(name)} {number(metrics[name])}'
                      for name in names[:4])


def model_line(tags: Mapping[str, str], fit_run: Optional[str] = None,
               link: Optional[str] = None) -> str:
    r"""A prediction's or monitoring run's sentence: the model that scored
    and the fit run that trained it, linked when ``link`` is given."""
    lead = _LEADS.get(tags.get('kumo.monitor') or tags.get('kumo.kind', ''),
                      _LEADS['prediction'])
    model = _model(tags) or f"bundle `{tags.get('kumo.bundle_id', '')[:8]}`"
    if fit_run is None:
        return f'{lead} {model}.'
    fit = f'[{fit_run}]({link})' if link else f'`{fit_run}`'
    return f'{lead} {model}, trained by {fit}.'


def body(intro: Optional[str], summaries: Sequence[Summary],
         files: Sequence[str]) -> List[str]:
    r"""The card after its heading: ``intro``, problems, brief tables,
    notes, where things are, and the run's files."""
    lines = [] if intro is None else ['', intro]
    problems = _notes(summaries, 'problems')
    if problems:
        lines += ['', '**Problems**', '', *problems]
    tables = [s for s in summaries if isinstance(s, Table)]
    for table in tables:
        if table.name != 'locations':
            lines += ['', table_text(table)]
    notes = _notes(summaries, 'notes')
    if notes:
        lines += ['', '**Notes**', '', *notes]
    for table in tables:
        if table.name == 'locations':
            lines += ['', _where(table)]
    return [*lines, '', files_line(files)]


def block(summaries: Sequence[Summary]) -> str:
    r"""A part of the card a later step adds, such as drift: its brief
    tables, then its notes."""
    return '\n\n'.join([
        *(table_text(s) for s in summaries if isinstance(s, Table)),
        *(['\n'.join(_notes(summaries, 'notes'))]
          if _notes(summaries, 'notes') else [])])


def files_line(files: Sequence[str]) -> str:
    listed = ', '.join(f'`{name}`' for name in files
                       if name != PROVENANCE_FILE)
    return (f'Full lineage: `{PROVENANCE_FILE}`.'
            + (f' Tables: {listed}.' if listed else ''))


def table_text(table: Table) -> str:
    r"""``**Title**`` and the table in Markdown."""
    numeric = [all(isinstance(row[i], (int, float)) or row[i] is None
                   for row in table.rows) for i in range(len(table.columns))]
    return '\n'.join([
        f'**{_title(table.name)}**', '', _row(table.columns),
        _row(['---:' if right else '---' for right in numeric]),
        *(_row([_cell(value) for value in row]) for row in table.rows)
    ])


def number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return f'{int(value):,}'
    if not math.isfinite(value) or abs(value) >= 1000:
        return f'{value:,.0f}'
    digits = 2 - math.floor(math.log10(abs(value)))
    return f'{value:.{digits}f}'.rstrip('0').rstrip('.')


def _where(table: Table) -> str:
    r"""The ``locations`` table as "Where things are": each artifact by its
    kind and short id, and its full location, linked to its console."""
    return '\n'.join([
        '**Where things are**', '', '| Artifact | Location |', '| --- | --- |',
        *(_row([_artifact(kind, str(id)), _location(str(location))])
          for kind, id, location in table.rows)])


def _artifact(kind: str, id: str) -> str:
    label = _ARTIFACTS.get(kind, kind.replace('_', ' '))
    return f"{label} `{id if kind in _NAMED else id[:8]}`"


def _location(location: str) -> str:
    scheme, _, path = location.partition('://')
    bucket, _, key = path.partition('/')
    text = f"`{location.replace('|', '%7C')}`"
    console = _CONSOLES.get(scheme)
    return (f'[{text}]({console.format(bucket, quote(key.rstrip("/")))})'
            if console and bucket else text)


def _notes(summaries: Sequence[Summary], section: str) -> List[str]:
    return [f'- {s.text}' for s in summaries
            if isinstance(s, Note) and s.section == section]


def _model(tags: Mapping[str, str]) -> Optional[str]:
    name, version = tags.get('kumo.model_name'), tags.get('kumo.model_version')
    return f'`{name}` v{version}' if name and version else None


def _meaning(name: str) -> str:
    meaning = _MEANINGS.get(name.partition('/')[2])
    return f'{name} ({meaning})' if meaning else name


def _title(name: str) -> str:
    title = _TITLES.get(name, name.rpartition('/')[2].replace('_', ' '))
    return title[:1].upper() + title[1:]


def _row(cells: Sequence[str]) -> str:
    return '| ' + ' | '.join(cells) + ' |'


def _cell(value: Any) -> str:
    return '' if value is None else number(value).replace('|', '\\|')


def _at(name: str) -> Optional[str]:
    found = re.search(r'_at_(\d+)$', name)
    return found and found.group(1)


def _key(name: str) -> str:
    return name.replace('@', '_at_')
