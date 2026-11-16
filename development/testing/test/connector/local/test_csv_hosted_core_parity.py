# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare hosted CSV metadata and exports against the shared file reader."""

import os
import random
from unittest.mock import patch

import pytest

from kumo.connector.csv_connector import CSVConnector

#: Value families chosen for where the two implementations could disagree.
VOCABULARY = {
    'integer': ['1', '2', '3'],
    'negative': ['-1', '-2', '-3'],
    'signed': ['+1', '+2', '+3'],
    'grouped': ['1_000', '2_000', '3_000'],
    'floating': ['1.5', '2.5', '3.5'],
    'whole_float': ['1.0', '2.0', '3.0'],
    'exponent': ['1e3', '2e3', '3e3'],
    'leading_zero': ['01234', '00987', '01111'],
    'boolean': ['true', 'false', 'true'],
    'boolean_upper': ['TRUE', 'FALSE', 'TRUE'],
    'boolean_title': ['True', 'False', 'True'],
    'text': ['alpha', 'beta', 'gamma'],
    'hex': ['0x10', '0x20', '0x30'],
    'iso_date': ['2026-01-01', '2026-01-02', '2026-01-03'],
    'timestamp':
    ['2026-01-01 10:00:00', '2026-01-02 11:00:00', '2026-01-03 12:00:00'],
    'na_tokens': ['NA', 'N/A', 'null'],
    'empty': ['', '', ''],
    'mixed_number_text': ['1', 'ABC', '3'],
    'mixed_int_float': ['1', '2.5', '3'],
    'wide_integer': ['9007199254740993', '9007199254740994', '1'],
    'currency': ['$1', '$2', '$3'],
    'padded': [' 1', ' 2', ' 3'],
    'quoted': ['"1"', '"2"', '"3"'],
    'unicode': ['café', 'naïve', 'über'],
}

#: Mixtures *within* one column: every rule that reads a column as a whole
#: only misfires when two families share one.
MIXTURES = {
    'na_and_boolean': ['true', 'NA', 'false'],
    'na_and_integer': ['1', 'NA', '3'],
    'na_and_floating': ['1.5', 'NA', '2.5'],
    'na_and_text': ['alpha', 'NA', 'gamma'],
    'empty_and_boolean': ['true', '', 'false'],
    'empty_and_integer': ['1', '', '3'],
    'empty_and_text': ['alpha', '', 'gamma'],
    'integer_and_whole_float': ['1', '10.0', '3'],
    'integer_and_wide': ['1', '9007199254740993', '3'],
    'floating_and_infinity': ['1.5', 'inf', '-inf'],
    'floating_and_nan': ['1.5', 'nan', '2.5'],
    'boolean_and_text': ['true', 'maybe', 'false'],
    'unknown_na_and_integer': ['1', '<NA>', '3'],
    'unknown_na_and_text': ['alpha', 'None', 'gamma'],
}

PLATFORM_CANNOT_STAGE = {'unknown_na_and_integer'}


def _write(directory, columns):
    os.makedirs(directory, exist_ok=True)
    names = list(columns)
    height = max(len(values) for values in columns.values())
    rows = [
        ','.join(columns[name][index % len(columns[name])] for name in names)
        for index in range(height)
    ]
    body = ','.join(names) + '\n' + '\n'.join(rows) + '\n'
    with open(os.path.join(directory, 't.csv'), 'w', encoding='utf-8') as out:
        out.write(body)
    return body


#: Row counts that straddle where each side stops sampling.
SAMPLE_BOUNDARIES = [1_200, 100_100]


def _corpus():
    """One table per family, one holding every family at once, and seeded
    mixtures -- so a rule that only misfires beside another is still reached.
    """
    families = dict(VOCABULARY, **MIXTURES)
    cases = [(name, {name: values}) for name, values in families.items()]
    cases.append(('all', dict(families)))
    rng = random.Random(20260901)
    names = sorted(families)
    for index in range(25):
        picked = rng.sample(names, 4)
        cases.append((f'mixed{index}', {
            name: families[name]
            for name in picked
        }))
    return cases


def _long_corpus():
    """Tables long enough to cross a sampling boundary, with the value that
    contradicts the sample placed past it."""
    cases = []
    for rows in SAMPLE_BOUNDARIES:
        cases.append((f'late_text_{rows}', {'a': ['1'] * rows + ['ABC', '2']}))
        cases.append((f'late_float_{rows}', {'a': ['1'] * rows + ['2.5', '3']}))
        cases.append((f'late_whole_float_{rows}', {
            'a': ['1'] * rows + ['10.0', '2']
        }))
        cases.append((f'late_infinity_{rows}', {
            'a': ['1.5'] * rows + ['inf', '2.5']
        }))
        cases.append((f'late_na_{rows}', {'a': ['1'] * rows + ['NA', '2']}))
    return cases


def _hosted(root, table):
    connector = CSVConnector(name='parity', root_dir=root)
    metadata = connector._get_table_cols_metadata(table, skip_cache=True)
    if metadata is None:
        return 'refused'
    return {column.name: column.dtype for column in metadata.cols}


def _core(root, table):
    from kumo_core.sources.object_store import ObjectStoreSourceReader
    from pyarrow.fs import LocalFileSystem
    return ObjectStoreSourceReader(LocalFileSystem(), root=root).dtypes(table)


def _outcome(call, *args):
    try:
        return call(*args)
    except Exception:
        return 'refused'


REFUSED = 'refused'


def _hosted_staged(root, table, dtypes, tmp_path):
    import pyarrow.parquet as pq
    from kumo_core.types.schema import IngestedTableFingerprint

    connector = CSVConnector(name='parity', root_dir=root)
    fingerprint = IngestedTableFingerprint(connector_name='parity',
                                           source_name=table, cols=dict(dtypes),
                                           pkey=None, time_col=None,
                                           end_time_col=None)
    destination = str(tmp_path / 'hosted')
    with patch('kumo.data.utils.get_materialize_path',
               return_value=destination):
        connector.export_table(table, fingerprint, 'version')
    written = pq.read_table(destination)
    return {
        name: written.column(name).to_pylist()
        for name in written.schema.names
    }


def _core_staged(root, table, dtypes, tmp_path):
    import pyarrow.parquet as pq
    from kumo_core.sources.object_store import ObjectStoreSourceReader
    from kumo_core.sources.types import ReadMode
    from pyarrow.fs import LocalFileSystem

    reader = ObjectStoreSourceReader(LocalFileSystem(), root=root)
    staged = reader.prepare(table, str(tmp_path / 'core'), columns=dict(dtypes),
                            mode=ReadMode.STREAMED)
    written = pq.read_table(sorted(staged.paths))
    return {
        name: written.column(name).to_pylist()
        for name in written.schema.names
    }


def _comparable(values, dtype):
    import math

    out = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out.append(None)
        elif (dtype.is_int() and isinstance(value, float)
              and math.isfinite(value)):
            out.append(int(round(value)))
        else:
            out.append(value)
    return out


def _assert_staged_alike(label, columns, dtypes, core, hosted):
    for column, dtype in dtypes.items():
        left = _comparable(core[column], dtype)
        right = _comparable(hosted[column], dtype)
        assert left == right, (
            f'{label}.{column} ({dtype}): core staged {core[column]}, '
            f'platform staged {hosted[column]}')


def _staging_pair(root, table, dtypes, tmp_path):
    hosted = _outcome(_hosted_staged, root, table, dtypes, tmp_path)
    core = _outcome(_core_staged, root, table, dtypes, tmp_path)
    return hosted, core


@pytest.mark.parametrize('name,columns', _corpus(),
                         ids=[case[0] for case in _corpus()])
def test_the_two_readers_type_a_csv_the_same_way(name, columns, tmp_path):
    root = str(tmp_path / name)
    _write(os.path.join(root, name), columns)

    hosted = _outcome(_hosted, root, name)
    core = _outcome(_core, root, name)

    if hosted == REFUSED or core == REFUSED:
        assert hosted == core == REFUSED, (
            f'{name}: one side refused and the other did not -- '
            f'hosted={hosted}, core={core}')
        return
    assert core == hosted, f'{name}: hosted={hosted} core={core}'


@pytest.mark.parametrize('name,columns', _corpus(),
                         ids=[case[0] for case in _corpus()])
def test_the_two_readers_stage_the_same_values(name, columns, tmp_path):
    """Agreeing on the type is not agreeing on the data: ``1_000`` is an
    integer to both and the value 1000 only if the text is rewritten first.
    """
    root = str(tmp_path / name)
    _write(os.path.join(root, name), columns)

    dtypes = _outcome(_hosted, root, name)
    if dtypes == REFUSED:
        assert _outcome(_core, root, name) == REFUSED
        return

    hosted, core = _staging_pair(root, name, dtypes, tmp_path)
    refused = set(columns) & PLATFORM_CANNOT_STAGE
    if refused:
        assert hosted == REFUSED, (
            f'{name}: the platform is now able to stage {sorted(refused)}; '
            f'PLATFORM_CANNOT_STAGE is stale and the comparison it suppresses '
            f'should be reinstated')
        assert core == REFUSED, (name, core)
        return

    assert hosted != REFUSED, f'{name}: the platform refused to stage it'
    assert core != REFUSED, f'{name}: this reader refused to stage it'
    _assert_staged_alike(name, columns, dtypes, core, hosted)


@pytest.mark.parametrize('name,columns', _long_corpus(),
                         ids=[case[0] for case in _long_corpus()])
def test_the_two_readers_agree_past_a_sampling_boundary(name, columns,
                                                        tmp_path):
    """Each side stops sampling somewhere, and a table shorter than both
    boundaries cannot tell whether they stop in the same place. The staged
    values are compared as well as the type, because the row that contradicts
    the sample is invisible to both catalogs and only appears in the data.
    """
    root = str(tmp_path / name)
    _write(os.path.join(root, name), columns)
    hosted = _outcome(_hosted, root, name)
    core = _outcome(_core, root, name)
    assert core == hosted, f'{name}: hosted={hosted} core={core}'
    if hosted == REFUSED:
        return

    staged_hosted, staged_core = _staging_pair(root, name, hosted, tmp_path)
    if staged_hosted == REFUSED:
        assert staged_core == REFUSED, (
            f'{name}: the platform refuses to stage this and core read '
            f'{staged_core}')
        return
    _assert_staged_alike(name, columns, hosted, staged_core, staged_hosted)


@pytest.mark.parametrize('dtype_name', ['date', 'time', 'timedelta'])
def test_a_declared_type_with_no_arrow_spelling_stages_alike(
        dtype_name, tmp_path):
    """A graph may declare a type CSV inference never produces. The hosted
    export has no Arrow spelling for these either, and leaves the column to
    Arrow's own inference rather than to text.
    """
    from kumoapi.typing import Dtype

    root = str(tmp_path / dtype_name)
    _write(os.path.join(root, dtype_name), {'a': VOCABULARY['timestamp']})
    declared = {'a': Dtype(dtype_name)}

    hosted, core = _staging_pair(root, dtype_name, declared, tmp_path)
    assert hosted != REFUSED and core != REFUSED
    assert core['a'] == hosted['a']


def test_a_ragged_row_is_refused_by_both_before_anything_is_staged(tmp_path):
    """The platform's metadata reader is pandas with ``on_bad_lines='skip'``
    and types the table; its export is Arrow, which raises. This reader refuses
    one step earlier, at the catalog. Neither backend stages the file.
    """
    root = str(tmp_path / 'ragged')
    rows = ['a,b'] + [f'{index},{index}' for index in range(20_000)]
    rows.insert(15_000, '1,2,3')
    os.makedirs(os.path.join(root, 'ragged'), exist_ok=True)
    with open(os.path.join(root, 'ragged', 't.csv'), 'w',
              encoding='utf-8') as out:
        out.write('\n'.join(rows) + '\n')

    hosted = _outcome(_hosted, root, 'ragged')
    assert hosted != REFUSED, 'the platform still types a file it will not read'
    assert _hosted_staged_or_refused(root, 'ragged', hosted,
                                     tmp_path) == REFUSED
    assert _outcome(_core, root, 'ragged') == REFUSED


def _hosted_staged_or_refused(root, table, dtypes, tmp_path):
    return _outcome(_hosted_staged, root, table, dtypes, tmp_path)


def test_the_corpus_covers_every_family():
    """A family added to the vocabulary and not reached by a case would look
    like coverage without being any."""
    reached = set()
    for _, columns in _corpus():
        reached.update(columns)
    assert reached == set(VOCABULARY) | set(MIXTURES)


def test_the_two_backends_hold_a_csv_table_to_the_same_ceilings():
    """A table the cloud refuses must not be buildable locally, so the
    partition and size ceilings are one pair of constants both enforce."""
    from kumo_core.sources import layout

    from kumo.connector import csv_connector

    assert csv_connector.MAX_CSV_PARTITIONS == layout.MAX_CSV_PARTITIONS
    assert csv_connector.MAX_CSV_DATASIZE == layout.MAX_CSV_DATASIZE
    assert layout.csv_partitions_refusal(layout.MAX_CSV_PARTITIONS) is None
    assert 'partitions' in str(
        layout.csv_partitions_refusal(layout.MAX_CSV_PARTITIONS + 1))
    assert layout.csv_size_refusal(layout.MAX_CSV_DATASIZE) is None
    assert 'bytes' in str(layout.csv_size_refusal(layout.MAX_CSV_DATASIZE + 1))
