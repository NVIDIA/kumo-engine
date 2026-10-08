# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check real fixture lifecycles in separate pytest processes."""

from pathlib import Path

import pytest

pytest_plugins = ['pytester']


@pytest.mark.parametrize('setup, enabled', [
    ('test', True),
    ('test', False),
    ('test/connector/local', True),
    ('test/graph/local', True),
])
@pytest.mark.parametrize('fail', [False, True])
def test_cache_lifecycle(pytester, monkeypatch, setup, enabled, fail):
    root = Path(__file__).resolve().parents[3]
    conftest = root / setup / 'conftest.py'
    # Exercise ordinary discovery, without the public runner's plugin.
    source = conftest.read_text() if conftest.exists() else ''
    source += '''
import pytest
from kumo.graph.table import Table

@pytest.fixture(scope='session', autouse=True)
def check_statistics_restored():
    original = (Table.compute_col_stats, Table.compute_exact_stats)
    yield
    assert (Table.compute_col_stats, Table.compute_exact_stats) == original
'''
    if not enabled:
        source += '''
@pytest.fixture
def auto_setup_kumo_backend():
    return False
'''
    pytester.makeconftest(source)
    monkeypatch.setenv('HOME', str(pytester.path / 'home'))
    monkeypatch.setenv('KUMO_HOME', str(pytester.path / 'home/kumo'))
    pytester.makepyfile(f'''
from pathlib import Path
import pytest
import kumo.home
from kumo import data
from kumo_core.types import ColStatType
from statedb.utils.errors import UnsupportedInDiskgraphError
from test.graph.local.test_table import _build_pandas_table

persist = Path(data.get_persist_dir())
pseudo = Path(kumo.home.HomeDirManager(kumo.home.get_home_dir()).pseudo_time_dir)
cache = None

def test_write():
    global cache
    if {enabled!r}:
        assert persist.is_dir(), 'Fixture must prepare the cache directory'
    table = _build_pandas_table()
    if {setup == 'test' and enabled!r}:
        stats = table.compute_col_stats(col_stat_types=[ColStatType.NA_FRACTION], approx=False)
        assert stats['name'][ColStatType.NA_FRACTION].value == 0.2
    else:
        with pytest.raises(UnsupportedInDiskgraphError, match='Requested col stats missing'):
            table.compute_col_stats(col_stat_types=[ColStatType.NA_FRACTION], approx=False)
    cache = data.get_active_cache()
    for path in (persist, pseudo):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'previous-test').write_text('test data')
    if {fail!r}:
        pytest.fail('intentional test failure')

def test_next_starts_clean():
    for path in (persist, pseudo):
        assert (path / 'previous-test').exists() is {not enabled!r}
    assert (data.get_active_cache() is cache) is {not enabled!r}
''')
    result = pytester.runpytest_subprocess(
        '-c', str(root / 'tests/public_setup/pytest.ini'),
        f'--confcutdir={pytester.path}', f'--rootdir={pytester.path}',
        '-p', 'no:cacheprovider', '-o', 'addopts=', '-q',
        str(pytester.path), timeout=120)
    result.assert_outcomes(passed=1 if fail else 2, failed=int(fail))
    if fail:
        result.stdout.fnmatch_lines(['*Failed: intentional test failure*'])
