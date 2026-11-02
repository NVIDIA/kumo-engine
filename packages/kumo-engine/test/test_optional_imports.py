# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bare `pip install kumo-engine` ships provider code but not its
third-party dependencies (those come with the `kumo-engine[<provider>]`
extras). Importing `kumo.engine` and using `FileConnector` must work there;
using a provider connector must fail with an ImportError naming the extra.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ENGINE = Path(__file__).parents[1] / 'kumo' / 'engine'

#: Third-party packages that only an extra installs.
OPTIONAL_THIRD_PARTY = (
    'snowflake',
    'databricks',
    'boto3',
    'botocore',
    's3fs',
    'google',
    'azure',
    'gcsfs',
)
PROVIDER_PACKAGES = (
    'kumo_snowflake',
    'kumo_databricks',
    'kumo_s3',
    'kumo_gcs',
    'kumo_adls',
    'kumo_bigquery',
)
FORBIDDEN_AT_IMPORT = PROVIDER_PACKAGES + OPTIONAL_THIRD_PARTY


def _imported_modules(path: Path) -> set:
    """Top-level names of every module `path` imports, anywhere in the file."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            found.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add(node.module.split('.')[0])
    return found


@pytest.mark.parametrize('path', sorted(ENGINE.rglob('*.py')),
                         ids=lambda p: str(p.relative_to(ENGINE)))
def test_engine_modules_import_providers_only_through_import_provider(path):
    if path.name == '_extras.py':  # the helper itself
        return
    offending = _imported_modules(path) & set(FORBIDDEN_AT_IMPORT)
    assert not offending, (
        f'{path.relative_to(ENGINE)} imports {sorted(offending)} directly. '
        f'Provider code may not be imported at module level (it breaks '
        f'`import kumo.engine` on an install without the matching extra), and '
        f'inside a function it must go through '
        f'kumo.engine.connector._extras.import_provider so a missing extra '
        f'raises an ImportError that names it.')


_BARE_INSTALL = textwrap.dedent('''
    import importlib.abc
    import sys
    import types

    BLOCKED = {blocked!r}


    class Block(importlib.abc.MetaPathFinder):
        """Make the extras' third-party packages look uninstalled."""

        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}",
                                          name=name)


    sys.meta_path.insert(0, Block())

    import kumo.engine as engine

    # What works without any extra.
    engine.FileConnector(root_dir={root!r})
    assert engine.__version__

    # Everything else is importable. Using it either fails on the missing
    # dependency, naming the extra, or the provider defers that import
    # further and fails later.
    for cls, extra in (
            (engine.ADLSConnector, 'adls'),
            (engine.SnowflakeConnector, 'snowflake'),
            (engine.DatabricksConnector, 'databricks'),
            (engine.S3Connector, 's3'),
            (engine.GCSConnector, 'gcs'),
            (engine.BigQueryConnector, 'bigquery'),
    ):
        for attr in ('source_backend', 'sink_backend'):
            prop = cls.__dict__.get(attr)
            if prop is None:  # inherited, or not offered by this provider
                continue
            try:
                # cached_property exposes `.func`, property `.fget`.
                (getattr(prop, 'func', None) or prop.fget)(
                    types.SimpleNamespace())
            except ImportError as e:
                assert f"kumo-engine[{{extra}}]" in str(e), (cls, attr, e)
            except AttributeError:
                # The provider module imported without its extra (it defers
                # its own third-party imports), so we got past the import
                # and only the stand-in `self` is missing. Fine: nothing
                # broke at import time.
                pass
    print("ok")
''')


def test_file_connector_works_without_any_extra(tmp_path):
    result = subprocess.run(
        [
            sys.executable, '-c',
            _BARE_INSTALL.format(blocked=set(OPTIONAL_THIRD_PARTY),
                                 root=str(tmp_path))
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'
