# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the shared builder with a small fake native build boundary."""
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / 'scripts/nightly_build.py'


class WheelBuildChecks(unittest.TestCase):
    def test_build_receipt_binds_wheel_version_and_unchanged_source(self):
        self.assertTrue(SCRIPT.is_file(), 'Shared builder is missing')
        spec = importlib.util.spec_from_file_location('builder', SCRIPT)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            product = root / 'product'
            product.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ['git', '-C', str(product), *args], text=True).strip()

            git('init', '-b', 'main')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.com')
            (product / 'VERSION').write_text('1.2.3\n')
            script = product / 'packages/kumo-engine/scripts/build_wheel.sh'
            script.parent.mkdir(parents=True)
            script.write_text('''#!/bin/sh
exec python3 - "$@" <<'BUILD'
import argparse,os,pathlib,subprocess,zipfile
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir')
parser.add_argument('--kumo-repo',default=os.environ.get('KUMO_REPO',os.getcwd()))
parser.add_argument('--version',default=os.environ.get('BUNDLE_VERSION'))
args=parser.parse_args()
p=pathlib.Path(args.output_dir);p.mkdir(exist_ok=True)
v=args.version or '1.2.3+'+subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()
with zipfile.ZipFile(p/'candidate.whl','w') as z:
 z.writestr('kumo_engine.dist-info/METADATA','Name: kumo-engine\\nVersion: '+v+'\\n')
 z.writestr('selected-source.txt',args.kumo_repo)
BUILD
''')
            script.chmod(0o755)
            git('add', '.')
            git('commit', '-m', 'fixture')
            with patch.dict(os.environ, KUMO_REPO='wrong-source',
                            BUNDLE_VERSION='9.9.9'):
                result = builder.build(product, root / 'output')
            self.assertEqual(result['revision'], git('rev-parse', 'HEAD'))
            self.assertEqual(result['version'],
                             '1.2.3+' + git('rev-parse', '--short', 'HEAD'))
            self.assertEqual(
                json.loads((root / 'output/build.json').read_text()), result)
            with zipfile.ZipFile(root / 'output/candidate.whl') as archive:
                self.assertEqual(
                    archive.read('selected-source.txt').decode(), str(product))
            (product / 'VERSION').write_text('changed')
            with self.assertRaises(ValueError):
                builder.build(product, root / 'rejected')


if __name__ == '__main__':
    unittest.main()
