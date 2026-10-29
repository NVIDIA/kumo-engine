# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject changed generator sources and incomplete generated data caches."""

import hashlib
import json
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'packages/kumo-engine/test/pr172'


class PRDataTests(unittest.TestCase):
    def test_source_check_rejects_changed_pinned_source(self):
        verify = runpy.run_path(str(SOURCE / 'prepare.py'))['verify_source']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'scalebench/generate.py'
            path.parent.mkdir()
            path.write_bytes(b'print("fixed")\n')
            record = {
                'copied_files_sha256': {
                    'scalebench/generate.py':
                    hashlib.sha256(b'print("fixed")\n').hexdigest()
                }
            }
            verify(root, record)
            path.write_bytes(b'print("changed")\n')
            with self.assertRaisesRegex(ValueError,
                                        'source fingerprint differs'):
                verify(root, record)

    def test_source_check_rejects_a_manifest_path_escape(self):
        verify = runpy.run_path(str(SOURCE / 'prepare.py'))['verify_source']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'outside.py'
            path.write_bytes(b'outside')
            source = root / 'source'
            source.mkdir()
            record = {
                'copied_files_sha256': {
                    '../outside.py': hashlib.sha256(b'outside').hexdigest()
                }
            }
            with self.assertRaisesRegex(ValueError,
                                        'source fingerprint differs'):
                verify(source, record)

    def test_cached_source_rejects_unpinned_executable_files(self):
        verify = runpy.run_path(str(SOURCE / 'prepare.py'))['verify_source']
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            file = source / 'scalebench/generate.py'
            file.parent.mkdir()
            file.write_bytes(b'fixed')
            record = {
                'copied_files_sha256': {
                    'scalebench/generate.py':
                    hashlib.sha256(b'fixed').hexdigest()
                },
                'generated_profiles_sha256': {}
            }
            verify(source, record, exact=True)
            (source / 'scalebench/__init__.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source inventory differs'):
                verify(source, record, exact=True)

    def test_validation_stamp_cannot_cover_a_changed_table(self):
        self.assertTrue((SOURCE / 'prepare.py').is_file())
        validate = runpy.run_path(str(SOURCE /
                                      'prepare.py'))['validate_dataset']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / 'adtech/pr'
            table = profile / 'users/part-0.parquet'
            table.parent.mkdir(parents=True)
            table.write_bytes(b'unchanged rows')
            manifest = {
                'graph': 'adtech',
                'rung': 'pr',
                'shards': 2,
                'seed': 20261009,
                'shard_entities': 1024,
                'total_rows': 2,
                'tables': {
                    'users': {
                        'rows':
                        2,
                        'files': [{
                            'path':
                            'adtech/pr/users/part-0.parquet',
                            'rows':
                            2,
                            'bytes':
                            14,
                            'sha256':
                            hashlib.sha256(b'unchanged rows').hexdigest()
                        }]
                    }
                },
            }
            content = json.dumps(manifest).encode()
            (profile / 'manifest.json').write_bytes(content)
            (profile / '_VALIDATED.json').write_text(
                json.dumps({
                    'ok':
                    True,
                    'checksums_verified':
                    True,
                    'violations':
                    0,
                    'manifest_sha256':
                    hashlib.sha256(content).hexdigest()
                }))
            expected = {
                'adtech': {
                    'shards': 2,
                    'shard_entities': 1024,
                    'rows': {
                        'users': 2
                    }
                }
            }
            validate(root, expected)
            unexpected = profile / 'users/extra.parquet'
            unexpected.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'table inventory differs'):
                validate(root, expected)
            unexpected.unlink()
            table.write_bytes(b'changed values')
            with self.assertRaisesRegex(ValueError,
                                        'table fingerprint differs'):
                validate(root, expected)
            table.write_bytes(b'unchanged rows')
            stamp = profile / '_VALIDATED.json'
            value = json.loads(stamp.read_text())
            value['checksums_verified'] = False
            stamp.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'validation stamp'):
                validate(root, expected)


if __name__ == '__main__':
    unittest.main()
