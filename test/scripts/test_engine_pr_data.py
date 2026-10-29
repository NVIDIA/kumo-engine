# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject corrupted or unsafe bundled PR172 fixtures without external access."""

import hashlib
import io
import json
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.agent_local

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'packages/kumo-engine/test/pr172'


def record(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def fixture_archive(directory, entries):
    archive = directory / 'fixture.tar.gz'
    files = {}
    with tarfile.open(archive, 'w:gz') as stream:
        for name, content, kind in entries:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = len(content) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                entry.linkname = '../outside'
            stream.addfile(entry, io.BytesIO(content))
            files[name] = dict(bytes=len(content),
                               sha256=hashlib.sha256(content).hexdigest())
    return archive, {'archive': record(archive), 'files': files}


class PRDataTests(unittest.TestCase):
    def setUp(self):
        self.worker = runpy.run_path(str(SOURCE / 'prepare.py'))

    def test_bundled_data_extracts_and_reuses_without_a_source_checkout(self):
        self.assertIn('prepare_data', self.worker)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'inputs'
            first = self.worker['prepare_data'](output)
            self.assertEqual(first['data_root'], str(output.resolve() / 'data'))
            self.assertEqual(first['source_root'], str(SOURCE))
            self.assertEqual(first['rows'], {'adtech': 556127,
                                             'delivery': 1442968,
                                             'commerce': 74416})
            second = self.worker['prepare_data'](output)
            self.assertTrue(second['cache_reused'])
            table = next((output / 'data/adtech/pr/users').glob('*.parquet'))
            before = table.read_bytes()
            table.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'fixture fingerprint'):
                self.worker['prepare_data'](output)

    def test_cache_requires_its_unchanged_preparation_receipt(self):
        prepare = self.worker['prepare_data']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'inputs'
            prepare(output)
            receipt = output / 'pr172-inputs.json'
            original = receipt.read_bytes()
            files = {str(p.relative_to(output)): record(p)
                     for p in (output / 'data').rglob('*') if p.is_file()}
            receipt.unlink()
            with self.assertRaisesRegex(ValueError, 'cached preparation receipt'):
                prepare(output)
            self.assertFalse(receipt.exists())
            for key in ('data_root', 'source_root', 'provenance_sha256',
                        'archive_sha256', 'manifest_sha256', 'rows'):
                with self.subTest(key=key):
                    changed = json.loads(original)
                    changed[key] = 'changed'
                    content = json.dumps(changed).encode()
                    receipt.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, 'cached preparation receipt'):
                        prepare(output)
                    self.assertEqual(receipt.read_bytes(), content)
            receipt.write_bytes(original)
            self.assertTrue(prepare(output)['cache_reused'])
            self.assertEqual(files, {str(p.relative_to(output)): record(p)
                                     for p in (output / 'data').rglob('*')
                                     if p.is_file()})

    def test_corrupted_compressed_archive_is_rejected_before_extraction(self):
        self.assertIn('extract_archive', self.worker)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, integrity = fixture_archive(
                root, [('adtech/pr/table.parquet', b'rows', tarfile.REGTYPE)])
            archive.write_bytes(archive.read_bytes() + b'corruption')
            with self.assertRaisesRegex(ValueError, 'archive fingerprint'):
                self.worker['extract_archive'](archive, root / 'data', integrity)
            self.assertFalse((root / 'data').exists())

    def test_traversal_links_and_duplicate_members_are_rejected(self):
        self.assertIn('extract_archive', self.worker)
        cases = [('../escape', tarfile.REGTYPE),
                 ('/absolute', tarfile.REGTYPE),
                 ('adtech/../escape', tarfile.REGTYPE),
                 ('adtech/pr/link', tarfile.SYMTYPE)]
        for name, kind in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                archive, integrity = fixture_archive(root, [(name, b'rows', kind)])
                with self.assertRaisesRegex(ValueError, 'Unsafe archive member'):
                    self.worker['extract_archive'](archive, root / 'data', integrity)
                self.assertFalse((root / 'data').exists())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entry = ('adtech/pr/table.parquet', b'rows', tarfile.REGTYPE)
            archive, integrity = fixture_archive(root, [entry, entry])
            with self.assertRaisesRegex(ValueError, 'Duplicate archive member'):
                self.worker['extract_archive'](archive, root / 'data', integrity)

    def test_archive_inventory_and_member_bytes_are_checked(self):
        self.assertIn('extract_archive', self.worker)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, integrity = fixture_archive(
                root, [('adtech/pr/table.parquet', b'rows', tarfile.REGTYPE)])
            integrity['files']['missing'] = dict(bytes=0, sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'archive inventory'):
                self.worker['extract_archive'](archive, root / 'data', integrity)
            integrity['files'].pop('missing')
            integrity['files']['adtech/pr/table.parquet']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'fixture fingerprint'):
                self.worker['extract_archive'](archive, root / 'data', integrity)

    def test_validation_stamp_cannot_cover_changed_manifest_counts(self):
        validate = self.worker['validate_dataset']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / 'adtech/pr'
            table = profile / 'users/part-0.parquet'
            table.parent.mkdir(parents=True)
            pq.write_table(pa.table({'user_id': [1, 2]}), table)
            manifest = {
                'graph': 'adtech', 'rung': 'pr', 'shards': 2,
                'seed': 20261009, 'shard_entities': 1024, 'total_rows': 2,
                'tables': {'users': {'rows': 2, 'files': [{
                    'path': 'adtech/pr/users/part-0.parquet', 'rows': 2,
                    **record(table)}]}},
            }
            content = json.dumps(manifest).encode()
            (profile / 'manifest.json').write_bytes(content)
            stamp = profile / '_VALIDATED.json'
            stamp.write_text(json.dumps({
                'ok': True, 'checksums_verified': True, 'violations': 0,
                'manifest_sha256': hashlib.sha256(content).hexdigest()}))
            expected = {'adtech': {'shards': 2, 'shard_entities': 1024,
                                   'rows': {'users': 2}}}
            validate(root, expected)
            expected['adtech']['rows']['users'] = 3
            with self.assertRaisesRegex(ValueError, 'profile differs'):
                validate(root, expected)
            expected['adtech']['rows']['users'] = 3
            manifest['total_rows'] = 3
            manifest['tables']['users']['rows'] = 3
            manifest['tables']['users']['files'][0]['rows'] = 3
            content = json.dumps(manifest).encode()
            (profile / 'manifest.json').write_bytes(content)
            value = json.loads(stamp.read_text())
            value['manifest_sha256'] = hashlib.sha256(content).hexdigest()
            stamp.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Parquet row count'):
                validate(root, expected)
            value = json.loads(stamp.read_text())
            value['checksums_verified'] = False
            stamp.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'validation stamp'):
                validate(root, expected)


if __name__ == '__main__':
    unittest.main()
