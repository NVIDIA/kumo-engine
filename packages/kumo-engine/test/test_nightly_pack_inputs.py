# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split real tiny tar streams without Engine, storage access, or model work."""

import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_pack_inputs.py'
FAMILIES = ('adtech', 'delivery', 'commerce', 'streaming', 'community',
            'payments', 'fraud')


class ForwardOnly(io.BytesIO):
    def read(self, size=-1):
        assert 0 <= size <= 1024 * 1024, 'Use bounded forward reads'
        return super().read(size)

    def seek(self, *args):
        raise AssertionError('The source cannot seek')


class PackInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payloads = {
            'schema.json': b'{"fixture":"schema"}',
            'dependencies.json': b'{}'
        }
        graphs, cases = {}, []
        for family in FAMILIES:
            raw = f'raw/{family}/table/part.parquet'
            prefix = f'cases/{family}'
            names = [raw, f'source-manifests/{family}.json'] + [
                f'{prefix}/{name}'
                for name in ('query.json', 'population.parquet',
                             'normal.parquet', 'control.parquet', 'probe.json')
            ]
            self.payloads.update(
                {name: ('archive fixture: ' + name).encode()
                 for name in names})
            graphs[family] = dict(profile=family, schema='schema.json',
                                  sources={'table': f'raw/{family}/table'})
            for arm in ('normal', 'control'):
                row = dict(scale='100M', task='binary', variant=family, arm=arm,
                           graph=family, descriptor=prefix + '/query.json',
                           prediction_input=prefix + '/population.parquet',
                           normal_rows=[prefix + '/normal.parquet'],
                           preparation_probe=prefix + '/probe.json')
                if arm == 'control':
                    row['control_rows'] = [prefix + '/control.parquet']
                cases.append(row)
        self.manifest = dict(
            format='kumo-engine-nightly-inputs-v2', scale='100M', cases=cases,
            graphs=graphs, reference_dependencies='dependencies.json', files={
                name: hashlib.sha256(value).hexdigest()
                for name, value in self.payloads.items()
            })
        self.original = json.dumps(self.manifest, indent=2).encode() + b'\n'
        self.entries = [('manifest.json', self.original, tarfile.REGTYPE)
                        ] + [(name, value, tarfile.REGTYPE)
                             for name, value in sorted(self.payloads.items())]

    def api(self):
        self.assertTrue(SOURCE.is_file(), 'The streaming splitter is missing')
        spec = importlib.util.spec_from_file_location('nightly_pack_tested',
                                                      SOURCE)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        return api

    def archive(self, entries=None):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as archive:
            for name, value, kind in self.entries if entries is None else entries:
                member = tarfile.TarInfo(name)
                member.type, member.size = kind, len(value)
                member.mode, member.mtime = 0o666, 1700000000
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = '../outside'
                    member.size = 0
                archive.addfile(member, io.BytesIO(value))
        return output.getvalue()

    def split(self, source, expected=None, output=None):
        return self.api().pack_workloads(
            ForwardOnly(source), output or self.root / 'outputs', self.original,
            expected or hashlib.sha256(source).hexdigest())

    def failed(self, source, expected=None):
        with self.assertRaises((ValueError, tarfile.TarError, OSError)):
            self.split(source, expected)
        self.assertFalse((self.root / 'outputs').exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_round_trip_preserves_original_manifest_and_family_payloads(self):
        source = self.archive()
        receipts = self.split(source)
        self.assertEqual(set(receipts), set(FAMILIES))
        for family, receipt in receipts.items():
            path = Path(receipt['archive_path'])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                receipt['archive_sha256'])
            self.assertEqual(path.stat().st_size, receipt['archive_bytes'])
            self.assertEqual(receipt['manifest_sha256'],
                             hashlib.sha256(self.original).hexdigest())
            self.assertEqual(receipt['parent_archive_sha256'],
                             hashlib.sha256(source).hexdigest())
            with tarfile.open(path) as archive:
                expected = {'manifest.json', 'schema.json', 'dependencies.json'}
                expected |= {
                    name
                    for name in self.payloads
                    if name.startswith((f'raw/{family}/', f'cases/{family}/'))
                }
                expected.add(f'source-manifests/{family}.json')
                self.assertEqual(set(archive.getnames()), expected)
                self.assertEqual(receipt['files'], len(expected))
                for member in archive:
                    self.assertEqual((member.mode, member.mtime), (0o444, 0))
                    value = (self.original if member.name == 'manifest.json'
                             else self.payloads[member.name])
                    self.assertEqual(archive.extractfile(member).read(), value)

    def test_changed_payload_and_whole_source_hash_fail_without_outputs(self):
        changed = list(self.entries)
        name, _, kind = changed[-1]
        changed[-1] = (name, b'changed', kind)
        self.failed(self.archive(changed))
        self.failed(self.archive(), '0' * 64)

    def test_missing_duplicate_extra_or_changed_manifest_fail(self):
        changed_manifest = [
            ('manifest.json', self.original + b' ', tarfile.REGTYPE)
        ] + self.entries[1:]
        for entries in (self.entries[:-1], self.entries + self.entries[-1:],
                        self.entries +
                        [('extra.txt', b'extra', tarfile.REGTYPE)],
                        self.entries + self.entries[:1], changed_manifest):
            with self.subTest(entries=len(entries)):
                self.failed(self.archive(entries))

    def test_unsafe_names_and_links_fail(self):
        for name in ('../escape', '/absolute', './file', 'a//b', 'a\\b'):
            with self.subTest(name=name):
                self.failed(
                    self.archive(self.entries +
                                 [(name, b'x', tarfile.REGTYPE)]))
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
                     tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind):
                self.failed(self.archive(self.entries + [('link', b'', kind)]))

    def test_truncated_or_nonzero_trailing_data_fail(self):
        source = self.archive()
        with tarfile.open(fileobj=io.BytesIO(source)) as archive:
            end = max(member.offset_data + (member.size + 511) // 512 * 512
                      for member in archive)
        for bad in (source[:-1], source[:1024], source[:end],
                    source[:end + 512], source + b'not a permitted tail',
                    source[:end] + b'x' * 512 + source[end + 512:]):
            with self.subTest(length=len(bad)):
                self.failed(bad)
        self.failed(source[:-512], hashlib.sha256(source).hexdigest())
        self.failed(source[:-1] + b'x', hashlib.sha256(source).hexdigest())

    def test_existing_output_and_publish_error_leave_owned_data_untouched(self):
        output = self.root / 'outputs'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.split(self.archive())
        self.assertEqual(marker.read_text(), 'preserve')
        marker.unlink()
        output.rmdir()
        api = self.api()
        with patch.object(api.os, 'rename',
                          side_effect=OSError('publish failed')):
            with self.assertRaises(OSError):
                api.pack_workloads(ForwardOnly(self.archive()), output,
                                   self.original,
                                   hashlib.sha256(self.archive()).hexdigest())
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
