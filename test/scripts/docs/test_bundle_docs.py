# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check authored Kumo links in the real bundle layout, without live services."""
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'repository_docs', ROOT / 'scripts/docs/check_repository_docs.py')
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


class BundleDocsTest(unittest.TestCase):
    def test_local_links_survive_bundle_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helm = root / 'helm'
            shutil.copytree(ROOT / 'helm', helm, symlinks=True)
            shutil.copy(ROOT / 'LICENSE', root / 'LICENSE')
            # The assembler only checks existence. This inert fixture supplies
            # no trust material; nothing is rendered, installed, or published.
            ca = helm / 'kumo-temporal/files/aws-rds-roots.pem'
            ca.parent.mkdir(parents=True, exist_ok=True)
            ca.write_text('Documentation layout fixture only.\n')
            result = subprocess.run(
                ['bash', str(helm / 'kumo-stack/build-bundle.sh'),
                 str(root / 'output')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = root / 'output/kumo-stack'
            files = [bundle / 'README.md', *sorted((bundle / 'docs').glob('*.md'))]
            for chart in sorted((bundle / 'charts').glob('kumo-*')):
                files.append(chart / 'README.md')
                files.extend(sorted((chart / 'docs').glob('*.md')))
            errors = []
            for path in files:
                for line, link in docs.links(path.read_text()):
                    target = urlsplit(link)
                    if target.scheme or target.netloc:
                        continue
                    dest = (path.parent / unquote(target.path)).resolve()
                    if not dest.is_relative_to(bundle) or not dest.exists():
                        errors.append(f'{path.relative_to(bundle)}:{line}: {link}')
            self.assertFalse(errors, 'Broken packaged links:\n' + '\n'.join(errors))


if __name__ == '__main__':
    unittest.main()
