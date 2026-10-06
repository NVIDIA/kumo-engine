# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog regeneration detects drift without requiring an installed UI."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / '.agents/skills/code-ui/scripts/generate-catalog-docs.sh'


class CatalogDocsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.catalog = self.root / 'packages/kumo-ui-catalog'
        (self.catalog / 'src').mkdir(parents=True)
        (self.catalog / 'package.json').write_text('{"version": "1.0.0"}\n')
        (self.catalog / 'src/index.ts').write_text('export { Button } from "./components/ui/button";\n')
        self.output = self.root / '.agents/skills/code-ui/references/catalog-components.md'
        self.output.parent.mkdir(parents=True)

    def run_script(self, *args, env=None):
        return subprocess.run(['bash', str(SCRIPT), *args], cwd=self.root,
                              capture_output=True, text=True, env=env)

    def test_in_tree_source_and_read_only_drift_check(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = self.output.read_bytes()
        self.assertIn(b'Button', before)
        self.assertEqual(self.run_script('--check').returncode, 0)
        with (self.catalog / 'src/index.ts').open('a') as f:
            f.write('export { Badge } from "./components/ui/badge";\n')
        result = self.run_script('--check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale', result.stdout + result.stderr)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.run_script().returncode, 0)
        self.assertIn(b'Badge', self.output.read_bytes())
        self.assertEqual(self.run_script('--check').returncode, 0)

    def test_output_is_independent_of_collation_locale(self):
        available = subprocess.check_output(['locale', '-a'], text=True).splitlines()
        english = next((name for name in ('en_US.UTF-8', 'en_US.utf8')
                        if name in available), None)
        if english is None:
            self.skipTest('English UTF-8 locale is not installed')
        (self.catalog / 'src/index.ts').write_text(
            'export { Zebra, alpha } from "./components/ui/example";\n'
            'export type { ZebraProps, alphaProps } from "./components/ui/example";\n')
        result = self.run_script(env={**os.environ, 'LC_ALL': 'C'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = self.output.read_bytes()
        env = {**os.environ, 'LC_ALL': english}
        result = self.run_script('--check', env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_script(env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.output.read_bytes(), before)

    def test_runtime_functions_are_not_classified_as_types(self):
        (self.catalog / 'src/index.ts').write_text(
            'export { Button, buttonVariants } from "./components/ui/button";\n'
            'export { useDebounce } from "./hooks/use-debounce";\n'
            'export type { ButtonProps } from "./components/ui/button";\n')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runtime, types = self.output.read_text().split('## Type Exports')
        for name in ('Button', 'buttonVariants', 'useDebounce'):
            self.assertIn(f'`{name}`', runtime)
            self.assertNotIn(f'`{name}`', types)
        self.assertIn('`ButtonProps`', types)
        self.assertNotIn('`ButtonProps`', runtime)


class CatalogDiffScanTest(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[3] / (
        'packages/kumo-ui-catalog/.agents/skills/review-ui-catalog/scripts/diff-scan.sh')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gh = self.root / 'gh'
        self.gh.write_text('''#!/bin/sh
if [ "$GH_FAIL" = 1 ]; then exit 2; fi
if [ "$#" -ne 3 ] || [ "$1" != pr ] || [ "$2" != diff ] || [ "$3" != 17 ]; then
  exit 2
fi
cat "$DIFF_FIXTURE"
''')
        self.gh.chmod(0o755)
        self.fixture = self.root / 'pr.diff'

    def run_scan(self, *, fail=False):
        env = {**os.environ, 'PATH': f'{self.root}:{os.environ["PATH"]}',
               'DIFF_FIXTURE': str(self.fixture), 'GH_FAIL': '1' if fail else '0'}
        return subprocess.run(['bash', str(self.SCRIPT), '17'],
                              capture_output=True, text=True, env=env)

    def test_scans_monorepo_catalog_path_without_other_packages(self):
        self.fixture.write_text('''diff --git a/packages/kumo-ui-catalog/src/components/ui/badge.tsx b/packages/kumo-ui-catalog/src/components/ui/badge.tsx
--- a/packages/kumo-ui-catalog/src/components/ui/badge.tsx
+++ b/packages/kumo-ui-catalog/src/components/ui/badge.tsx
@@ -1,0 +1,1 @@
+<div style={{ color: "red" }} />
diff --git a/ui/src/components/other.tsx b/ui/src/components/other.tsx
--- a/ui/src/components/other.tsx
+++ b/ui/src/components/other.tsx
@@ -1,0 +1,1 @@
+<div style={{ color: "blue" }} />
''')
        result = self.run_scan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('inline-styles\tpackages/kumo-ui-catalog/src/components/ui/badge.tsx\t1',
                      result.stdout)
        self.assertNotIn('ui/src/components/other.tsx', result.stdout)

    def test_renames_use_destination_for_filtering_and_reporting(self):
        outside = 'ui/src/components/badge.tsx'
        inside = 'packages/kumo-ui-catalog/src/components/ui/badge.tsx'
        renamed = 'packages/kumo-ui-catalog/src/components/ui/renamed-badge.tsx'
        for old, new, expected in ((outside, inside, True),
                                   (inside, renamed, True),
                                   (inside, outside, False)):
            with self.subTest(old=old, new=new):
                self.fixture.write_text(f'''diff --git a/{old} b/{new}
similarity index 95%
rename from {old}
rename to {new}
--- a/{old}
+++ b/{new}
@@ -5,1 +7,2 @@
 unchanged context
+<div style={{{{ color: "red" }}}} />
''')
                result = self.run_scan()
                self.assertEqual(result.returncode, 0, result.stderr)
                if expected:
                    self.assertEqual(result.stdout.splitlines(), [
                        f'inline-styles\t{new}\t8\t<div style={{{{ color: "red" }}}} />'])
                    self.assertIn('1 findings in 1 files', result.stderr)
                else:
                    self.assertEqual(result.stdout, '')
                    self.assertIn('0 findings in 0 files', result.stderr)

    def test_diff_fetch_failure_is_not_reported_as_zero_findings(self):
        self.fixture.write_text('')
        result = self.run_scan(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('0 findings', result.stderr)


if __name__ == '__main__':
    unittest.main()
