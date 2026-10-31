# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise public discovery, Docker contexts, failure reporting and cleanup."""

import contextlib
import io
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import test_public as runner  # noqa: E402


class PublicRunnerTest(unittest.TestCase):
    def test_focused_selection_reaches_every_group(self):
        for group, paths in runner.GROUPS.items():
            for path in paths:
                selected = next((runner.ROOT / path).glob('test_*.py'))
                selected = str(selected.relative_to(runner.ROOT))
                with self.subTest(group=group, file=selected), \
                        tempfile.TemporaryDirectory() as directory:
                    reports = Path(directory)
                    (reports / 'wheel').mkdir()
                    (reports / 'wheel/diskgraph-example.whl').touch()
                    commands = []

                    def execute(command, **kwargs):
                        if command[0] == 'openssl':
                            Path(command[command.index('-out') + 1]).touch()
                        if command[:2] == ['docker', 'run']:
                            commands.append(command)
                            output = Path(command[command.index('-v') +
                                                  1].rsplit(':', 1)[0])
                            (output / f'{group}.xml').write_text(
                                '<testsuites><testsuite tests="1"/></testsuites>')
                        return subprocess.CompletedProcess(command, 0)

                    arguments = ['test_public.py', '--group', group,
                                 '--reports', directory, '--test-file', selected,
                                 '-k', 'keep and not drop']
                    with patch.object(sys, 'argv', arguments), \
                            patch.object(runner.subprocess, 'check_output', return_value='x86_64'), \
                            patch.object(runner.subprocess, 'run', side_effect=execute), \
                            patch.object(runner, 'build_wheel'), \
                            patch.object(runner, 'build_image'), \
                            patch.object(runner, 'image_environment', return_value={}), \
                            contextlib.redirect_stdout(io.StringIO()), \
                            contextlib.redirect_stderr(io.StringIO()):
                        try:
                            result = runner.main()
                        except SystemExit as error:
                            result = error.code
                    self.assertEqual(result, 0)
                    command, = commands
                    self.assertIn(selected, command)
                    self.assertEqual(command[command.index('-k') + 1],
                                     'keep and not drop')
                    for root in paths:
                        self.assertNotIn(root, command)
                    if group in runner.SUITES:
                        self.assertEqual(command[command.index('--test-file') + 1],
                                         selected)

    def test_focused_run_rejects_multiple_groups_and_build_only(self):
        for flags in ([], ['--group', 'connectors'],
                      ['--group', 'csv', '--build-wheel']):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                with patch.object(sys, 'argv', ['test_public.py', '--reports', directory, '-k', 'keep', *flags]), \
                        patch.object(runner.subprocess, 'check_output') as probe, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        runner.main()
                    self.assertEqual(error.exception.code, 2)
                    probe.assert_not_called()

    def test_groups_discover_directories(self):
        for group, paths in runner.GROUPS.items():
            for path in paths:
                with self.subTest(group=group, path=path):
                    self.assertTrue((runner.ROOT / path).is_dir(),
                                    'Public groups must discover directories')

    @unittest.skipUnless(shutil.which('docker'), 'Docker is required')
    def test_docker_includes_new_tests_but_not_developer_credentials(self):
        for profile in ('public', 'public_setup'):
            with self.subTest(profile=profile), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context = root / 'context'
                context.mkdir()
                paths = set()
                for group, entries in runner.GROUPS.items():
                    if (group in runner.SUITES) != (profile == 'public'):
                        continue
                    paths.update(Path(entry) for entry in entries)
                for path in paths:
                    target = context / path / 'new_cases'
                    target.mkdir(parents=True, exist_ok=True)
                    (target / 'test_unregistered.py').write_text(
                        'def test_new(): assert True\n')
                    (target / '.env').write_text('SYNTHETIC=placeholder\n')
                artifact_modules = (
                    'artifacts.py', 'config.py', 'manifest.py',
                    'io/__init__.py', 'io/filesystem.py', 'io/json.py',
                ) if profile == 'public' else ()
                core_path = Path('packages/kumo-core/src/kumo_core')
                for module in artifact_modules:
                    target = context / core_path / module
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text('# artifact dependency fixture\n')
                manifests = []
                if profile == 'public_setup':
                    workspace = tomllib.loads(
                        (runner.ROOT / 'pyproject.toml').read_text())
                    manifests = [
                        Path(member) / 'pyproject.toml' for member in
                        workspace['tool']['uv']['workspace']['members']
                    ]
                    for manifest in manifests:
                        target = context / manifest
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text('# workspace metadata fixture\n')
                (context / 'Dockerfile').write_text('FROM scratch\nCOPY . /\n')
                shutil.copyfile(
                    runner.ROOT / 'tests' / profile / 'Dockerfile.dockerignore',
                    context / 'Dockerfile.dockerignore')
                subprocess.run([
                    'docker', 'buildx', 'build', '--output',
                    f'type=local,dest={root / "output"}', str(context)
                ], check=True, capture_output=True, timeout=60)
                for path in paths:
                    target = root / 'output' / path / 'new_cases'
                    self.assertTrue((target / 'test_unregistered.py').is_file(),
                                    str(path))
                    self.assertFalse((target / '.env').exists(), str(path))
                for module in artifact_modules:
                    target = root / 'output' / core_path / module
                    self.assertTrue(target.is_file(), str(module))
                for manifest in manifests:
                    self.assertTrue((root / 'output' / manifest).is_file(),
                                    str(manifest))

    @unittest.skipUnless(shutil.which('docker'), 'Docker is required')
    def test_wheel_rebuild_replaces_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / 'context'
            context.mkdir()
            (context / 'Dockerfile').write_text(
                'FROM scratch\nCOPY diskgraph-*.whl /\n')
            destination = root / 'wheel'
            execute = subprocess.run

            def export(command, **kwargs):
                # Use the real exporter without compiling DiskGraph.
                command = command.copy()
                command[command.index('--file') + 1] = str(context / 'Dockerfile')
                command[-1] = str(context)
                return execute(command, capture_output=True, timeout=60,
                               **kwargs)

            with patch.object(runner.subprocess, 'check_output', return_value=''), \
                    patch.object(runner.subprocess, 'run', side_effect=export):
                for version in ('old', 'new'):
                    source = context / f'diskgraph-{version}.whl'
                    source.write_text(version)
                    runner.build_wheel(destination)
                    self.assertEqual(list(destination.iterdir()),
                                     [destination / source.name])
                    self.assertEqual((destination / source.name).read_text(),
                                     version)
                    source.unlink()
            self.assertEqual(set(root.iterdir()), {context, destination})

    def test_failed_wheel_build_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'wheel'
            destination.mkdir()
            wheel = destination / 'diskgraph-example.whl'
            wheel.write_text('previous successful build')

            def fail_export(command, **kwargs):
                output = Path(command[command.index('--output') + 1].split(
                    'dest=', 1)[1])
                output.mkdir(parents=True, exist_ok=True)
                (output / wheel.name).write_text('partial failed build')
                raise subprocess.CalledProcessError(1, command)

            with patch.object(runner.subprocess, 'check_output', return_value=''), \
                    patch.object(runner.subprocess, 'run', side_effect=fail_export):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.build_wheel(destination)
            self.assertEqual(wheel.read_text(), 'previous successful build')
            self.assertEqual(list(root.iterdir()), [destination])

    def test_build_and_reuse_wheel_options_are_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['test_public.py', '--reports', directory, '--build-wheel', '--wheel-dir', directory]), \
                    patch.object(runner.subprocess, 'check_output', return_value='x86_64'), \
                    patch.object(runner, 'build_wheel') as build, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                build.assert_not_called()

    def test_setup_failure_clears_only_selected_reports(self):
        for build, group, flags in (
            ('build_image', 'csv', []),
            ('build_wheel', 'local-sdk', []),
            ('build_wheel', 'local-sdk', ['--build-wheel']),
        ):
            with self.subTest(build=build, flags=flags), \
                    tempfile.TemporaryDirectory() as directory:
                reports = Path(directory)
                old_reports = [
                    reports / name
                    for name in ('results.json', f'{group}.xml', f'{group}.log')
                ]
                for path in old_reports:
                    path.write_text('previous successful run')
                unrelated = reports / 'parquet.xml'
                unrelated.write_text('another group')
                wheel = reports / 'wheel'
                wheel.mkdir()
                (wheel / 'diskgraph-example.whl').touch()

                def fail_build(*args):
                    for path in old_reports:
                        self.assertEqual(path.exists(), bool(flags), path.name)
                    raise subprocess.CalledProcessError(1, 'build')

                with patch.object(sys, 'argv', ['test_public.py', '--group', group, '--reports', directory, *flags]), \
                        patch.object(runner.subprocess, 'check_output', return_value='x86_64'), \
                        patch.object(runner.subprocess, 'run'), \
                        patch.object(runner, build, side_effect=fail_build), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if build == 'build_wheel':
                        with self.assertRaises(subprocess.CalledProcessError):
                            runner.main()
                    else:
                        self.assertEqual(runner.main(), 1)
                self.assertEqual(unrelated.read_text(), 'another group')
                self.assertTrue((wheel / 'diskgraph-example.whl').exists())

    def test_reports_keep_private_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            reports.chmod(0o700)
            with patch.object(sys, 'argv', ['test_public.py', '--group', 'csv', '--reports', directory]), \
                    patch.object(runner.subprocess, 'check_output', return_value='revision'), \
                    patch.object(runner.subprocess, 'run'), \
                    patch.object(runner, 'build_image'), \
                    patch.object(runner, 'image_environment', return_value={}), \
                    patch.object(runner, 'run_group', return_value={'exit_code': 0, 'passed': 1}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(reports.stat().st_mode & 0o777, 0o700)

    def test_container_cannot_supply_a_report_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory) / 'reports'
            reports.mkdir()
            outside = Path(directory) / 'outside.xml'
            original = '<testsuites><testsuite tests="1"/></testsuites>'
            outside.write_text(original)
            mounts = []

            def execute(command, **kwargs):
                if command[:2] == ['docker', 'run']:
                    output = Path(command[command.index('-v') +
                                          1].split(':')[0])
                    mounts.append(output)
                    (output / 'csv.xml').symlink_to(outside)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(runner.subprocess, 'run', side_effect=execute), \
                    contextlib.redirect_stdout(io.StringIO()):
                row = runner.run_group('csv', 'image', reports, [], '',
                                       'test-owned')
            self.assertNotEqual(row['exit_code'], 0)
            self.assertNotIn(reports, mounts)
            self.assertEqual(outside.read_text(), original)
            self.assertFalse((reports / 'csv.xml').exists())

    def test_all_groups_run_after_a_test_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            wheel = reports / 'wheel'
            wheel.mkdir()
            (wheel / 'diskgraph-example.whl').touch()
            rows = [{
                'exit_code': 1,
                'passed': 1
            }] + [{
                'exit_code': 0,
                'passed': 1
            } for _ in range(len(runner.GROUPS) - 1)]
            previous = signal.getsignal(signal.SIGTERM)
            self.addCleanup(signal.signal, signal.SIGTERM, previous)
            with patch.object(sys, 'argv', ['test_public.py', '--reports', directory]), \
                    patch.object(runner.subprocess, 'check_output', return_value='x86_64'), \
                    patch.object(runner.subprocess, 'run'), \
                    patch.object(runner, 'build_wheel'), \
                    patch.object(runner, 'build_image') as build, \
                    patch.object(runner, 'image_environment', return_value={}), \
                    patch.object(runner, 'run_group', side_effect=rows) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 1)
            self.assertEqual([call.args[0] for call in run.call_args_list],
                             list(runner.GROUPS))
            self.assertEqual([call.args[0] for call in build.call_args_list],
                             ['connectors', 'sdk', 'full'])

    def test_build_failure_does_not_stop_other_images(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / 'wheel'
            wheel.mkdir()
            (wheel / 'diskgraph-example.whl').touch()

            def build(profile, *args):
                if profile == 'sdk':
                    raise subprocess.CalledProcessError(1, 'docker build')

            with patch.object(sys, 'argv', ['test_public.py', '--reports', directory]), \
                    patch.object(runner.subprocess, 'check_output', return_value='x86_64'), \
                    patch.object(runner.subprocess, 'run'), \
                    patch.object(runner, 'build_wheel'), \
                    patch.object(runner, 'build_image', side_effect=build), \
                    patch.object(runner, 'image_environment', return_value={}), \
                    patch.object(runner, 'run_group', return_value={'exit_code': 0, 'passed': 1}) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 1)
            self.assertEqual(
                [call.args[0] for call in run.call_args_list],
                [group for group in runner.GROUPS if group != 'hosted-sdk'])

    def test_timeout_and_cancel_both_remove_minio(self):
        for error in (subprocess.TimeoutExpired('docker run',
                                                1), KeyboardInterrupt()):
            with self.subTest(error=type(
                    error)), tempfile.TemporaryDirectory() as directory:
                commands = []

                def execute(command, **kwargs):
                    commands.append(command)
                    if command[0] == 'openssl':
                        Path(directory, 'public.crt').touch(mode=0o600)
                    if command[:2] == ['docker', 'run']:
                        raise error
                    return subprocess.CompletedProcess(command, 0)

                compose = ['docker', 'compose', '-p', 'test-owned']
                with patch.object(runner.subprocess, 'run', side_effect=execute), \
                        patch.object(runner.subprocess, 'check_output', return_value='minio-id'), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if isinstance(error, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            runner.run_group('minio', 'image', Path(directory),
                                             compose, directory, 'test-owned')
                    else:
                        row = runner.run_group('minio', 'image',
                                               Path(directory), compose,
                                               directory, 'test-owned')
                        self.assertEqual(row['exit_code'], 1)
                self.assertIn(['docker', 'rm', '-f', 'test-owned-minio'],
                              commands)
                self.assertIn(compose + ['down', '-v'], commands)
                command = next(c for c in commands
                               if c[:2] == ['docker', 'run'])
                self.assertEqual(command[command.index('--network') + 1],
                                 'container:minio-id')

    def test_plain_tests_have_no_network_or_host_credentials(self):
        for group in ('csv', 'mixed-files'):
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(runner.subprocess, 'run',
                                  return_value=subprocess.CompletedProcess([], 0)) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    row = runner.run_group(group, 'image', Path(directory), [], '',
                                           'test-owned')
                command = run.call_args_list[0].args[0]
                self.assertEqual(command[command.index('--network') + 1], 'none')
                self.assertNotIn('-e', command)
                self.assertNotIn('--env-file', command)
                self.assertEqual(command.count('-v'), 1)
                mount = command[command.index('-v') + 1]
                self.assertTrue(mount.endswith(':/reports'))
                self.assertNotEqual(mount, f'{directory}:/reports')
                # A successful Docker exit without results must not count as a pass.
                self.assertFalse(row.get('passed'))
            if group == 'mixed-files':
                self.assertIn('test/io_utils/fsutil/local', command)
                self.assertIn('--ignore=test/io_utils/fsutil/local/live', command)


if __name__ == '__main__':
    unittest.main()
