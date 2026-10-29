#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the component wheels collected by build_wheel.sh into one wheel.

The merged wheel is named after the `kumo-engine` component and its metadata
is derived from that component's own requirements:

* Requirements reachable from kumo-engine's unconditional dependencies (and
  the extras it requests of bundled components, e.g. `kumo-core[gpu]`) become
  the wheel's plain `Requires-Dist`.
* Each extra kumo-engine declares (`snowflake`, `databricks`, ...) is kept as an
  extra. The bundled component it names ships its code in every install, but
  its third-party dependencies are only required with the extra, so
  `pip install kumo-engine` gets `kumo_snowflake` without
  `snowflake-connector-python`.

A bundled component that no dependency or extra of kumo-engine reaches is an
error: its dependencies would otherwise be dropped silently.
"""

import argparse
import base64
import configparser
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

try:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version
except ImportError:
    # pip is already required by the builder and carries packaging.
    from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
    from pip._vendor.packaging.version import InvalidVersion, Version

ENGINE = 'kumo-engine'
# Extras that exist for development only and are not part of the release.
DEV_EXTRAS = {'test'}
# Published on their own, so they must stay PyPI dependencies, never be merged.
NEVER_BUNDLED = {'kumo-glove-6b-50d'}
# Headers copied from kumo-engine's own metadata so the bundle's PyPI page is
# kumo-engine's.
PASSTHROUGH_HEADERS = ('Summary', 'Keywords', 'Author', 'Author-email',
                       'Description-Content-Type', 'Classifier', 'Project-URL')


def log(message):
    print(f'[wheel] {message}', file=sys.stderr, flush=True)


def fail(message):
    raise SystemExit(f'[wheel] ERROR: {message}')


def canonical(dist):
    return re.sub(r'[-_.]+', '-', dist).lower()


_EXTRA_TERM = r'''\(?\s*extra\s*==\s*["']([^"']+)["']\s*\)?'''
_REQUIREMENT = re.compile(r'^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*'
                          r'(?:\[(?P<extras>[^\]]*)\])?\s*'
                          r'(?P<rest>[^;]*?)\s*(?:;\s*(?P<marker>.*?))?\s*$')


def parse_requirement(requirement):
    """Split a Requires-Dist line.

    Returns (dist, extras, spec, extra, marker): `spec` is everything before
    the marker (name, extras and version/URL), `extra` the `extra == "x"` the
    requirement is conditional on, and `marker` the remaining environment
    marker.
    """
    match = _REQUIREMENT.match(requirement)
    if match is None:
        fail(f'cannot parse requirement {requirement!r}')
    extras = {
        canonical(e)
        for e in (match['extras'] or '').split(',') if e.strip()
    }
    spec = requirement.split(';', 1)[0].strip()
    marker = match['marker'] or None
    extra = None
    if marker:
        for pattern in (rf'\s+and\s+{_EXTRA_TERM}\s*$',
                        rf'^\s*{_EXTRA_TERM}\s+and\s+',
                        rf'^\s*{_EXTRA_TERM}\s*$'):
            found = re.search(pattern, marker)
            if found:
                extra = canonical(found.group(1))
                marker = (marker[:found.start()] +
                          marker[found.end():]).strip() or None
                break
        else:
            if re.search(r'\bextra\b', marker):
                fail(f'unsupported marker in {requirement!r}')
    return match['name'], extras, spec, extra, marker


def with_marker(spec, marker, extra=None):
    terms = [f'({marker})' if marker and extra else marker] if marker else []
    if extra:
        terms.append(f'extra == "{extra}"')
    return f'{spec}; {" and ".join(terms)}' if terms else spec


def wheel_tags(wheel):
    """Parse the compressed tag set out of a wheel filename."""
    stem = wheel.name[:-len('.whl')]
    python, abi, platform = stem.rsplit('-', 3)[1:]
    return [(py, ab, pl) for py in python.split('.') for ab in abi.split('.')
            for pl in platform.split('.')]


def is_universal(wheel):
    return any(
        py.startswith('py3') and abi == 'none'
        for py, abi, _ in wheel_tags(wheel))


def native_python_tags(wheel, min_minor):
    """cpNN tags under which this abi3 wheel runs on Python 3.<min_minor>."""
    return [
        py for py, abi, _ in wheel_tags(wheel) if abi == 'abi3'
        and py.startswith('cp3') and int(py.removeprefix('cp3')) <= min_minor
    ]


def read_component(wheel, classification, origin):
    with zipfile.ZipFile(wheel) as archive:
        entry = next(n for n in archive.namelist()
                     if n.endswith('.dist-info/METADATA'))
        message = Parser().parsestr(archive.read(entry).decode('utf-8'))
    return {
        'wheel': wheel,
        'classification': classification,
        'origin': origin,
        'message': message,
        'name': message['Name'],
        'version': message['Version'],
        'requires': message.get_all('Requires-Dist') or [],
        'extras': message.get_all('Provides-Extra') or [],
    }


def resolve_python_requirement(components):
    """Intersect component constraints and find the first allowed CPython 3 minor."""
    combined = SpecifierSet()
    candidates = {Version('3.0.0')}
    constraints = []
    for component in components:
        declared = component['message']['Requires-Python'] or ''
        log(f"{component['name']} Requires-Python: {declared or '(unspecified)'}")
        try:
            specifiers = SpecifierSet(declared)
        except InvalidSpecifier as error:
            fail(f"{component['name']} has invalid Requires-Python "
                 f"{declared!r}: {error}")
        combined &= specifiers
        if declared:
            constraints.append(f"{component['name']}: {declared}")
        for specifier in specifiers:
            try:
                version = Version(specifier.version.removesuffix('.*'))
            except InvalidVersion:
                fail(f"{component['name']} has a non-version Requires-Python "
                     f"constraint: {specifier}")
            major, minor, patch = (version.release + (0, 0))[:3]
            # Stable interpreter versions can enter/leave a specifier range
            # at its boundary, the next patch, or the next minor (wildcards).
            for release in ((major, minor, patch), (major, minor, patch + 1),
                            (major, minor + 1, 0)):
                if release[0] == 3:
                    candidates.add(Version('.'.join(map(str, release))))
    compatible = sorted(version for version in candidates
                        if combined.contains(version, prereleases=False))
    if not compatible:
        fail('bundled Requires-Python constraints allow no common stable '
             'CPython 3 version:\n    ' + '\n    '.join(constraints))
    minimums = [specifier for specifier in combined
                if specifier.operator == '>=']
    if minimums:
        strongest = max(minimums, key=lambda specifier: Version(specifier.version))
        combined = SpecifierSet(','.join(
            str(specifier) for specifier in combined
            if specifier.operator != '>=' or specifier == strongest))
    log(f'combined Requires-Python: {combined}')
    return str(combined), compatible[0].release[1]


def resolve_requirements(components):
    """Split the engine's dependency closure into base and per-extra lists.

    Returns (base, extras, reached): `base` and each `extras[name]` are lists
    of requirement strings (`extras[name]` without its marker), `reached` the
    canonical names of the bundled components they cover.
    """
    by_name = {canonical(c['name']): c for c in components}
    engine = by_name.get(canonical(ENGINE))
    if engine is None:
        fail(f'no {ENGINE} component to take the bundle metadata from')
    reached = set()

    def expand(component, extras, visited):
        """Requirements `component` adds beyond what `visited` already covers.

        `visited` maps each component to the extras already expanded, so a
        component the base install covers is not repeated under an extra.
        """
        name = canonical(component['name'])
        first = name not in visited
        expanded = visited.setdefault(name, set())
        new = extras - expanded
        if not first and not new:
            return []
        expanded |= extras
        reached.add(name)
        found = []
        for requirement in component['requires']:
            dist, ref_extras, spec, extra, marker = parse_requirement(
                requirement)
            if extra is None and not first:
                continue
            if extra is not None and extra not in new:
                continue
            target = by_name.get(canonical(dist))
            if target is not None:
                found += expand(target, ref_extras, visited)
            elif '@' in spec:
                # A direct URL for anything the bundle doesn't carry would
                # break an index-only install.
                log(f'dropping direct-URL requirement: {requirement}')
            else:
                found.append((spec, marker))
        return found

    base_visited = {}
    base = expand(engine, set(), base_visited)

    extras = {}
    for extra in engine['extras']:
        if extra in DEV_EXTRAS:
            continue
        visited = {k: set(v) for k, v in base_visited.items()}
        found, skipped = [], False
        for requirement in engine['requires']:
            dist, ref_extras, spec, requested, marker = parse_requirement(
                requirement)
            if requested != canonical(extra):
                continue
            target = by_name.get(canonical(dist))
            if target is not None:
                found += expand(target, ref_extras, visited)
            elif (canonical(dist).startswith('kumo-')
                  and canonical(dist) not in NEVER_BUNDLED):
                # An extra naming a provider that is not bundled would point
                # at a first-party package that is not on any index (unlike
                # NEVER_BUNDLED, which are published): drop the extra.
                log(f'dropping extra {extra!r}: {dist} is not bundled')
                skipped = True
            else:
                found.append((spec, marker))
        if not skipped:
            extras[extra] = found

    unreachable = sorted(set(by_name) - reached)
    if unreachable:
        fail(
            f'{", ".join(unreachable)} are bundled but not required by '
            f'{ENGINE} or any of its extras, so their dependencies would '
            f'be dropped. Add the extra to packages/kumo-engine/pyproject.toml.'
        )

    def dedupe(pairs, extra=None):
        out = []
        for spec, marker in pairs:
            line = with_marker(spec, marker, extra)
            if line not in out:
                out.append(line)
        return out

    return (dedupe(base), {
        name: dedupe(pairs, canonical(name))
        for name, pairs in extras.items()
    }, reached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheels_dir', type=Path)
    parser.add_argument(
        'components_tsv', type=Path,
        help='one `<wheel filename>\\t<classification>\\t'
        '<origin>` row each')
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--name', default=ENGINE)
    parser.add_argument('--version', required=True)
    parser.add_argument(
        '--platform-tag', required=True,
        help='platform tag for the native components, '
        'e.g. manylinux_2_34_x86_64')
    parser.add_argument('--notice', type=Path, required=True,
                        help='third-party notice to ship')
    parser.add_argument('--license', type=Path, required=True,
                        help='first-party license to ship')
    parser.add_argument('--license-expression', required=True,
                        help='SPDX expression for the whole wheel')
    parser.add_argument('--offline-report', action='store_true')
    args = parser.parse_args()
    name, version = args.name, args.version
    wheels_dir = args.wheels_dir

    components = []
    for row in args.components_tsv.read_text(encoding='utf-8').splitlines():
        if row.strip():
            filename, classification, origin = row.split('\t', 2)
            components.append(
                read_component(wheels_dir / filename, classification, origin))

    bundled_forbidden = sorted(c['name'] for c in components
                               if canonical(c['name']) in NEVER_BUNDLED)
    if bundled_forbidden:
        fail(f'{", ".join(bundled_forbidden)} must not be bundled; they are '
             f'PyPI dependencies of {ENGINE}.')
    engine = next(
        (c for c in components if canonical(c['name']) == canonical(ENGINE)),
        None)
    if engine is None:
        fail(f'no {ENGINE} component')
    if not engine['message']['Requires-Python']:
        fail(f'{ENGINE} must declare Requires-Python')
    requires_python, min_minor = resolve_python_requirement(components)

    # Refuse to ship a native wheel that cannot load on every interpreter the
    # bundle claims to support: one abi3 wheel built for <= the floor serves
    # them all.
    native = [c for c in components if not is_universal(c['wheel'])]
    unusable = [
        c for c in native if not native_python_tags(c['wheel'], min_minor)
    ]
    if unusable:
        listing = '\n'.join(f"    {c['name']}: {c['wheel'].name}"
                            for c in unusable)
        fail(f'these components are not py3-none-any or abi3 wheels built '
             f'for Python 3.{min_minor} or older, so one wheel cannot serve '
             f'{requires_python}:\n{listing}')
    if native:
        python_tag = max((t for c in native
                          for t in native_python_tags(c['wheel'], min_minor)),
                         key=lambda t: int(t.removeprefix('cp3')))
        tag = f'{python_tag}-abi3-{args.platform_tag}'
    else:
        tag = 'py3-none-any'

    base, extras, _ = resolve_requirements(components)

    # Unpack every component into one tree, refusing silent overwrites. A
    # component's own .dist-info is dropped -- the bundle ships one of its own --
    # and its provenance is kept in kumo_bundle_manifest.json instead.
    staging = Path(
        tempfile.mkdtemp(prefix='kumo-wheel-staging-',
                         dir=str(wheels_dir.parent)))
    bundle_data = f"{name.replace('-', '_')}-{version}.data"
    owners = {}
    entry_points = configparser.ConfigParser(interpolation=None,
                                             delimiters=('=', ))
    entry_points.optionxform = str
    for component in components:
        with zipfile.ZipFile(component['wheel']) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                target_name = info.filename
                if re.match(r'[^/]+\.dist-info/entry_points\.txt$',
                            target_name):
                    # Plugins are found through entry points, so keep them.
                    found = configparser.ConfigParser(interpolation=None,
                                                      delimiters=('=', ))
                    found.optionxform = str
                    found.read_string(archive.read(info).decode('utf-8'))
                    for group in found.sections():
                        if not entry_points.has_section(group):
                            entry_points.add_section(group)
                        for key, value in found.items(group):
                            if entry_points.get(group, key,
                                                fallback=value) != value:
                                fail(f'entry point {group}:{key} is '
                                     f'declared twice, differently')
                            entry_points.set(group, key, value)
                    continue
                if re.match(r'[^/]+\.dist-info/', target_name):
                    continue
                # A component's .data/ tree is keyed to that component's own
                # name-version, which the installer will not recognise under
                # the bundle's dist-info. Re-key it to the bundle's.
                data_match = re.match(r'[^/]+\.data/(.*)', target_name)
                if data_match:
                    target_name = f'{bundle_data}/{data_match.group(1)}'
                previous = owners.get(target_name)
                if previous is not None and previous != component['name']:
                    fail(f"{target_name} is provided by both {previous} and "
                         f"{component['name']}; refusing to merge")
                owners[target_name] = component['name']
                destination = staging / target_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
                # The zip's executable bit matters: DiskGraph's bundled ingest
                # binary is spawned as a subprocess.
                mode = info.external_attr >> 16
                if mode & 0o111:
                    destination.chmod(destination.stat().st_mode | 0o111)
        log(f"merged {component['name']} {component['version']}")

    # A development bundle adds its commit to the component's stamped version.
    # Keep the installed module consistent with the assembled wheel metadata.
    version_module = staging / 'kumo/engine/_version.py'
    if version_module.is_file():
        source, count = re.subn(r'(?m)^__version__\s*=.*$',
                                f'__version__ = {version!r}',
                                version_module.read_text(encoding='utf-8'))
        if count != 1:
            fail('expected one stamped kumo.engine version')
        version_module.write_text(source, encoding='utf-8')

    dist_info = staging / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    if entry_points.sections():
        with open(dist_info / 'entry_points.txt', 'w',
                  encoding='utf-8') as handle:
            entry_points.write(handle)

    description = engine['message'].get_payload() or ''
    metadata = [
        # 2.4 is what defines License-File.
        'Metadata-Version: 2.4',
        f'Name: {name}',
        f'Version: {version}',
        f'Requires-Python: {requires_python}',
        f'License-Expression: {args.license_expression}',
        'License-File: LICENSE',
        'License-File: THIRD-PARTY.txt',
    ]
    for header in PASSTHROUGH_HEADERS:
        metadata += [
            f'{header}: {value}'
            for value in engine['message'].get_all(header) or []
        ]
    metadata += [f'Provides-Extra: {extra}' for extra in extras]
    metadata += [f'Requires-Dist: {requirement}' for requirement in base]
    for requirements in extras.values():
        metadata += [
            f'Requires-Dist: {requirement}' for requirement in requirements
        ]
    metadata += ['', description]
    (dist_info / 'METADATA').write_text('\n'.join(metadata) + '\n',
                                        encoding='utf-8')
    for label, path in (('first-party license', args.license),
                        ('third-party notice', args.notice)):
        if not path.is_file():
            fail(f'no {label} at {path}. The merge drops every component '
                 f'.dist-info, so without it the wheel ships no license '
                 f'text at all.')
    (dist_info / 'licenses').mkdir(parents=True, exist_ok=True)
    for target, path in (('LICENSE', args.license), ('THIRD-PARTY.txt',
                                                     args.notice)):
        (dist_info / 'licenses' / target).write_bytes(path.read_bytes())
        log(f'included {target} ({path.stat().st_size // 1024} KiB)')

    (dist_info / 'WHEEL').write_text(
        'Wheel-Version: 1.0\n'
        'Generator: merge_wheels.py\n'
        'Root-Is-Purelib: false\n'
        f'Tag: {tag}\n', encoding='utf-8')
    (staging / 'kumo_bundle_manifest.json').write_text(
        json.dumps(
            {
                'bundle': {
                    'name': name,
                    'version': version,
                    'tag': tag,
                    'requires_python': requires_python,
                },
                'components': [{
                    'name': c['name'],
                    'version': c['version'],
                    'requires_python': c['message']['Requires-Python'],
                    'classification': c['classification'],
                    'origin': c['origin'],
                } for c in components],
                'requires':
                base,
                'extras': {
                    k: v
                    for k, v in extras.items()
                },
            }, indent=2) + '\n', encoding='utf-8')

    # Zip the staging tree, writing RECORD last: it lists every other file.
    output = args.out_dir / f"{name.replace('-', '_')}-{version}-{tag}.whl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in staging.rglob('*') if p.is_file()):
            relative = path.relative_to(staging).as_posix()
            data = path.read_bytes()
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
            rows.append([relative, f'sha256={digest}', str(len(data))])
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        record_name = (dist_info.relative_to(staging) / 'RECORD').as_posix()
        rows.append([record_name, '', ''])
        buffer = tempfile.NamedTemporaryFile('w', newline='', suffix='.csv',
                                             delete=False)
        with buffer:
            csv.writer(buffer).writerows(rows)
        archive.write(buffer.name, record_name)
        os.unlink(buffer.name)

    log('bundled: ' + ', '.join(f"{c['name']} {c['version']}"
                                for c in components))
    if args.offline_report:
        log('still resolved from an index at install time:')
        for requirement in base:
            print(f'    {requirement}', file=sys.stderr)
        for extra, requirements in extras.items():
            print(f'  [{extra}]', file=sys.stderr)
            for requirement in requirements:
                print(f'    {requirement}', file=sys.stderr)
    print(output)


if __name__ == '__main__':
    main()
