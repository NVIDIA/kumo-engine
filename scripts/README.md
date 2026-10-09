# Repository scripts

This directory contains scripts for developer setup, CI checks, operational utilities, and releases. See [release tooling](./release/README.md) for the release-specific commands.

## Version stamping

kumo-engine, kumo-sdk, the platform, the Studio images, and the Helm charts share one version, kept in the root [`VERSION`](../VERSION) file as a PEP 440 string such as `1.1.0.dev0` (also `1.1.0`, `1.1.0b1` and `1.1.0rc1`). After changing it, run:

```bash
python3 scripts/stamp_version.py
```

The [script](./stamp_version.py) writes the version into every site that carries it: `kumo.engine._version` (the kumo-engine wheel's version), `kumoai_studio/_version.py` (the kumo-sdk wheel's version), `common_utils/version_file.py` (the platform's release version) and the Helm charts, using the SemVer form (`1.1.0-dev.0`, `1.1.0-rc.1`) for Helm. CI runs `python3 scripts/stamp_version.py --check`, which fails when any site differs from `VERSION`.

The kumo-sdk release workflows change the version through [`packages/kumo-sdk/scripts/set_version.py`](../packages/kumo-sdk/scripts/set_version.py), which writes `VERSION` and runs the stamp, so a release branch's `rc`/final versions and the next `.dev0` on master are shared by every artifact. See [releasing kumo-engine](../docs/development/releasing-kumo-engine.md).

## Python dead-code check

Run the check from a checkout with the repository tools installed:

```bash
./scripts/check_dead_code.sh
```

The [wrapper](./check_dead_code.sh) runs Vulture 2.14 once against the current Python source tree. The [Vulture configuration](../pyproject.toml) sets a minimum confidence of 80% and excludes generated grammar, tests, examples, third-party code, the imported DiskGraph package, agent files, and local environments. The DiskGraph mirror must remain identical to its source, so cleanup belongs in that source before a mirror refresh. This matches the other Python linters' DiskGraph exclusion. Imports are reported at 90%; unused arguments and unreachable code are reported at 100%. Most unused functions, classes, and local variables are below the current threshold.

Every finding in that configured scope fails the check, including findings in files a pull request did not edit. For example, deleting the last use of an import in one file can make an import in another, unchanged file unused. The whole tree is analyzed together so unchanged callers are also visible. There is no stored findings baseline and no `--diff` mode.

### Handling findings

Inspect each finding before changing code:

- Remove unused code and update its callers when it has no required interface role.
- Preserve required callback, workflow, and library signatures. Explicitly discard unused arguments with `del` and explain the compatibility requirement.
- Keep imports required by quoted type annotations. If Vulture cannot recognize that use, add a narrow `# noqa: F401` with a comment explaining the annotation.
- For protocol declarations whose parameter names must match an implementation, use a narrow `# noqa: F841` on the declaration and explain that requirement.

These exceptions are recorded beside the affected code. Avoid global name ignores or suppressing a finding solely because it already existed. See [Vulture's suppression documentation](https://github.com/jendrikseipp/vulture#handling-false-positives) for the supported mechanisms. Rerun the full check after cleanup.

The wrapper exits with Vulture's status: 0 means no findings, 3 means dead-code findings, and 1 or 2 indicates an analysis or configuration error. It rejects arguments rather than allowing a caller to narrow the configured scan.

### Continuous integration and release audit

[Routine CI integration](https://github.com/kumo-ai/kumo/pull/30947) invokes this same command without a commit range. Public-reference and credential checks retain their own scope settings.

[The release audit](https://github.com/kumo-ai/kumo/pull/31027) invokes the same full-tree dead-code check alongside its other publication checks. Its report helps review findings; it does not suppress them or change routine CI policy.

### Focused verification

```bash
mise exec -- uvx --with vulture==2.14 --with pytest python -m pytest \
    -q --confcutdir=test/scripts test/scripts/test_check_dead_code.py
```

The tests exercise cross-file usage, removal of the last caller in another file, configured exclusions, narrow exceptions, and wrapper exit behavior.
