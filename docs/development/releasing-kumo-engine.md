# Releasing kumo-engine

`kumo-engine` is published as one wheel that bundles `kumo-core`, `kumo-pql`,
DiskGraph and every provider package (`kumo-snowflake`, `kumo-databricks`, ...).
Provider *code* ships in every install; their third-party dependencies come
with the matching extra:

```
pip install kumo-engine               # FileConnector works; SnowflakeConnector
                                      # raises ImportError naming the extra
pip install 'kumo-engine[snowflake]'  # adds snowflake-connector-python, ...
```

The extras are the ones declared in
[`packages/kumo-engine/pyproject.toml`](../../packages/kumo-engine/pyproject.toml);
`scripts/merge_wheels.py` turns each into the dependencies of the provider it
names and fails if a bundled package is not reachable from any of them.
Provider code must be imported at the point of use, never at module top level,
so a bare install can still `import kumo.engine`
(`packages/kumo-engine/test/test_optional_imports.py` enforces this).

`kumo-api` and `kumo-glove-6b-50d` are published separately and are ordinary
PyPI dependencies (`kumo-glove-6b-50d` through the `text` extra). The wheel never
bundles them, and `merge_wheels.py` refuses to.

The wheel is `cp311-abi3-manylinux_2_34_x86_64`: Linux x86_64, glibc >= 2.34,
CPython >= 3.11 (DiskGraph's stable-ABI floor).

`scripts/build_fat_wheel.sh` and its manual workflow, *Kumo SDK - Build fat
wheel*, are unchanged: they still build the date-stamped `kumo-fat-bundle`
wheel for Snowflake and Databricks notebooks. They are unrelated to the
published `kumo-engine` wheel described here.

## Versioning

The root [`VERSION`](../../VERSION) file is the only place the version is
written, and it is shared by kumo-engine, kumo-sdk, the platform, the Studio
images and the Helm charts. `scripts/stamp_version.py` copies it into
`kumo.engine._version` (which becomes the engine wheel's metadata version),
`kumoai_studio/_version.py` (the SDK wheel's version), the platform's
`common_utils/version_file.py` and the charts; CI checks they agree.

To change it, edit `VERSION` and run `python3 scripts/stamp_version.py`, or use
`packages/kumo-sdk/scripts/set_version.py <version>` (what the SDK release
workflows do). Versions are `X.Y.Z`, `X.Y.ZbN`, `X.Y.ZrcN` or `X.Y.Z.devN`, so
an engine release is cut at the same version as the SDK's. PyPI never accepts a
re-upload of a version, so a bad release is fixed by releasing the next
version.

`kumo.engine.__version__` is the stamped value.

`kumo_core.version.get_kumo_version()` asks version providers, registered in the
`kumo_core.version_providers` entry-point group, in entry-point name order and
uses the first answer; then the git revision of a source checkout; then
`latest`. `kumo_core` only defines the mechanism; the providers belong to the
distributions that register them:

- the platform (root `setup.py`) registers `container-file`
  (`common_utils.version_file`), the image version in `~/kumo_version.txt`, and
  `release-version`, the stamped `VERSION`, for an installed platform with no
  version file;
- kumo-engine registers `kumo-engine` (`kumo_engine_version`, a top-level module
  so loading it does not import the engine), the installed wheel's version.

Editable installs defer, so a source checkout still reports its git revision.
Names decide precedence, so the image version wins, then the engine's, then the
platform's `release-version`.

## Cut a release

1. **Check the dependencies are on PyPI.** The wheel requires
   `kumo-api==<packages/kumo-api version>` and, for `[text]`,
   `kumo-glove-6b-50d==1.0.0`. Release them first if they are new.
2. **Set the version** (`VERSION` plus `scripts/stamp_version.py`) on a branch
   and merge it. The tag must point at a commit that contains it.
3. **Publish a GitHub release** tagged `kumo-engine-v<version>` (for example
   `kumo-engine-v1.1.0`), targeting that commit. Tick *pre-release* for
   `b`/`rc`/`dev` versions. Creating the release publishes the tag; there is no
   separate tag push.
4. **Watch the workflow**, [Kumo Engine - Build and publish wheel][workflow]. It:
   - fails fast unless the tag is exactly `kumo-engine-v` plus the `VERSION`
     file and every stamped file agrees with it;
   - builds the wheel with `--release`, so the version is exactly the tag's;
   - attaches the wheel to the GitHub release;
   - uploads it to PyPI (only when `KUMO_ENGINE_PUBLICATION_ENABLED` is
     `true`; see below).
5. **Smoke-test the published wheel** in a clean virtualenv:

   ```bash
   pip install kumo-engine==<version>
   python -c "import kumo.engine as k; print(k.__version__); k.FileConnector"
   pip install 'kumo-engine[snowflake]==<version>'
   ```

Other releases in the repository (for example `kumo-sdk-v*`) do not trigger
this workflow.

[workflow]: ../../.github/workflows/kumo-engine-wheel.yml

## If a release fails

| Failure | Fix |
| --- | --- |
| Tag does not match `VERSION` | Delete the release and tag, fix the version (or the tag), publish again. Nothing was uploaded. |
| Build or glibc-floor check fails | Fix on the branch, delete the release and tag, then recreate the tag on the fixed commit. Nothing was uploaded. |
| PyPI upload fails before the version appears | Re-run the failed job from the Actions page. |
| A bad wheel reached PyPI | Yank it on PyPI and release the next version. Do not reuse the version. |

## Manual builds

Running the workflow by hand (Actions, *Run workflow*) builds the wheel as
`<version>+<commit>` and keeps it as a workflow artifact for 30 days. It never
publishes: the publish job only runs for release events, and PyPI rejects local
versions anyway. Inputs:

- `extra_components`: restrict the optional providers (an extra whose provider
  is left out disappears from that wheel);

## Build locally

```bash
packages/kumo-engine/scripts/build_wheel.sh --help
packages/kumo-engine/scripts/build_wheel.sh --offline-report   # <version>+<sha>
```

Needs `cargo` (and `cargo-zigbuild`, installed on demand), a current committed
`THIRD-PARTY.txt` (checked with `scripts/gen_third_party.py`), and network
access for the torch-geometric wheel. `--verify-image IMAGE` installs the wheel
into a Docker image and checks that the closure imports and that provider
dependencies are absent from a bare install.

## One-time PyPI setup

Publication is off until the repository variable
`KUMO_ENGINE_PUBLICATION_ENABLED` is `true`. Before enabling it:

1. Create the `kumo-engine` project on PyPI.
2. Add a [trusted publisher](https://docs.pypi.org/trusted-publishers/) for
   this repository with workflow `kumo-engine-wheel.yml` and environment
   `kumo-engine-release`.
3. Create the `kumo-engine-release` environment in the repository settings
   (add required reviewers if releases should need approval).
4. Review the license metadata: the wheel carries the `License-Expression`
   derived by `scripts/gen_third_party.py` and ships `LICENSE`,
   `THIRD-PARTY.txt` and `KUMO-API-LICENSE`.
