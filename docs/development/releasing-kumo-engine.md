# Releasing kumo-engine

`kumo-engine` is published as one wheel that bundles `kumo-api`, `kumo-pql`,
`kumo-core`, DiskGraph and every provider package (`kumo-snowflake`,
`kumo-databricks`, ...).
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

`kumo-api`, `kumo-pql`, `kumo-core` and every provider are always bundled.
Kumo-owned source uses Apache-2.0; bundled third-party source and native
components retain their obligations in `THIRD-PARTY.txt`.
`kumo-glove-6b-50d` stays a separate PyPI dependency through the `text` extra;
`merge_wheels.py` refuses to bundle it. It holds the GloVe word vectors Kumo
uses by default to encode text columns; without it, a text column cannot be
encoded and the error asks for `pip install "kumo-engine[text]"`.
`torch-geometric>=2.8.1,<3` is also resolved separately at install time.

Releases are cut from `NVIDIA/kumo-engine`: its GitHub releases trigger the
workflows below, which attach the wheels to the release. Their `kitmaker` job
then has Kitmaker, NVIDIA's release service, fetch the attached wheel and
publish it to PyPI and pypi.nvidia.com. No workflow here uploads to PyPI or
holds PyPI credentials.

The wheel is `cp311-abi3-manylinux_2_34_x86_64`: Linux x86_64, glibc >= 2.34,
CPython >= 3.11 (DiskGraph's stable-ABI floor). The merger reads every bundled
wheel's `Requires-Python` and writes their intersection into the final wheel.
Stricter minimums, upper bounds, and exclusions are retained; incompatible
constraints fail the build. Constraints are also recorded in
`kumo_bundle_manifest.json`. Native ABI tags are checked against the resulting
Python minimum.

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
   a `kumo-glove-6b-50d` in the range `packages/kumo-engine/pyproject.toml`
   allows (`>=1.0.0,<2`) for `[text]`. Release it first if it is new, by
   publishing a GitHub release tagged `kumo-glove-6b-50d-v<version>`
   ([its workflow][glove-workflow]), which publishes it the same way as step 5.
   A manual run of the engine workflow tells you whether it is there: its
   `check-requirements` job names every install PyPI cannot satisfy.
2. **Set the version** (`VERSION` plus `scripts/stamp_version.py`) on a branch
   and merge it. The tag must point at a commit that contains it.
3. **Publish a GitHub release** tagged `kumo-engine-v<version>` (for example
   `kumo-engine-v1.1.0`), targeting that commit. Tick *pre-release* for
   `b`/`rc`/`dev` versions. Creating the release publishes the tag; there is no
   separate tag push.
4. **Watch the workflow**, [Kumo Engine - Build and publish wheel][workflow]. It:
   - fails fast unless the tag is exactly `kumo-engine-v` plus the `VERSION`
     file and every stamped file agrees with it;
   - builds the wheels with `--release`, so the version is exactly the tag's;
   - resolves the x86_64 wheel against PyPI, alone and with each extra, on
     every Python version it claims, and stops if any fails;
   - attaches the x86_64 and ARM64 wheels to the GitHub release.
5. **Approve the publication.** The `kitmaker` job waits for approval on the
   `kumo-engine-release` environment. It then checks that the Portal project
   has the expected name and owner and that the attached wheel is the one this
   run built, has Kitmaker dry-run it, and only then publishes it to PyPI and
   pypi.nvidia.com, where it confirms both serve exactly those bytes. Only the
   x86_64 wheel is currently built and published. A publication cannot be recalled, and PyPI never
   accepts the same file name twice.
6. **Smoke-test the published wheel** in a clean virtualenv:

   ```bash
   pip install kumo-engine==<version>
   python -c "import kumo.engine as k; print(k.__version__); k.FileConnector"
   pip install 'kumo-engine[snowflake]==<version>'
   ```

Other releases in the repository (for example `kumo-sdk-v*`) do not trigger
this workflow.

[workflow]: ../../.github/workflows/kumo-engine-wheel.yml
[glove-workflow]: ../../.github/workflows/kumo-glove-6b-50d-wheel.yml

## If a release fails

| Failure | Fix |
| --- | --- |
| Tag does not match `VERSION` | Delete the release and tag, fix the version (or the tag), publish again. Nothing was uploaded. |
| `check-requirements` fails | A requirement is not on PyPI yet, usually `kumo-glove-6b-50d`. Publish it, then re-run the failed jobs. Nothing was attached or published. |
| Build or glibc-floor check fails | Fix on the branch, delete the release and tag, then recreate the tag on the fixed commit. Nothing was uploaded. |
| `kitmaker` fails before *publication* starts | Nothing was published. Fix the cause and re-run the failed job. |
| `kitmaker` fails during *publication* | Check PyPI first: a file that reached it stays there. Re-run only if the file is missing. |
| A bad wheel reached PyPI | Release the next version. Do not reuse the version; removing a published file is reserved for a critical vulnerability or leak. |

## Manual builds

Running the workflow by hand (Actions, *Run workflow*) builds the wheel as
`<version>+<commit>` and keeps it as a workflow artifact for 30 days. It never
attaches or publishes anything: attaching runs only for release events, and PyPI
rejects local versions anyway.

## Build locally

```bash
packages/kumo-engine/scripts/build_wheel.sh --help
packages/kumo-engine/scripts/build_wheel.sh --offline-report   # <version>+<sha>
```

Needs `cargo` (and `cargo-zigbuild`, installed on demand), a current committed
`THIRD-PARTY.txt` (checked with `scripts/gen_third_party.py`), and network
access for component build dependencies. `--verify-image IMAGE` installs the
wheel into a Docker image and checks that the closure imports and that provider
dependencies are absent from a bare install.

## PyPI projects

`kumo-engine` and `kumo-glove-6b-50d` belong to NVIDIA's PyPI organization
and are published through Kitmaker Portal projects 6903 and 6904. Because
Kitmaker downloads the wheels from the GitHub release, the repository must be
public when a release is published. The `kumo-engine-release` environment
needs:

- required reviewers, since its approval is what publishes;
- the secret `KITMAKER_API_TOKEN`, a token for a Kitmaker service account
  owned by the Portal projects' owner;
- the variable `KITMAKER_PIC`, that owner's email address.

Review the license metadata: the wheel carries the `License-Expression` derived by `scripts/gen_third_party.py` and ships `LICENSE` and `THIRD-PARTY.txt`.

## Charon connectivity and validation

The engine and GloVe release jobs call
[Kitmaker Portal via Charon](../../.github/workflows/kitmaker-portal.yml).
It runs on GitHub-hosted Ubuntu, opens a Teleport application tunnel to Charon
Ferry, and authenticates each Portal request with a fresh GitHub OIDC token
plus the Kitmaker bearer token. Portal credentials stay in the
`kumo-engine-release` environment. Charon/Teleport must authorize this repository,
workflow, and ref; authorization for cuOpt does not grant access to Kumo.
The tunnel uses `ext-nv-prd-apps.teleport.sh:443`. The original cuOpt PR
used an older staging proxy; its replacement is documented in
[cuOpt PR #2161](https://github.com/NVIDIA/cuopt/pull/2161).

To test connectivity before merging, push the workflow and script changes on
`automate-kitmaker` or a `kitmaker/**` branch. That push runs the branch's YAML.
It first probes the status of the nil UUID, then reads project 6903 and
verifies its name and owner. It does not build wheels, create a Kitmaker
release, or publish packages. A successful check
confirms connectivity and project-read access, not permission to submit a
release or fetch a wheel.

Once the workflow exists on the default branch, the same check can be dispatched
against a selected branch:

```bash
gh workflow run kitmaker-portal.yml --repo NVIDIA/kumo-engine --ref <branch>
```

Branch checks also require Charon/Teleport to allow branch refs. A tag-only
policy must be extended before these checks can succeed. Release authorization
must cover `kumo-engine-v*` and `kumo-glove-6b-50d-v*` tags.

The reusable workflow defaults `upload` to false. Release callers explicitly
set it to true after the release environment permits the job. For a standalone
Kitmaker validation without publication, run the script with `--dry-run` and
its usual release arguments. Add `--ferry` and the local tunnel Portal URL when
using Charon; this needs GitHub Actions OIDC variables and `id-token: write`.
The script requires `orjson`, already declared through the root project's
dynamic dependencies in `setup.py`; the workflow installs it in an isolated
`uv run` environment.

### Connectivity failures

`token expired or not found` from the original `nv-stg-ps` proxy can indicate
an outdated proxy, rather than an expired Kitmaker API token. Use the current
Teleport proxy before investigating Portal credentials.

A successful Teleport tunnel does not establish Charon backend authorization.
For HTTP 403, ask the Charon owner to check this repository's tenant grant
against the run's actual ref and requested routes. The branch check needs
`refs/heads/automate-kitmaker` (or the selected test branch),
`GET /api/v0/status/*`, and `GET /api/v0/projects/6903`. Release jobs also need
project reads for 6903 and 6904, POST access to each project's `/releases`, and
the corresponding release tag refs. Preserve the error response's request ID
for tracing. A denied read alone does not establish whether Portal credentials
are valid.
