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

Faiss is selected at install time: `faiss-gpu==1.14.3` on Linux x86-64 and
`faiss-cpu==1.14.3` on macOS. These markers live on the engine's direct
dependencies so merging bundled packages preserves them. This dependency
selection does not establish wheel or runtime support. Faiss is optional on
Windows and Linux ARM64; the `cpu` extra opts into CPU Faiss there.

Releases are cut from `NVIDIA/kumo-engine`: its GitHub releases trigger the
workflows below, which attach the wheels to the release. Their `kitmaker` job
then has Kitmaker, NVIDIA's release service, fetch the attached wheels and
publish them to PyPI and pypi.nvidia.com. No workflow here uploads to PyPI or
holds PyPI credentials.

The build matrix targets `cp311-abi3-manylinux_2_34_x86_64` and
`cp311-abi3-manylinux_2_34_aarch64` for Linux with glibc >= 2.34,
`cp311-abi3-win_amd64` for native Windows x86-64, and
`cp311-abi3-macosx_14_0_arm64` for Apple Silicon with macOS >= 14.0.
WSL uses the Linux x86-64 wheel. All require CPython >= 3.11
(DiskGraph's stable-ABI floor), with classifiers for 3.11–3.14. Adding build
targets does not establish published availability or installed-wheel runtime
qualification. Release attachment and publication currently allow only Linux
x86-64 and macOS ARM64. Linux ARM64 remains a CI artifact until installed-wheel
qualification is integrated; native Windows remains experimental and CI-only.
Windows release support uses WSL with the Linux x86-64 wheel. The merger reads
every bundled wheel's `Requires-Python` and writes their intersection into the
final wheel.
Stricter minimums, upper bounds, and exclusions are retained; incompatible
constraints fail the build. Constraints are also recorded in
`kumo_bundle_manifest.json`. Native ABI tags are checked against the resulting
Python minimum. Redundant `>=` bounds are collapsed to the strongest minimum
so PyPI shows `>=3.11` rather than `>=3.10,>=3.11`.

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
   - resolves each platform's wheel against PyPI, alone and with each extra, on
     every Python version it claims, and stops if any fails;
   - attaches only the Linux x86-64 and macOS ARM64 wheels to the GitHub
     release; ARM64 Linux and experimental native Windows builds stay in CI.
5. **Approve the publication.** The `kitmaker` job waits for approval on the
   `kumo-engine-release` environment. It then checks that the Portal project
   has the expected name and owner and that the attached wheel is the one this
   run built, has Kitmaker dry-run it, and only then publishes it to PyPI and
   pypi.nvidia.com, where it confirms both serve exactly those bytes. All enabled
   platform wheels are published. A publication cannot be recalled, and PyPI never
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

On Apple Silicon, the same script builds a native macOS wheel with
`MACOSX_DEPLOYMENT_TARGET=14.0` by default; overrides below 14.0 are rejected
to match the Faiss dependency. It disables Zig and glibc checks
and takes the final platform tag from the built DiskGraph wheel. Rust, Xcode
Command Line Tools, and CMake are required. The release workflow builds this
wheel on a macOS ARM64 runner with the same deployment target.

For experimental native Windows builds, run the builder in Git Bash with
native Windows Python, the Rust MSVC toolchain, CMake, and an activated
Visual Studio x64 C++/Windows SDK environment. Set `PYTHON=python` if `python3`
is not available. Zig and ELF/glibc checks are disabled; the builder requires
DiskGraph's `win_amd64`
tag. `--verify-image` is not supported for Windows: qualify the installed
wheel natively. The workflow activates the image's existing MSVC tools;
it does not install system toolchains.
The checkout must preserve committed LF line endings: notice and Cargo lock
hashes verify exact source bytes. For a fresh local clone, use
`git -c core.autocrlf=false clone <repository-url>`. CI applies this setting
only to the checkout action's Git processes, without changing repository or
user Git configuration files or weakening the integrity checks.

Linux ARM64 uses the same Zig build and glibc checks as Linux x86-64,
with `--platform-tag manylinux_2_34_aarch64`. The workflow validates the
assembled native payload before retaining it as a CI artifact. Clean
installed-wheel CPU ingestion, training, and prediction are a separate
qualification requirement; GPU execution is not required for these platform
wheels.

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
