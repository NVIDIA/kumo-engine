# Hotfix & Release Guidelines

This directory contains release, backport, image-build, and artifact helpers.
Read the selected script before invoking it: these commands can create
branches/tags, publish artifacts, and change remote state. Obtain the release
owner's approval for the target version and publication destinations before
running a live release. These commands are not prerequisites for local
development or documentation validation.

For PR preparation and change-map requirements, read the
[review guide](../../docs/development/review.md).

## Publish images from NVIDIA/kumo-engine

The [image-set workflow](../../.github/workflows/_docker-kumo3-images-custom.yml)
runs from `main`. It builds and promotes CPU, GPU, and UI images together.
Use a full commit SHA to publish an immutable image set.

Before publication, an administrator must configure these GitHub environments
and their scoped AWS roles:

- `gha-kumo3-image-set-ecr-read`
- `gha-kumo3-image-set-promote`
- `gha-kumo-backend-cpu-publish`
- `gha-kumo-backend-gpu-publish`
- `gha-kumo-ui-image-publish`

Restrict each environment to `main`. The workflows name the required AWS roles.
Each role must trust its exact
`repo:NVIDIA/kumo-engine:environment:<ENVIRONMENT>` identity, not a fork or the
old repository. This code does not create environments or change cloud trust.
The nightly workflow also calls this publisher. Review its schedule before
enabling publication credentials.

For customer ACR publication, configure `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`, `ACR_NAME`, and `ACR_LOGIN_SERVER` in
`gha-kumo3-image-set-ecr-read`. The Azure identity must trust that environment
and have permission to push to the approved registry.

Do not run the following command before approval of the source commit and
publication destination. It publishes all three images to ECR.

```bash
gh workflow run _docker-kumo3-images-custom.yml \
  --repo NVIDIA/kumo-engine --ref main \
  -f kumo_repo_ref=FULL_COMMIT_SHA -f images=all -f bake_glove_level=basic
```

Add `-f publish_to_acr=true` only for the approved customer registry.
Record the published image digests and scan those exact images before deployment.
The Python build helper also selects this repository and defaults to `main`.
Its optional legacy mirror additionally needs `gha-kumo-image-sync-ecr-read`
and the target registry credentials named by the sync workflow.

The branch/tag helpers below still assume `master` and existing `v3.x` release
refs. Do not use them to cut a release from this repository's `main` branch.
Their migration is separate from image publication.

How to perform a hotfix
------------------------

Please note:

1. **Fix a bug/incident on `master` first**: Always merge the fix to `master` before backporting (unless it is a severe emergency blocking `master`).
2. **Backport to ALL affected versions**: You must forward-port/backport the fix to every supported release branch that is affected. Do not leave "holes" where a bug reappears in a future version.
3. **Gating for Certified Versions**: You may ONLY backport to a certified (released) branch if the bug is a **P0 / Incident** or a **release test failure**. Feature additions or minor bugs are not allowed on certified branches.

Process:

1.  **Fix the bug on `master`**.
    - Create a PR targeting `master`.
    - In the PR template, fill in **"If P0 Fix on master"** under **Hotfix Context** with the required context (incident link, impact, etc.).
    - Merge it.

2.  **Backport to release branches**.
    - For **each** affected release branch (for example, `v3.0`), create a backport PR via the backport script
    ```bash
    # Example: ./scripts/release/backport_pr.sh <MASTER_PR_ID> <RELEASE_BRANCH>
    ./scripts/release/backport_pr.sh 1234 v3.0
    ```

Release Process
------------------------

The release builds `kumo-ui`, `kumo-backend-cpu`, and
`kumo-backend-gpu`. Use the release script below.

### How to cut a release

The scripts preflight the GitHub CLI, authentication, and Python dependencies.
Run from inside the worktree so mise resolves the pinned toolchain, and prepare
the worktree venv before previewing or cutting a release:

```bash
mise trust
mise exec -- make install-venv
source .venv/bin/activate
```

1. Preview the release without creating or publishing release refs:

   ```bash
   ./scripts/release/start_kumo3_release.sh --dry-run false v3.0
   ```

   `--dry-run` prints the commands that would create or publish a release.
   It still runs preflight checks, fetches refs/tags, and contacts GitHub for
   release information and notes. Nothing is pushed. The positional `false`
   disables CI mode, so a live run prints image-build commands for you to run.

2. Cut it for real:

   ```bash
   ./scripts/release/start_kumo3_release.sh false v3.0
   ```

   This creates branch `v3.0`, tags `v3.0.0`, publishes a GitHub release with
   notes generated since the previous minor release, and prints the image-build
   commands.

3. Build the images. Run from inside the worktree so mise resolves the pinned
   toolchain, and invoke the worktree venv explicitly -- `build_kumo_images.py`
   imports `packaging`, and a bare `python3` can resolve outside the venv that
   holds it. The script preflights this before pushing anything.

   ```bash
   .venv/bin/python scripts/release/build_kumo_images.py v3.0.0 --app kumo3 --target saas
   .venv/bin/python scripts/release/build_kumo_images.py v3.0   --app kumo3 --target saas
   ```

   The commands printed by the release scripts additionally pass
   `--publish_to_acr`, which mirrors the built images to the configured Azure
   registry. Use that option only for the release destination approved by the
   release owner.

### Hotfixes

Same process as above -- fix on `master`, then `backport_pr.sh <PR> v3.0` --
and cut the release with `make_kumo3_hotfix_release.sh`.

Once the backport PRs are merged, run the hotfix script from the release branch:

```bash
git checkout v3.0 && git pull
# The script preflights `python3 -c "import retry, packaging"`, so it needs the
# worktree venv on PATH -- an ambient python3 does not have them.
mise trust
mise exec -- make install-venv
source .venv/bin/activate
# Preview it. Preflight checks, ref fetching, and GitHub reads still run.
# Nothing is tagged, pushed, or published.
./scripts/release/make_kumo3_hotfix_release.sh --dry-run false
# Cut it: tags v3.0.1, publishes the release, prints the image-build commands.
./scripts/release/make_kumo3_hotfix_release.sh false
```

It infers the next patch tag from the highest one on the branch. Pass `<TAG>`
and `<PREV_TAG>` explicitly to override that. Image builds are the same two
`build_kumo_images.py` commands as a release cut, for the new patch tag and for
the floating `v3.0` tag.

## Service SDK publication

SDK release branches use `kumo-sdk-vX.Y`; tags use `kumo-sdk-vX.Y.Z` with
optional `bN` or `rcN` suffixes. See the
[release preparation workflow](../../.github/workflows/kumo-sdk-release.yml)
and [tag workflow](../../.github/workflows/kumo-sdk-release-on-tag.yml).

The SDK version is the root [`VERSION`](../../VERSION) file, shared with
kumo-engine, the platform and the charts: the release workflow changes it through
`packages/kumo-sdk/scripts/set_version.py`, which also runs
`scripts/stamp_version.py` and stages every stamped file (`git add -u`).

The [upload workflow](../../.github/workflows/kumo-sdk-upload-wheels.yml)
requires `KUMO_SDK_PUBLICATION_ENABLED=true` and the `kumo-sdk-release`
environment, including tokens for the selected PyPI destinations. Before
production publication, validate a TestPyPI-only upload with
`upload-to-testpypi` enabled and `upload-to-pypi` and `upload-to-s3` disabled.
Coordinate other callers while enabling the repository-wide
gate; it does not restrict them to TestPyPI. Configure the credentials and
OIDC trust required by each selected workflow before running it.
