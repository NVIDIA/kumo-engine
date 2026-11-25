# Contributing

This repository is a **reference-architecture kit**, not a production fraud or
serving system. Development happens on the `v2` branch. Milestone status is
tracked in [docs/reference/roadmap.md](docs/reference/roadmap.md).

## Setup

```bash
uv sync --extra native --extra dev
make check
```

Python 3.12 is required (see `.python-version`). Use only synthetic fixtures
in tests and CI.

## Rules

1. Read the root `AGENTS.md` and any scoped `AGENTS.md` near the files you change.
2. Do not commit real customer, account, card, merchant, or transaction data.
3. Preserve temporal ordering and label-availability semantics.
4. Add or update unit and contract tests in the same change.
5. Run the commands in `AGENTS.md` and report what you ran versus skipped.

## Development branches

- Create short-lived branches from `v2`; use a descriptive prefix such as
  `feature/`, `fix/`, or `docs/`.
- Open merge requests against `v2`. Do not push changes directly to the shared
  development branch.
- The `v1` branch is frozen except for documentation, security, and
  reproducibility fixes. Open those merge requests against `v1`; do not
  automatically backport them to `v2`. Make a separate `v2` merge request when
  the fix applies to both versions.
- The historical `WS5`/`WS6`/`WS7` workstream branches remain frozen on `v1`
  and are not merged into `v2`. See
  [the v1-to-v2 migration guide](docs/migration/v1-to-v2.md).

## Commits

- Keep commits focused on one logical change and avoid unrelated formatting or
  refactors.
- Write concise, imperative commit subjects, for example
  `Add causal-mask validation`. Conventional Commit type/scope prefixes are
  welcome but not required.
- Include tests and documentation with the change they support. Documentation
  sources of truth are listed in `AGENTS.md` and `docs/AGENTS.md`; do not
  duplicate the catalog or compatibility matrix in `README.md`.
- Never commit real data, credentials, generated private artifacts, or
  unrelated generated files.

## Local validation

Run the CPU-safe checks before requesting review:

```bash
make check
```

This mirrors the lint, type-check, unit, contract, and recipe-smoke jobs in
GitLab CI. CI jobs target the shared Linux Docker runner pool using the tags in
`.gitlab-ci.yml`; shared runners must be enabled for the project. Untagged jobs
remain queued because those runners do not accept them.
Notebook CI uses the large runner class and two OpenMP/MKL threads because its
smokes retain the lab model's dimensions. Its per-cell limit is ten minutes;
the overall job limit remains thirty minutes.

Run narrower checks while developing when appropriate, but include
the commands and results in the merge request. State any skipped validation and
why, especially for optional GPU or external-backend paths.

When you change a notebook, or code a notebook calls, also run
`make test-notebooks` (needs `uv sync --extra native --extra dev --extra notebooks`). CI runs
it as the `notebooks` job. Notebook conventions are in
[notebooks/STYLE.md](notebooks/STYLE.md).

When you change documentation or a public docstring, run `make docs`
(needs `uv sync --extra docs`). It builds the site in strict mode, so a broken
link fails, as the `docs` CI job does. A new page needs an entry under `nav:`
in `mkdocs.yml`. A new public package needs a page under `docs/_site/api/`.

## Merge requests

Use the GitLab merge-request template. A merge request to `v2` is ready to
merge only when:

- the required GitLab pipeline passes;
- required `CODEOWNERS` approval is present;
- review discussions are resolved; and
- the affected tests, documentation, contracts, support tier, and temporal
  assumptions are included in the template.

Prefer squash merging to keep `v2` history readable, unless preserving an
intentional commit series is useful for review or bisection. GitLab
protected-branch and approval settings must enforce this policy; this document
does not replace those project-level controls.

## Design-first changes

Open the appropriate GitLab issue or proposal before implementing a change to a
data contract, label semantics, temporal split, public API, architecture
support tier, or dependency boundary. Use the architecture, data-contract, and
recipe proposal templates in `.gitlab/issue_templates/`.

Follow the breaking-change workflow in `AGENTS.md`: update the applicable
contract or architecture documentation, affected recipes, compatibility matrix,
migration notes, and regression tests in the same merge request.

## Ownership and escalation

`CODEOWNERS` identifies paths requiring designated review, including contracts,
splits, labels, evaluation, and governance documentation. Maintainers must
keep the GitLab group or user handles current and configure corresponding
approval rules for protected branches.

For conduct concerns, follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). For
security vulnerabilities, follow [SECURITY.md](SECURITY.md); do not disclose
them in a GitLab issue or merge request.

## Versions and releases

Releases are maintainer-owned. Version, changelog, compatibility documentation,
and migration notes must be updated together when applicable. The release
checklist and tag policy are in
[docs/governance/release-process.md](docs/governance/release-process.md).
