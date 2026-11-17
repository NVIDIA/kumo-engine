# Repository layout

Status: implementation proposed in the accompanying PR.

The root previously mixed the Studio application, reusable engine packages,
release scripts, deployment configuration, examples and test trees. New work
had several plausible homes, and the wheel builder implicitly bundled new
projects placed under `packages/`.

## Target and ownership

```text
apps/
  studio/             Application Python packages, frontend and deployment assets
packages/
  contracts/          Shared schemas, PQL and settings
  data/               DiskGraph and the PyTorch batch-data API
  runtime/            Shared engine runtime; existing GNN implementation remains here
  models/             Model artifacts and model-specific adapters
  integrations/       Source providers and platform integrations
  clients/            Local engine SDK and remote Studio SDK
  ui/                 Reusable web component library
development/
  tooling/            Repository build, release and maintenance commands
  testing/            Cross-component tests and their environments
  benchmarks/         Cross-component workload generation and measurements
  config/             Shared development tool configuration
  third-party/        Dependency attribution inputs and build artifacts
docs/                Shared architecture, development guidance and designs
```

A component owns its internal layout. Examples, tests, benchmarks and
experiments belong with the component they exercise. The existing SDK examples
and distributable agent skills live with `packages/clients/kumo-engine`.
Existing Studio utility/tool modules remain with Studio because they are part
of its Python distribution; separating their implementations is a later change.
Cross-component validation and benchmarks live under `development/`.

`packages/data/kumo-pytorch` exposes the `kumo.pytorch` API for preparing
relational data as model batches for training and inference. It belongs with
data execution, alongside DiskGraph. Its experimental status remains in its
documentation and package metadata; the directory groups describe responsibility.
Reference models inside that package do not make it a model adapter.

`development/repository-moves.json` records the exact old-to-new path mapping
and the starting revision. It is a migration record, not a runtime path resolver.
No compatibility symlink tree recreates the old repository layout.

## Preserved contracts and deliberate changes

Python distribution names, import namespaces, entry-point names, public SDK
APIs and native crate boundaries stay the same. This change does not extract
GNN modules, add a family registry, move SDM from another repository, or change
model/index formats.

Studio becomes the `apps/studio` workspace project. The root `pyproject.toml`
is a virtual workspace. Select Studio explicitly with
`uv sync --package kumo --extra dev --extra cpu`; install its distribution from
`apps/studio` when using pip. Its runtime packages retain their original names,
including the shared `kumo` namespace. Repository-only `scripts`, `test`, and
`tests` packages no longer enter the Studio runtime wheel accidentally.

The merged engine wheel reads `packages/engine-bundle.txt`. DiskGraph retains
its separate native build step. Adding a client, experiment or grouping directory
cannot silently add it to the released engine wheel.

The root keeps introductory/governance documents, Git and GitHub/agent discovery
files, the workspace manifest and lock, release/legal files, and small command
entry points. Shared tool configuration has explicit paths in `dev`, Make and CI.
CodeRabbit's service-discovery configuration remains at the root. Backend Docker
builds use Dockerfile-specific ignore files generated from
`development/config/dockerignore`.

## Contributor workflow

Use `./dev` with the prepared development Python interpreter for shared checks:

```bash
.venv/bin/python dev layout-check
.venv/bin/python dev test development/testing/test/scripts
.venv/bin/python dev lint apps/studio/rest
.venv/bin/python dev types
.venv/bin/python dev pre-commit run --all-files
```

Existing `make` commands remain available through a small root include. Individual
packages retain their own test and build commands. CI uses the relocated source
paths and shared configuration.

For foundation-model batch inference, add model-specific adaptation under
`packages/models`, reuse the batch execution in `packages/runtime`, and expose
it through the SDK under `packages/clients`. Extend `packages/data` or platform
integrations only when their contracts actually need to change. An independently
operated worker belongs under `apps`; a model adapter does not require a new app.
Link the system design across these components rather than creating a directory
for each feature.

## Migration and validation

Move directories and update consumers together: workspace/build metadata,
Docker COPY destinations, npm local dependencies, workflows and path filters,
source-location helpers, fixture repositories, developer commands, agent guidance,
and documentation links. Update generated lockfiles with their owning tools.
Preserve imported provenance records and data fixtures byte-for-byte.

Validation covers the layout checker, lockfile consistency, Python artifact
builds and runtime module inventories, engine assembly tests with native-build
stand-ins, source/manifest licensing, local documentation links, workflow helper
tests, and focused component suites. The full release GPU matrix, cloud connector execution,
image deployment and full-scale workloads still need their appropriate environments;
local static checks do not establish those results.

Before merge, all required CI must pass or have a documented baseline failure
reproduced at the starting revision. Deployment/image owners must review path
changes in their build and launch flows. Do not merge on the strength of rename
similarity alone.

## Local validation on 2026-11-16

| Check | Result |
| --- | --- |
| PQL parser suite | 1,805 passed |
| API package suite | 1,138 passed |
| Remote Studio SDK | 664 passed, 28 skipped |
| Connector import boundaries | 15 passed |
| Repository tooling suite | 941 passed, 3 skipped; 12 existing Faiss fixture failures reproduced on the starting revision |
| Distributable agent skills | 290 passed |
| Selected SDK training, prediction, reload, monitoring and presentation tests | 123 passed with a freshly built native DiskGraph wheel |
| Workflow helper tests | 33 passed |
| Local launch configuration | 45 concurrent-devserver and Compose configuration checks passed |
| Chart packaging and CSI guards | Bundle, CSI error-path and CSI node-coverage checks passed |
| Selected UI proxy, editor-help and table-overview tests | 64 passed |
| Core service-import boundary | Passed |
| UI type check and component-library build | Passed |
| Python package builds | All selected pure-Python workspace projects built; native DiskGraph and merged engine wheel built |
| Source archive round trips | Engine and Studio archives rebuilt into wheels |
| Isolated merged-wheel imports | All 12 bundled import surfaces resolved inside the installed artifact; Studio/remote SDK absent |
| Dependency resolution | Lock check passed; all 322 registry-resolved package records unchanged |
| Licensing | Source headers, 15 distribution manifests and 14 pure-Python wheel artifacts passed |
| Documentation and migration inventory | Local links/agent contract and all 8,122 original file locations accounted for |

The local native/merged wheel uses this host's glibc 2.35 with `--no-zig` and an
explicit `MAX_GLIBC=2.35`. It is a validation artifact, not a portable release.
The release workflow retains its existing Zig build and default glibc ceiling.

GitHub's PR-files API returns at most 3,000 files. This migration exceeds that
limit; review the move manifest and full Git diff locally. The existing PR
change-map automation needs separate accommodation for this migration before
it can validate the complete file inventory.
