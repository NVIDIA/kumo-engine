# Working in Kumo

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the system map and
[docs/README.md](docs/README.md) for canonical documentation.
This file contains agent navigation and working constraints; shared engineering
knowledge belongs in ordinary docs.

## Find context by the change

Before editing, identify the affected components and read their applicable
instructions. Follow API, runtime-configuration, and generated-artifact
connections across directories. When scope expands, read the additional
guidance before changing that area. Starting at the repository root does not
mean every nested instruction file has already been loaded.

| Change | Read and inspect |
| --- | --- |
| REST/API response | `rest/`, `packages/kumo-api/`, `packages/kumo-sdk/`, [UI instructions](ui/AGENTS.md), [generator map](docs/development/generated-artifacts.md) |
| Runtime setting or default | `service_framework/`, affected service/worker consumers, [Helm requirements](helm/kumo-helm/docs/requirements.md), `docker/compose/` |
| Worker/GPU/cache behavior | `pool/`, `kumo/diskgraph/`, `packages/kumo-core/`, [DiskGraph source ownership](docs/development/generated-artifacts.md#diskgraph-source), [Helm guide](helm/README.md) |
| Query syntax or table semantics | [PQL instructions](packages/kumo-pql/AGENTS.md), [query integration](kumo/pquery/AGENTS.md), [DiskGraph source ownership](docs/development/generated-artifacts.md#diskgraph-source), UI grammar/help generators |
| UI feature or shared component | [UI instructions](ui/AGENTS.md), [catalog instructions](packages/kumo-ui-catalog/CLAUDE.md), shared API and generated clients |
| Filesystem/backend behavior | [fsutil instructions](io_utils/fsutil/AGENTS.md), affected connector and storage consumers |
| Local Python interface | `packages/kumo-engine/`, `packages/kumo-core/`, relevant connector packages and their tests |
| Documentation or skills | [maintenance](docs/maintenance/README.md), [agent setup](docs/development/agents.md), [.agents/README.md](.agents/README.md) |

## Working rules

- Be concise in user replies and authored output, including PR descriptions,
  review comments, commit messages, code comments, and documentation. Avoid
  repetition and unnecessary wording; include detail only when it helps someone
  understand, verify, or act.
- Verify factual claims and commands against current source. Distinguish an
  existing implementation from a verified public release or support commitment.
- Preserve user changes. Keep edits and validation scoped to the task.
- Use the repository's [setup guide](docs/development/setup.md). Report missing
  prerequisites; do not silently install system packages, edit shell/global
  configuration, or contact private services to make a task proceed.
- Public skills are optional repository development procedures. They do not
  require a private catalog, MCP server, or employee credentials.
- Read the relevant [generated-artifact rules](docs/development/generated-artifacts.md)
  before modifying generated output. Change its source and regenerate.
- Select meaningful [validation](docs/development/testing.md); report commands,
  results, and checks that could not run. Documentation needs a factual review
  as well as reference validation.
- Read [review requirements](docs/development/review.md) before preparing a PR,
  including the single maintained change-map comment.
- Require task-specific authorization before cluster/cloud mutations, external
  publishing, or sending messages. Local validation is not permission to run
  Helm installation or decommission commands.

## Instructions and skills

Repository skills live in [.agents/skills](.agents/skills/); catalog skills
remain in [the catalog package](packages/kumo-ui-catalog/.agents/README.md).
Use [.agents/README.md](.agents/README.md) to find the relevant procedure.
Read skills only when they apply, and follow their linked engineering guides.
