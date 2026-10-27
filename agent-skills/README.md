# Agent skills: `kumo-data-science`

Agent skill for building, training, and running inference with Kumo models. One shared source tree
serves every host; host-specific behavior lives under `skills/hosts/`. Three hosts:

| Host | Package | For |
|---|---|---|
| `hosts/snowflake-coco/` | `coco` | Snowflake Cortex Code (CoCo), inside Snowflake |
| `hosts/generic/` | `generic` | a coding agent beside the user — Claude Code, Codex — generating code for any of the six Kumo connectors |
| `hosts/databricks-genie/` | `genie` | Databricks Genie Code, inside a Databricks workspace, generating notebooks for a GPU cluster |

This directory is deliberately not under `.agents/skills/` or `.claude/skills/`: Codex, Claude Code
and the CoCo CLI load those for repo developers, and this skill triggers on "kumo" and "train a
model".

Why the rules are what they are is in `DESIGN.md`. It is not packaged.

## Layout

`skills/` uses the source layout. `kumo-data-science/` is the router skill; the other directories
are sub-skills and references it loads:

| Directory | Role |
|---|---|
| `kumo-data-science/` | Router: intake, validation, preflight, notebook generation |
| `kumoai-reference/` | SDK reference the generated code is checked against |
| `hosts/` | Host-specific behavior: access, source and destination forms, preflight, connect step |
| `kumo-model-explanations/` | Explaining predictions |
| `kumo-training-diagnostics/` | Diagnosing training runs |
| `kumo-model-persistence/` | Saving and reloading trained models |
| `kumo-custom-populations/` | Custom entity populations |
| `kumo-model-tuning/` | Model tuning |

## Packaging

The committed layout is not the layout that ships. In a package every directory other than
`kumo-data-science/` nests **inside** `kumo-data-science/`, so the router reaches them by relative
path. Build a host package with:

```bash
python3 agent-skills/scripts/package_skill.py coco        # -> agent-skills/dist/coco/
python3 agent-skills/scripts/package_skill.py generic     # -> agent-skills/dist/generic/skills/
python3 agent-skills/scripts/package_skill.py genie       # -> agent-skills/dist/genie/skills/
python3 agent-skills/scripts/package_skill.py genie --deploy   # + upload to your Genie Code skills folder
python3 agent-skills/scripts/package_skill.py --check     # build every host, verify references
```

A package carries exactly one host, and the core reaches it only as `hosts/<host>/`; tests fail if
a core file names a host. To use the `generic` package, copy
`dist/generic/skills/kumo-data-science/` into the agent's user-level skills directory (for Claude
Code, `~/.claude/skills/`) — not into this repository, for the reason above. `genie --deploy`
uploads the `genie` package to `/Users/<you>/.assistant/skills/kumo-data-science/` in the
workspace named by `DATABRICKS_HOST` and `DATABRICKS_TOKEN` and reads every file back; a new Genie
Code chat loads it.

## Shipping in the `kumo-engine` wheel

Users install the skill from Kumo itself, so it always matches the installed SDK:

```python
import kumo.engine as kumo
kumo.install_skills()                  # Genie on Databricks; else every local agent found
kumo.install_skills('claude')          # ~/.claude/skills/        (scope='project': ./.claude/skills/)
kumo.install_skills('codex')           # ~/.agents/skills/        (scope='project': ./.agents/skills/)
kumo.install_skills('coco')            # ~/.snowflake/cortex/skills/ (scope='project': ./.cortex/skills/)
kumo.install_skills('coco', scope='workspace', workspace='MY_WS')   # a Snowsight workspace
kumo.install_skills('coco', scope='extension')                      # a Cortex Extension
kumo.install_skills('genie')           # /Users/<you>/.assistant/skills/
kumo.install_skills(path='...')        # any other agent's skills directory (generic package)
```

`python -m kumo.engine.skills [agent] [--scope ...]` does the same from a shell. Claude Code and
Codex get the `generic` package, CoCo `coco`, Genie `genie`.

`packages/kumo-engine/setup.py` calls `package_skill.bundle()` on every build from a checkout,
writing all three packages to `kumo/engine/_skills/` (gitignored), which the wheel ships as package
data. In a source checkout `install_skills` builds from `agent-skills/` directly instead, so an
edit is installed without a rebuild. Each install replaces only its own `kumo-data-science/`
directory. `tests/test_install_skills.py` covers every agent and scope; the Genie and Snowflake
account paths run there against fakes.

The build stamps the router's `SKILL-BUILD` marker with the source commit
(`<commit date>-<short sha>`, `-dirty` for uncommitted edits); never edit it by hand. Every build
fails if a markdown link or a backticked package path does not resolve inside the
package. The `Agent Skills` CI workflow runs the same check on every change under `agent-skills/`.

The skills cite Kumo code by module path, as the agent sees it in a notebook
(`kumo_core.train.trainer.Trainer.explain`), never by commit SHA. `tests/test_source_references.py`
resolves each one against `packages/`, and checks every `kumo.<name>` against `kumo.engine.__all__`,
so the workflow also runs on changes to `kumo-engine`, `kumo-core`, `kumo-api` and `kumo-snowflake`.
`kumoml.*` lives outside this repository and is not checked.
