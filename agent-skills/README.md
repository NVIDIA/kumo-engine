# Agent skills: `kumo-data-science`

Agent skill for building, training, and running inference with Kumo models. One shared source tree
serves every host; host-specific behavior lives under `skills/hosts/`. Currently shipped to
Snowflake Cortex Code (CoCo).

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
| `hosts/` | Host-specific behavior (currently Snowflake CoCo) |
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
python3 agent-skills/scripts/package_skill.py --check     # build every host, verify references
```

The build stamps the router's `SKILL-BUILD` marker with the source commit
(`<commit date>-<short sha>`, `-dirty` for uncommitted edits); never edit it by hand. Every build
fails if a markdown link or a backticked package path does not resolve inside the
package. The `Agent Skills` CI workflow runs the same check on every change under `agent-skills/`.

The skills cite Kumo code by module path, as the agent sees it in a notebook
(`kumo_core.train.trainer.Trainer.explain`), never by commit SHA. `tests/test_source_references.py`
resolves each one against `packages/`, and checks every `kumo.<name>` against `kumo.engine.__all__`,
so the workflow also runs on changes to `kumo-engine`, `kumo-core`, `kumo-api` and `kumo-snowflake`.
`kumoml.*` lives outside this repository and is not checked.
