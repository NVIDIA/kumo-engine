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
| `kumo-graph-visualization/` | Provide graph visuals for users; inspect schemas during model iteration |
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
python3 agent-skills/scripts/package_skill.py --zip DIR   # kumo-data-science-<host>.zip for every host
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
kumo.install_skills('genie-cli')       # ~/.agents/skills/, which the Genie Code CLI reads
kumo.install_skills(path='...')        # any other agent's skills directory (generic package)
```

`python -m kumo.engine.skills [agent] [--scope ...]` does the same from a shell. A shell has no
Snowpark session, so `--scope workspace` and `--scope extension` need `--connection NAME`, a
`connections.toml` entry as the `snow` CLI uses. Claude Code and Codex get the `generic` package,
CoCo `coco`, Genie `genie`. The Genie Code CLI works locally through its shell, like Codex, and
reads Codex's `.agents/skills`, so it gets the `generic` package there; a second copy in
`~/.genie/skills` would load as a competing skill.

Cortex Code also reads `~/.claude/skills`, and a copy there wins over `~/.snowflake/cortex/skills`.
With both agents installed, CoCo would run the `generic` package, so `install_skills()` skips CoCo's
user install and says so, `install_skills('coco')` warns, and `scope='project'` or
`scope='extension'` gives CoCo its own.

Each package is stamped `SKILL-BUILD: <date>-<sha>`. An install prints the build it installed and
the one it replaced. `import kumo.engine` warns when a local install is from another build than the
wheel, and `python -m kumo.engine.skills --check` lists the local installs and exits 1 if any is
stale. A Cortex Extension gets a new version only when the package changed, and the install waits
for Snowflake's scan of it, which Cortex Code needs before it installs the version.

`packages/kumo-engine/setup.py` calls `package_skill.bundle()` on every wheel or sdist build from
a checkout, writing all three packages to `kumo/engine/_skills/` (gitignored), which the wheel
ships as package data. An editable install (`uv sync`, `pip install -e`) skips it and leaves
`kumo/engine/_skills/` as it was: in a source checkout `install_skills` builds from `agent-skills/`
directly, so an edit is installed without a rebuild. Each install replaces only its own `kumo-data-science/`
directory. `tests/test_install_skills.py` covers every agent and scope; the Genie and Snowflake
account paths run there against fakes.

## Download zips on each release

For anyone installing the skill without the wheel (a docs download button, an install script),
every `kumo-engine-v*` GitHub release carries one zip per host, attached by the `skill-zips` job in
`.github/workflows/kumo-engine-wheel.yml` beside the wheels built from the same commit:

```
https://github.com/NVIDIA/kumo-engine/releases/download/kumo-engine-v<version>/kumo-data-science-generic.zip
https://github.com/NVIDIA/kumo-engine/releases/download/kumo-engine-v<version>/kumo-data-science-coco.zip
https://github.com/NVIDIA/kumo-engine/releases/download/kumo-engine-v<version>/kumo-data-science-genie.zip
```

Each zip holds `kumo-data-science/` at its root, the directory `install_skills()` installs: unzip it
into the agent's skills directory (the same locations as above). Link by release tag, never through
`releases/latest`: this repository also publishes `kumo-sdk-v*`, model and wheel releases, and
whichever is newest is "latest". The zips are reproducible, so `/skill-zips` can rebuild a
release's zips from its tag and compare them byte for byte.

The build stamps the router's `SKILL-BUILD` marker with the source commit
(`<commit date>-<short sha>`, `-dirty` for uncommitted edits); never edit it by hand. Every build
fails if a markdown link or a backticked package path does not resolve inside the
package. The `Agent Skills` CI workflow runs the same check on every change under `agent-skills/`.

The skills cite Kumo code by module path, as the agent sees it in a notebook
(`kumo_core.train.trainer.Trainer.explain`), never by commit SHA. `tests/test_source_references.py`
resolves each one against `packages/`, and checks every `kumo.<name>` against `kumo.engine.__all__`,
so the workflow also runs on changes to `kumo-engine`, `kumo-core`, `kumo-api` and `kumo-snowflake`.
`kumoml.*` lives outside this repository and is not checked.

## Publishing to the NVIDIA skills catalog

[NVIDIA/skills](https://github.com/NVIDIA/skills) pulls every product's skills from its source repo
once an hour. It accepts only content that NVSkills CI has evaluated and signed. NVSkills CI runs
only on PRs that change a root-level `skills/` directory. So the catalog reads a built copy in
`skills/` at the repository root, not `agent-skills/skills/`.

The catalog does not allow a skill inside another skill. The package nests seven skills inside the
router (`kumoai-reference/` and the six `kumo-*/` capability skills), so the published copy lays
all eight out side by side: `skills/kumo-data-science/`, `skills/kumoai-reference/`,
`skills/kumo-model-tuning/` and so on, each its own catalog skill. Their links to one another
(`../kumo-model-persistence/SKILL.md`) resolve the same way in both layouts. A backticked path that
starts with a skill's name (`kumoai-reference/references/outputs.md`) names that skill's directory,
as the router's *Paths in these files* says; any other path is relative to the skill that names it.
`publish_nvskills.py` checks every reference in the published layout too, so a path that resolves
only when nested (`accept/preflight.md` written in `kumoai-reference/`) fails the build: write it
`kumo-data-science/accept/preflight.md`.

The catalog sync deletes any skill directory that lacks its own `skill.oms.sig`, `skill-card.md`
and eval dataset:

| Path, in each of the eight skill directories | Written by |
|---|---|
| everything else | `agent-skills/scripts/publish_nvskills.py`: the `generic` package |
| `evals/` | copied from `agent-skills/nvskills/<skill>/`, e.g. `agent-skills/nvskills/kumo-model-tuning/evals/evals.json` |
| `skill-card.md` | first from `agent-skills/nvskills/<skill>/`; NVSkills CI rewrites it when it signs, adding the evaluation results, and from then on a refresh leaves it alone. To change a signed card, edit `skills/<skill>/skill-card.md` in your PR |
| `BENCHMARK.md`, `skill.oms.sig` | NVSkills CI, in your PR; never by hand |

Once `agent-skills/nvskills/` exists, the build fails unless every skill has both a card and an
`evals/*.json`, so a new skill cannot be published without them. Until then
`publish_nvskills.py` lists what is missing.

Whatever is on `main` is what the catalog publishes. To change the skill:

1. Edit `agent-skills/`, never `skills/`.
2. Run `python3 agent-skills/scripts/publish_nvskills.py` and commit `skills/` with your change.
   `tests/test_publish_nvskills.py` fails the PR if you forget. It ignores the `SKILL-BUILD`
   stamp, so a rebuild on an unrelated commit is not required.
3. A maintainer comments `/nvskills-ci` on the PR. NVSkills CI runs the evals and pushes
   `BENCHMARK.md` and `skill.oms.sig` as a commit by `nv-skills-ci[bot]`. copy-pr-bot does not
   trust that commit, so `kumo-core-gpu-ci` runs only after a maintainer comments
   `/ok to test <sha of that commit>`. A later push to the PR needs another `/nvskills-ci`.
4. Merge. The catalog picks the change up within the hour.

A change merged without step 3, a revert included, is "sig drift" to the catalog sync: it keeps
the last signed copy, so the catalog serves content that `main` no longer has until the next signed
merge. Run `/nvskills-ci` on reverts too.

### Before the first publish

The catalog publishes nothing until all of this is done. Each skill has its own card and evals
under `agent-skills/nvskills/`. The cards are rendered with the catalog's `skill-card-generator`
(template in NVIDIA/skills `docs/skill-cards.mdx`) and pass its `validate_submission.py`. The
evals use the `evals/evals.json` schema of published catalog skills, and assert only behaviour the
skill's own files document. Each set has one negative case (`expected_skill: null`).

| Skill | Card | Evals |
|---|---|---|
| `kumo-data-science` (router) | [x] `kumo-data-science/skill-card.md` | [x] `kumo-data-science/evals/evals.json` |
| `kumoai-reference` | [x] `kumoai-reference/skill-card.md` | [x] `kumoai-reference/evals/evals.json` |
| `kumo-custom-populations` | [x] `kumo-custom-populations/skill-card.md` | [x] `kumo-custom-populations/evals/evals.json` |
| `kumo-graph-visualization` | [x] `kumo-graph-visualization/skill-card.md` | [x] `kumo-graph-visualization/evals/evals.json` |
| `kumo-model-explanations` | [x] `kumo-model-explanations/skill-card.md` | [x] `kumo-model-explanations/evals/evals.json` |
| `kumo-model-persistence` | [x] `kumo-model-persistence/skill-card.md` | [x] `kumo-model-persistence/evals/evals.json` |
| `kumo-model-tuning` | [x] `kumo-model-tuning/skill-card.md` | [x] `kumo-model-tuning/evals/evals.json` |
| `kumo-training-diagnostics` | [x] `kumo-training-diagnostics/skill-card.md` | [x] `kumo-training-diagnostics/evals/evals.json` |

- [x] the `NVSKILLS_CI_DISPATCH_TOKEN` repository secret, from the NVSkills CI maintainers;
      `request-nvskills-ci.yml` fails without it
- [x] kumo-engine registered with NVSkills CI (NVIDIA/nvskills-ci#112: `main`, external profile,
      no fork PRs) and the `nv-skills-ci` GitHub App installed
- [x] the catalog PR to NVIDIA/skills adding `components.d/kumo-engine.yml`, with one `skills:`
      entry per directory (`path: skills/<skill>/`, `catalog_dir: <skill>`), all eight:
      NVIDIA/skills#885, a draft. Its `verify-source-onboarding` check verifies only the paths the
      file lists, and passes only once the signed copy is on `main`
- [x] `request-nvskills-ci.yml` on `main` (NVIDIA/kumo-engine#371). `/nvskills-ci` is an
      `issue_comment` trigger, so it runs only from `main`'s copy
- [ ] confirmation from the NVSkills CI maintainers that a skill may link to files in a sibling
      skill directory (`kumo-data-science/` reads `kumoai-reference/`). Each directory is signed
      and published as its own skill
- [ ] a signed copy on `main`: a PR through steps 1–4 whose `BENCHMARK.md` says PASS in every
      skill directory
