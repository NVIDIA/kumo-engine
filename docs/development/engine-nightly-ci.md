# Engine nightly check

The nightly check trains sixteen fresh models on the frozen PR172 `100M`
profiles. Each of eight tasks runs once with normal answers and once with
scrambled answers. It reuses the PR runner's model recipes with seed zero.

Two RTX jobs each run eight primary fits. One job also runs the existing
quickstart and behavior tests, including weighted training, time filters,
prediction counts, warm starts, and model reuse in a fresh process. Those
extra fits are counted separately. The host's shared GPU lock prevents
overlapping model jobs.

Every primary case requires real CUDA work, the complete training recipe,
complete finite predictions, repeat and partial predictions, and exact
checkpoint restoration. Link predictions require a certified ranking after
filtering known connections. Failed execution or missing results fail the
required job. Quality and unqualified numerical differences report only.

The reference is the approved `kumo-engine==1.0.0` wheel. Its sixteen
completed fits and retained recovery results become a pinned, data-only
reference index. Routine CI does not rerun reference fits or execute old
audit scripts. Each candidate uses the same frozen input files, queries,
splits, model recipes, and prediction populations.

## Run configuration

The workflow starts at 10:00 UTC daily, or manually with `product_ref`.
Trusted copied PR branches also run it when the nightly implementation or
shared GPU tests change. Manual dispatch requires the workflow on the
default branch. No complete nightly runtime limit is qualified yet.

Set these repository variables after verifying the existing RTX route:

| Variable | Value |
| --- | --- |
| `KUMO_ENGINE_NIGHTLY_RTX_RUNNER` | Existing label that selects the two retained RTX hosts |
| `KUMO_ENGINE_NIGHTLY_ROOT` | Common absolute root containing `inputs/`, `reference.json`, and the existing shared `gpu-study.lock` |
| `KUMO_ENGINE_NIGHTLY_INPUT_SHA256` | SHA-256 of `inputs/manifest.json` |
| `KUMO_ENGINE_NIGHTLY_REFERENCE_SHA256` | SHA-256 of `reference.json` |

Both hosts require the same admitted data and metadata. The input manifest
pins the frozen 98-package runtime and allowed GPUs. The installer uses a
fresh Python 3.12 environment, public PyPI, and the exact candidate wheel.
It checks installed files against that archive. The workflow retains raw
results, models, installation receipts, and build identity. Full input data
and native indexes stay on the hosts.

The previous GitHub GPU run used a managed T4. Successful SSH tests on the
retained RTX hosts do not establish GitHub dispatch to them. Their route,
final data admission, and a complete automated candidate run remain to be
verified. This workflow creates no runners and publishes no packages.
