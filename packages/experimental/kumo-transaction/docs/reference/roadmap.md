# Roadmap

This page is the single source of truth for **milestone** status. Capability
tiers live in the [compatibility matrix](compatibility-matrix.md), and every
recorded run is indexed in [benchmarks/results](../../benchmarks/results/README.md).

A milestone is a reference-architecture target built on the SDK's contracts.
It is unrelated to the core export surface (`TrxFM`, `Flat`, `Decoder`,
`NoPE`, `EagerAttention`, `LastTokenPool`, `NextEvent`, `Binary`), which is a
version-stability promise. A milestone is **met** only when every criterion is
checked and its evidence is a record under `benchmarks/results/` from a
committed config, following the [evaluation protocol](../concepts/evaluation-protocol.md).
Smoke runs never count.

Both milestones share one headline: on PR-AUC, XGBoost on raw features plus
the TFM embedding (`xgboost_raw_embedding`) against XGBoost on the raw
features alone (`xgboost_raw`), with the same fixed setting for both
([ADR 0018](../architecture/0018-one-headline-comparison.md)).

| Milestone | Recipe | Status |
|---|---|---|
| 1. NVIDIA TFM Blueprint v1, rebuilt on the SDK | `blueprint_v1` | evidence at three seeds; the history embedding's lift over raw does not hold |
| 2. Blueprint v2: PRAGMA's encoders under a causal history decoder | `blueprint_v2`, compared with `pragma_like` | met: lift over raw on four of five tasks at three seeds |

## Milestone 1: Blueprint v1 on TabFormer

The target is v1's design on the [NVIDIA TFM Blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model),
not its bytes or published numbers. v1's checkpoint was trained with unshifted
labels and learned to copy its input, and its probe embedded each transaction
without history ([ADR 0017](../architecture/0017-v1-blueprint-on-the-sdk.md)).
The exact replay and its errata are on branch `archive/v1-replay`; v1's
artifacts are frozen at tag `v1.0.0`.

- [x] **Data.** TabFormer through the `tabformer` adapter with v1's date split
  (`tfm prepare-tabformer`).
- [x] **Tokenization.** v1's flat field layout with dense ids, less the
  constant `CUST` token: 11 fields per event, 12 tokens with the separator.
- [x] **Pretraining.** v1's Llama decoder (25.9M parameters; v1's own has 29M,
  from a larger vocabulary), `NextEvent` with targets shifted one position, on
  per-user 4096-token windows for about 3,000 steps, `position: rope`.
- [x] **Embeddings.** One per scored transaction, pooled at the last token of
  the user's recent history up to and including it
  (`evaluation.history_window`), beside v1's single-event embedding as a
  control.
- [x] **Downstream.** The headline arms with identifiers excluded from the raw
  features, and the hand-built history baseline `tabular_xgboost` beside them.
- [x] **Evidence.** `recipes/blueprint_v1/config.yaml` at three seeds on
  every test row: [`blueprint_v1_seeds`](../../benchmarks/results/blueprint_v1_seeds.md).
  The single-transaction embedding lifts raw-feature XGBoost by 145–192%
  PR-AUC at every seed. The history embedding, the headline, lifts it by −1%
  to +42%, so that lift is not claimed.
- [ ] **Headline lift.** The history embedding beating raw features at every
  seed. Open: a readout that keeps the scored transaction in focus.

## Milestone 2: Blueprint v2 on pragmatiq

PRAGMA's profile and event encoders feed v1's causal history decoder. v1 is
the baseline; PRAGMA is a comparison target (`pragma_like`, at matched size on
the same rows), not something to reproduce bit for bit.

- [x] **Composition ADR.** [ADR 0012](../architecture/0012-blueprint-v2-composition.md):
  every objective declares its label horizon, cutoff semantics, outputs,
  gradient behavior, and leakage checks.
- [x] **Tokenization.** Key-value-time tokens with the `8·ln(1+Δt/8)` time
  transform, fitted on the train partition only.
- [x] **Encoders.** Typed profile and event encoders, shared with
  `pragma_like`; TimeRoPE over log-seconds and calendar embeddings in the
  history decoder. Contract tests cover causal invariance, anti-copy, padding,
  past-only positions, and profiles cut at the cutoff.
- [x] **Recipe.** `blueprint_v2` with a CPU smoke config and a pragmatiq
  config, in the catalog and matrix.
- [x] **Backends.** Eager, and `ddp` passing the
  [ADR 0011](../architecture/0011-training-backend-contract.md) suite (Gloo on
  CPU; 2 and 4 NCCL ranks). No multi-GPU result is recorded.
- [x] **Downstream.** A history-window probe with histories rebuilt at each
  row's cutoff, the single-event control, and the spliced identity-theft fraud
  task ([ADR 0015](../architecture/0015-dataset-strategy-and-spliced-fraud.md))
  beside churn, default, LTV, and AML on the same users.
- [x] **Leakage.** Entity-disjoint fraud probe with per-side label windows
  ([ADR 0013](../architecture/0013-fraud-probe-split-integrity.md)).
- [x] **Evidence.** `recipes/blueprint_v2/config.yaml` and
  `recipes/pragma_like/config.yaml` at three seeds, with the headline lift on
  every task: [`blueprint_v2_seeds`](../../benchmarks/results/blueprint_v2_seeds.md).
  `blueprint_v2`'s embedding lifts raw-feature XGBoost at every seed on fraud
  (+9%), churn (+18%), LTV (+2%), and the per-transaction fraud probe (+26%);
  default changes sign across seeds, and AML has too few positives to score.
  Neither model reaches the hand-built `tabular_xgboost` reference.

## Not planned for either milestone

Mamba backbones, ranking objectives, explainability, and production serving.
They stay `planned` in the compatibility matrix and are not stubbed as public
API. Sharded training (`fsdp2`) is shown on `blueprint_v1` only; v2's TimeRoPE
decoder is a native module, and at its sizes data parallelism needs
replication, not sharding.
