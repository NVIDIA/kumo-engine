# Blueprint v2 final sweep: pretraining budget and LoRA capacity (one GPU, three seeds)

The last sweep before the Blueprint v2 code freeze. It asks which of four
settings of `recipes/blueprint_v2/config.yaml` gives the widest lift over
raw-feature XGBoost, and which single arm and seed to show as the reference
example. Two settings were varied: pretraining steps (1,500 or 3,000) and LoRA
rank (8 or 16). The selection rule was written down before the first run.
This is not a quality, fraud, or production claim.

| field | value |
|---|---|
| config | `recipes/blueprint_v2/config.yaml`; seeds 42, 43, 44 |
| candidates | **A** no overrides. **B** `training.max_steps=3000`, `training.optimizer.warmup_steps=150`. **C** A's checkpoints re-evaluated with `fraud_task.finetune.rank=16`, `fraud_task.finetune.alpha=16.0`. **D** B's checkpoints re-evaluated with the same LoRA overrides |
| code | commit `124ab93`, run from a detached worktree with a clean tree; C and D used `benchmarks/seed_range.py --reuse-checkpoint` on copies of A's and B's run directories |
| data | `synth_100k_gen2_m31_seed0`; pretraining on every 128-event train-side window of all 100,000 users (token cache); probes on 10,000 users, most recent 128 events each; train cutoff 2024-06-01, validation cutoff 2024-09-01 |
| fraud task | spliced identity theft, 20% victims, matched donors ([dataset strategy](../../docs/reference/datasets.md)) |
| model | profile / event / decoder depths 2 / 4 / 4, hidden 256, 8 heads, TimeRoPE, SDPA, varlen; next-event fields plus token/event/key MLM; last-token readout; 11,513,856 parameters |
| training | eager, bf16, batch 128, Muon (LR 0.01) and AdamW (LR 1e-3), cosine schedule, 5% warmup; no non-finite steps. Loss at the last step: A 3.76–3.90, B 3.53–3.76 |
| probes | every XGBoost arm uses the config's fixed setting; LoRA on qkv, out, and the feed-forward layers with a linear head, early-stopped on 20% of the training entities |
| hardware | one NVIDIA GB10 (DGX Spark), torch 2.13.0+cu130, Python 3.12.14 |
| wall clock per seed | prepare 258–267 s for every candidate. Train: A 1,271–1,298 s, B 2,536–2,545 s. Evaluate 1,028–1,302 s. The four candidates ran one after another in 8 h 00 min |
| memory | 21.5–21.6 GiB peak GPU; host memory not measured |
| raw output | [`blueprint_v2_final_sweep.json`](blueprint_v2_final_sweep.json): each candidate's `seed_range.py` summary with every arm per seed, and the selection outcome |

## Selection rule

Fixed before any candidate ran:

- Headline tasks: fraud, transaction-level fraud (the history window), and
  churn in 6 months on PR-AUC. LTV positive on ROC-AUC: about 71% of its users
  are positive, so its PR-AUC starts near 0.91. Lift is \(x / x_{\text{raw}} - 1\)
  over `xgboost_raw` on the same seed's rows.
- Arms: `lora_<task>`, and `<task>_xgboost_raw_embedding:surprise` (raw +
  embedding + next-event surprise). One arm is used across all four tasks.
- An arm is eligible if its lift is positive at every seed on every headline task.
- B, C, or D replaces A only if one of its eligible arms beats the same arm of
  A on median lift on at least three of the four tasks.
- The winner's arm is its eligible arm with the higher worst seed-and-task lift.
- Default in 12 months (60–64 positives) and AML (one or two positives) are
  reported, never scored.

## Results

Lift over `xgboost_raw`, median [min, max] over the three seeds. Fraud,
transaction-level fraud, and churn are PR-AUC; LTV is ROC-AUC. All eight
candidate arms are eligible.

| Candidate | Arm | Fraud | Fraud, per txn | Churn 6m | LTV (ROC-AUC) |
|---|---|---|---|---|---|
| A | surprise | +115% [+95%, +140%] | +68% [+55%, +93%] | +20% [+5%, +22%] | +5% [+4%, +6%] |
| A | LoRA | +144% [+139%, +167%] | +36% [+16%, +48%] | +28% [+25%, +41%] | +5% [+3%, +6%] |
| B | surprise | +119% [+86%, +130%] | +81% [+65%, +90%] | +19% [+19%, +27%] | +5% [+5%, +5%] |
| B | LoRA | +156% [+126%, +168%] | +33% [+28%, +63%] | +31% [+30%, +38%] | +4% [+4%, +5%] |
| C | surprise | same as A | same as A | same as A | same as A |
| C | LoRA | +146% [+108%, +173%] | +39% [+15%, +49%] | +28% [+24%, +29%] | +4% [+3%, +4%] |
| D | surprise | same as B | same as B | same as B | same as B |
| **D** | **LoRA** | **+163% [+89%, +183%]** | **+50% [+33%, +77%]** | **+39% [+33%, +44%]** | **+5% [+3%, +7%]** |

The surprise arm reads only the pretrained checkpoint, so C's equals A's and
D's equals B's to the last digit.

Median PR-AUC (ROC-AUC for LTV), with the hand-built reference beside it:

| Task | `xgboost_raw` | D surprise | D LoRA | `tabular_xgboost` (reference) |
|---|---|---|---|---|
| fraud (386, 379, 375 positives) | 0.184 | 0.403 | 0.484 | 0.532 |
| fraud, per txn (618–640 positives) | 0.265 | 0.470 | 0.408 | 0.586 |
| churn 6m (295–299 positives) | 0.264 | 0.324 | 0.381 | 0.352 |
| LTV positive, ROC-AUC (1,679–1,733 positives) | 0.818 | 0.862 | 0.861 | 0.877 |

### Outcome

- **D wins.** Its surprise arm beats A's on fraud, transaction-level fraud, and
  LTV, and trails it on churn (+19% against +20%). Its LoRA arm also beats A's
  on fraud, transaction-level fraud, and churn, and trails it on LTV (+5.3%
  against +5.4%). B qualifies on the same surprise arm, but its LoRA arm beats
  A on only two tasks, so D is at least as good as B on every arm.
- **The arm is surprise.** Its worst seed-and-task lift is +4.9%, against
  LoRA's +3.4%. LoRA has the larger medians on fraud and churn, but a wider
  spread: D's LoRA fraud lift is +89% at seed 42 and +163% to +183% at the other
  two seeds.
- **D's LoRA passes the hand-built reference on churn** at every seed (0.356
  to 0.399 against 0.351 to 0.370 PR-AUC). On every other task, every model arm
  stays below `tabular_xgboost`.
- **Twice the pretraining steps helped the frozen arms.** Going from A to B
  raised the surprise arm's transaction-level fraud lift from +68% to +81%. The
  plain embedding's fraud lift moved from +9% [+9%, +12%] to +12% [−1%, +33%]:
  a higher median, but negative at one seed. Rank 16 helped only on the
  3,000-step checkpoints (D against B). On A's checkpoints, C against A, it
  changed little.

### Reference example: D, surprise arm, seed 42

Seed 42 is the recipe's declared seed. For fraud, it is D's weakest surprise
seed.

| Task | Metric | `xgboost_raw` | D surprise | Lift | Three-seed lift |
|---|---|---|---|---|---|
| fraud | PR-AUC | 0.201 | 0.374 | +86% | +86% to +130% |
| fraud | ROC-AUC | 0.659 | 0.751 | +14% | +14% to +18% |
| fraud, per txn | PR-AUC | 0.265 | 0.436 | +65% | +65% to +90% |
| fraud, per txn | ROC-AUC | 0.573 | 0.700 | +22% | +22% to +27% |
| churn 6m | PR-AUC | 0.301 | 0.358 | +19% | +19% to +27% |
| churn 6m | ROC-AUC | 0.695 | 0.748 | +8% | +8% to +9% |
| LTV positive | ROC-AUC | 0.813 | 0.856 | +5% | +5% to +5% |
| LTV positive | PR-AUC | 0.908 | 0.927 | +2% | +2% to +2% |

Default in 12 months, low support (64 positives at seed 42): PR-AUC 0.263 raw
against 0.275 (+4%), but ROC-AUC 0.774 against 0.728 (−6%). Across seeds the
surprise arm's default PR-AUC lift is −1% to +21%, so it is not a result.

### Configs left unchanged

`recipes/blueprint_v2/config.yaml` keeps 1,500 steps and LoRA rank 8 for the
freeze. The winning settings are the overrides above, committed as
[`benchmarks/ablations/blueprint_v2_final_d.yaml`](../ablations/blueprint_v2_final_d.yaml),
which a unit test holds equal to the recipe on every other key. D's seed-42
checkpoint is the GTC Berlin lab's `blueprint_v2_lab_tuned`. `pragma_like/config.yaml`
is held equal to it on `training` and `fraud_task`
(`tests/unit/test_pragma_configs.py`). Adopting D in the recipe would mean
rerunning `pragma_like` at the same settings. The pragmatiq notebooks use D's
seed-42 checkpoint (`tfm.runtime.lab.LAB_CONFIG`).

## Not shown

- **Selection and scoring use the same evaluation rows.** The rule was fixed
  in advance and requires a lift at every seed, but no held-out set was kept
  back to confirm the winner.
- **`pragma_like` was not rerun.** Its config is held equal to `blueprint_v2`'s
  on `training` and `fraud_task`, and the matched comparison in
  [`blueprint_v2_seeds`](blueprint_v2_seeds.md) uses 1,500 steps and rank 8 for
  both models.
- **A reproduces the earlier record closely, not exactly.** That record came
  from a working-tree snapshot on top of `e871249`. For example, LoRA fraud at
  seed 42 is 0.480 here against 0.479.
- One synthetic book, one model size, three seeds, one GPU, no multi-GPU run.
  pragmatiq results are never combined with TabFormer results.

Reproduce, from a clean checkout of `124ab93`:

```bash
export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv sync --extra baselines
uv run python benchmarks/seed_range.py --config recipes/blueprint_v2/config.yaml \
  --seeds 42 43 44 --run-root outputs/final_sweep/B --summary outputs/final_sweep/B.json \
  --device cuda --set training.max_steps=3000 --set training.optimizer.warmup_steps=150
cp -r outputs/final_sweep/B outputs/final_sweep/D
uv run python benchmarks/seed_range.py --config recipes/blueprint_v2/config.yaml \
  --seeds 42 43 44 --run-root outputs/final_sweep/D --summary outputs/final_sweep/D.json \
  --device cuda --reuse-checkpoint --set training.max_steps=3000 \
  --set training.optimizer.warmup_steps=150 \
  --set fraud_task.finetune.rank=16 --set fraud_task.finetune.alpha=16.0
uv run python benchmarks/lift_table.py outputs/final_sweep/D.json --reference xgboost_raw
```

A and C are the same commands without the step overrides; C copies A's run
directories. [`benchmarks/blueprint_v2_final_sweep.sh`](../blueprint_v2_final_sweep.sh)
runs all four candidates in this order, and
[`benchmarks/blueprint_v2_final_sweep.py`](../blueprint_v2_final_sweep.py)
applies the selection rule to their summaries and writes `score.json`.
