# Blueprint v2 hybrid: tuning at lab size on one GB10

A staged search for one lab-size configuration whose embedding adds to a
weak-feature XGBoost. The question is whether a pretrained history model can
stand in for hand-built history features. The chosen settings are
[`config.lab_tuned.yaml`](../../recipes/blueprint_v2/config.yaml) (now `blueprint_v2/config.yaml`);
the same settings, with scaled rates, are carried to the ~100M
`config.pragmatiq.yaml`. This is a synthetic book, one pretraining seed, and a
search scored on a proxy. It is evidence for these settings on this book, not a
quality claim.

| field | value |
|---|---|
| data | pragmatiq Generator v2 book `synth_100k_gen2_m31_seed0` (100,000 users, 31 months, 128.5M events), as in [`blueprint_v2_hybrid_lab_kvt_seeds.md`](blueprint_v2_hybrid_lab_kvt_seeds.md). Probes use the 10,000-user sample, split, and spliced identity-theft task of `config.lab.yaml`; train cutoff 2024-06-01, validation cutoff 2024-09-01, label delay 0h |
| pretraining corpus | from stage 1 on: every 128-event train-side window of all 100k users from a token cache (451,886 windows), not one row per sampled user |
| model | profile / event / decoder depths 2 / 4 / 4, hidden 256, 8 heads, TimeRoPE, SDPA; 8.4M parameters plus the embedding table (12,870-token vocabulary with 4,096 buckets per hashed key) |
| chosen training | Muon lr 0.01 on hidden matrices, AdamW lr 1e-3 on the rest, batch 128, 1,500 steps (about 0.4 epoch), 75 warmup, cosine, clip 1.0, bf16, next-event fields + token/event/key MLM, pretraining seed 42 |
| hardware | one NVIDIA GB10 (DGX Spark), 121 GB unified memory |
| wall clock | 1,439 s to train the chosen checkpoint; about 5 min to prepare the probe data; final evaluation about 6.5 min per probe-split seed |
| code | commit `1b1a36e` plus the uncommitted changes that this record ships with |
| raw output | [`blueprint_v2_hybrid_gb10_tuning.json`](blueprint_v2_hybrid_gb10_tuning.json): diagnostics, every stage's proxy summary, the three final evaluations, the fixed-setting evaluations (`final_fixed`, with LoRA and the history reference in `final_lora`), and the fine-tuning sweep |

## The comparison

The headline arm is `xgboost_raw_embedding` against `xgboost_raw`. The weak
features are the last event's own fields plus the profile, with no history
aggregates: the kind of row a model gets before anyone engineers features.
`tabular_xgboost` is the strong reference: the same model on hand-built
history features (counts, amount statistics, and recency over the window).
Both are tuned by Optuna. Gain recovered is

    (raw + embedding − raw) / (tabular − raw)   on PR-AUC,

the share of the hand-built features' lift that the embedding supplies without
them. 100% means the embedding replaces the feature engineering on that task.

## Headline

Final evaluation (`benchmarks/final_eval.py`) of the chosen checkpoint over
three probe-split seeds, 30 XGBoost trials per arm, each probe trained on a
seeded 80% of its training entities. Forecast tasks get a fresh entity split
per seed. The fraud tasks keep the splice's fixed evaluation side, because
mixing sides would leak. PR-AUC, median [min, max]:

| task (eval rows, positives) | `xgboost_raw` | raw + embedding | `tabular_xgboost` | gain recovered |
|---|---|---|---|---|
| fraud (2,895, 386) | 0.201 [0.194, 0.202] | **0.379** [0.365, 0.390] | 0.522 [0.520, 0.524] | 52%, 55%, 59% |
| history-window fraud (2,885, 621) | 0.260 [0.251, 0.270] | **0.404** [0.403, 0.408] | 0.522 [0.505, 0.531] | 55%, 56%, 57% |
| churn in 6 months (2,352, 299) | 0.257 [0.239, 0.273] | 0.316 [0.296, 0.338] | 0.334 [0.313, 0.348] | 88%, 50%, 104% |

"raw + embedding" here is the `last_surprise` view. It joins the last slot's
embedding to the model's own next-event surprise: its cross-entropy on each key
of the row's latest 1 and latest 8 events. Each event is scored from its
prefix, so the view uses nothing after the cutoff. With the last slot alone,
raw + embedding is 0.223 on fraud (1–10% gain recovered), 0.272 on history
fraud, and 0.299 on churn. Most of the fraud lift is in the surprise, which is
what a theft looks like to a next-event model. Surprise is computed by
`tfm.evaluation.surprise` (moved there from `proxy_eval.py` after this record;
the arithmetic is unchanged).

**Random-initialization control.** The same architecture untrained, scored by
the same script (`--random-init`). Raw + embedding (`last_surprise`) recovers
0%, 8%, and 1% of the fraud gap and 8%, 6%, and −8% of the history-fraud gap,
so the fraud lift comes from pretraining. On churn, random features alone
recover −7% to 65% (with the last slot, 21% to 44%). The churn gain is
therefore weak evidence for pretraining.

The ROC-AUC medians follow the same order (fraud 0.666 raw, 0.744 raw +
embedding, 0.792 tabular); all values are in the JSON.

## Fixed setting on four tasks

The lab notebooks do not search. They score every raw arm with one XGBoost
setting, `tfm.runtime.lab.LAB_XGBOOST_PARAMS` (the median of the 18 searched
selections above), and early-stop the tree count on an entity-disjoint inner
fold. `benchmarks/final_eval.py --fixed-xgboost` repeats that protocol on the
same checkpoint, probe splits, and seeds, over the notebooks' four tasks:
`history_window` (shown as `fraud_txn` in the notebooks), `churn_6m`,
`ltv_positive`, and `default_12m`. There is no hand-built reference arm, so no
gain recovered. These runs are at commit `9f02858` plus the `--fixed-xgboost`
option they ship with; each took about 5 minutes to prepare plus 3 to 3.5
minutes per seed. PR-AUC, median [min, max]; row counts are split seed 0's:

| task (eval rows, positives) | `xgboost_raw` | raw + last slot | raw + `last_surprise` | raw + `last_surprise`, untrained |
|---|---|---|---|---|
| `fraud_txn` (2,885, 621) | 0.268 [0.252, 0.281] | 0.275 [0.272, 0.286] | **0.391** [0.387, 0.406] | 0.259 [0.248, 0.269] |
| `churn_6m` (2,352, 299) | 0.259 [0.247, 0.284] | **0.320** [0.317, 0.340] | 0.318 [0.316, 0.321] | 0.289 [0.281, 0.293] |
| `ltv_positive` (2,352, 1,679) | 0.901 [0.900, 0.908] | **0.928** [0.925, 0.931] | 0.926 [0.925, 0.930] | 0.918 [0.909, 0.920] |
| `default_12m` (2,811, 64) | 0.200 [0.195, 0.246] | 0.208 [0.188, 0.281] | 0.181 [0.152, 0.295] | 0.187 [0.170, 0.272] |

- On `fraud_txn` and `churn_6m` the fixed setting reproduces the searched
  numbers above to within about 0.02, so the notebooks' simpler protocol does
  not change the comparison.
- `ltv_positive` gains 2.2 to 2.7 points with the last slot on every seed; the
  untrained last slot gains 0.0 to 0.6. Most users are profitable (71%
  positive), so PR-AUC starts near its ceiling. ROC-AUC moves from 0.801 to
  0.850.
- `default_12m` shows no reliable lift. Per-seed lifts with the last slot are
  +3.5, −0.7, and +0.7 points, and with `last_surprise` +4.9, −4.3, and −1.9;
  the ROC-AUC median drops from 0.735 to 0.696. With about 64 evaluation
  positives, a larger sample is needed before reading anything into this task.

## LoRA beside the frozen arms, four tasks

`final_eval.py --fixed-xgboost --finetune recipes/blueprint_v2_hybrid/config.lab_tuned.yaml`
adds the recipe's fine-tuning arm (LoRA rank 8, alpha 8, on qkv, out, and both
feed-forward projections of every block, 327,680 trainable parameters, a
linear head, early-stopped on 20% of the training entities) to the run above.
It trains on the same rows and splits, with fine-tuning seed
`config.seed + split_seed`. With `--random-init` the same LoRA trains on an
untrained network: the random weights stay frozen, and the adapters and head
learn the task from its labels. That control asks whether the pretrained
weights help fine-tuning, not what random features score.

The run directory is `outputs/proxy/stage3/causal_adr0017`: the same
checkpoint, with the two `null` keys that ADR 0017 removed dropped from its
config. Every XGBoost value matched the previous section's exactly. The last
column is the hand-built history arm under the same fixed setting
(`benchmarks/fixed_history_reference.py`), as context. Commit `50f4931` plus
the options this record ships with; about 11 minutes per seed with LoRA.
PR-AUC, median [min, max]:

| task | `xgboost_raw` | raw + embedding + surprise | LoRA | LoRA, untrained network | hand-built history |
|---|---|---|---|---|---|
| `fraud_txn` | 0.268 | **0.391** | 0.312 [0.273, 0.322] | 0.266 [0.239, 0.301] | 0.535 [0.530, 0.537] |
| `churn_6m` | 0.259 | 0.318 | **0.344** [0.329, 0.376] | 0.307 [0.302, 0.360] | 0.345 [0.327, 0.367] |
| `ltv_positive` | 0.901 | 0.926 | 0.920 [0.918, 0.928] | **0.935** [0.933, 0.935] | 0.937 [0.937, 0.942] |
| `default_12m` | 0.200 | 0.181 | **0.254** [0.210, 0.269] | 0.191 [0.148, 0.240] | 0.265 [0.222, 0.298] |

- No adaptation wins everywhere. Frozen embedding + surprise into XGBoost
  is best on `fraud_txn`; LoRA is best on churn and `default_12m`.
- LoRA's lift over raw is positive on every seed on all four tasks:
  `fraud_txn` +2.1, +4.1, +4.4 points; churn +9.2, +8.5, +8.2;
  `ltv_positive` +2.0, +1.8, +2.0; `default_12m` +2.2, +1.4, +5.3 (ROC-AUC
  0.735 to 0.757). On `fraud_txn` its head sees neither the raw fields nor
  surprise.
- On the untrained network, LoRA matches raw on `fraud_txn` and falls below
  it on `default_12m`, so the pretrained weights matter there. On churn it
  gets about two thirds of the pretrained gain.
- On `ltv_positive` the untrained network does best, at the level of the
  hand-built history features. The label follows income and churn, which a
  sum over the history shows. A model fitted to the labels learns that from
  random weights, and starting from the next-event pretraining does not
  help. The frozen arms are where pretraining shows on this task: the
  pretrained last slot adds 2.7 points, the untrained one 0.5.
- The hand-built history arm is above every model arm on `fraud_txn`, and
  level with LoRA on churn and `default_12m`.

## Fine-tuning

`benchmarks/adapt_sweep.py` on the chosen checkpoint, one run per variant, on
the proxy's fixed split: the fraud evaluation side is the one above, and the
forecast split is the proxy's. Early stopping holds out 20% of the training
entities, checks validation log-loss every 50 steps with patience 4, and
restores the best weights. PR-AUC (ROC-AUC):

| variant | fraud | churn 6m | history-window fraud |
|---|---|---|---|
| LoRA as in `config.lab.yaml` (rank 8, alpha 16, qkv/out, 300 steps) | 0.423 (0.773) | 0.352 (0.759) | 0.293 (0.614) |
| LoRA rank 8, alpha 8, qkv/out/MLP, early-stopped, lr 3e-4 | 0.432 (0.777) | 0.369 (0.782) | 0.313 (0.633) |
| **LoRA rank 8, alpha 8, qkv/out/MLP, early-stopped, lr 1e-3** | **0.480** (0.783) | **0.376** (0.784) | 0.306 (0.628) |
| same, joint fusion with the weak raw features | 0.418 (0.763) | 0.363 (0.780) | 0.299 (0.618) |
| full fine-tune, lr 1e-4, early-stopped | 0.281 (0.731) | 0.354 (0.780) | 0.318 (0.632) |
| reference on this split: `xgboost_raw` / `tabular_xgboost` | 0.199 / 0.528 | 0.273 / 0.348 | 0.259 / 0.529 |

The bold row is the tuned configs' `fraud_task.finetune`. It stopped at steps
400–600 of 3,000. On fraud it reaches 0.480 PR-AUC with no hand-built features,
and on churn it passes the tabular reference on this split. On history-window
fraud, fine-tuning the last slot does worse than the frozen `last_surprise`
view (0.404), which the fine-tuning head does not see. Fusing the weak raw
features into the fine-tuning head did not help. These numbers are from one
fine-tuning seed and one split, so they rank variants rather than record a
lift.

## The search

Each stage trained its variants from scratch and scored them with
`benchmarks/proxy_eval.py`, using the probe split above with 8 XGBoost
trials, frozen-readout logistic probes, held-out next-event loss, and
embedding health. Specs are under `benchmarks/sweeps/`. A variant was chosen
on held-out loss and the fraud probes together; the proxy never touched the
final evaluation's resampled splits. PR-AUC below; "raw + emb" is the
last-slot view on the proxy (`last_surprise` was added in stage 2).

| stage | variant | fraud last | fraud `last_surprise` | history `last_surprise` | churn raw + emb | held-out loss | same-user retrieval@1 |
|---|---|---|---|---|---|---|---|
| 0: gate (byte-level merchant, 1 epoch) | random init | 0.204 | – | – | 0.288 | – | – |
| | one row per user | 0.248 | – | – | 0.343 | 4.17 | – |
| | every window | 0.232 | – | – | 0.398 | 4.02 | – |
| 1: AdamW lr (shared hash range) | 5e-4 | 0.216 | 0.283 | 0.367 | 0.296 | 4.240 | 0.948 |
| | **1e-3** | 0.268 | 0.335 | 0.362 | 0.282 | 4.132 | 0.972 |
| | 2e-3 | 0.251 | 0.337 | 0.374 | 0.309 | 4.154 | 0.962 |
| 2: Muon | **0.01** | 0.257 | 0.318 | 0.376 | 0.301 | 4.047 | 0.872 |
| | 0.02 | 0.238 | 0.245 | 0.339 | 0.332 | 4.122 | 0.833 |
| 3: objective (per-key hash ranges) | random init | 0.190 | 0.193 | 0.271 | 0.276 | – | 0.231 |
| | **causal** | 0.271 | 0.303 | 0.353 | 0.341 | 4.010 | 0.905 |
| | bidirectional MLM (profile/history 2/4/4) | 0.192 | – | – | 0.328 | not comparable | 0.313 |
| 4: 512-event context | causal | 0.207 | 0.247 | 0.403 | 0.336 | 4.019 | 0.932 |
| 5: 3× steps (4,500) | causal | 0.226 | 0.233 | 0.360 | 0.352 | 3.898 | 0.706 |

Stage 1's `last_surprise` values come from stage 2's rescoring of the same
checkpoints. Held-out losses compare only within one tokenizer: byte-level in
stage 0, one shared hash range in stages 1–2, and one range per key from
stage 3 on. What each stage decided:

- **Stage 0: tokens before tuning.** Spelled byte by byte, the merchant's
  next-event cross-entropy was 3.31 nats per byte against a unigram floor of
  3.33. The model did not learn the most informative field, and raw +
  embedding did not beat raw on fraud. From stage 1 on, merchant, device, and
  counterparty are one hash-bucket token each, and `app_version` is a
  category. Every event then fits in 9 tokens (p50 7), down from p50 20 and
  max 39.
- **Stage 1: AdamW lr 1e-3** had the lowest held-out loss and the best fraud
  last-slot probe.
- **Stage 2: Muon at 0.01** lowered held-out loss further (4.047, final
  training loss 3.795 against 3.936) with similar probes; 0.02 was worse on
  both. The unmasked per-key loss showed the three hashed keys doing poorly
  in one shared bucket range. The tied output embedding cannot give one
  bucket three meanings, so each hashed key got its own range (below).
- **Stage 3: causal next-event training.** Bidirectional MLM probes were no
  better than random initialization on fraud (0.192 against 0.190), and its
  held-out loss is a masked-token loss, not comparable. Causal was kept.
- **Stage 4: 512 events per row.** History-window fraud rose (0.403), but
  fraud fell (0.247), and the tabular reference also got stronger on 512
  events (history 0.618). 128 was kept.
- **Stage 5: three times the steps.** Held-out loss fell to 3.898 and the
  merchant cross-entropy fell to 6.28, but same-user retrieval fell from
  0.905 to 0.706, and the fraud readouts dropped. The final evaluation agrees
  on fraud (raw + embedding 0.345 at 3×, against 0.379 at 1×). It went the
  other way on history-window fraud (0.418 against 0.404) and churn (0.324
  against 0.299 on the last slot). Lower pretraining loss did not mean a
  better fraud embedding, so the 1× checkpoint was chosen.

### Token diagnostics for the chosen tokenizer

`benchmarks/diagnostics_tokens.py` on 2,000 users (1.02M train-side events,
1.58M after the cutoff). Floors are the unigram distribution and "repeat the
key's previous value" (copy). The model column is the 3× checkpoint's held-out
next-event cross-entropy.

| key | events | unigram CE | copy CE | model CE |
|---|---|---|---|---|
| `merchant` (hashed) | 41.2% | 6.58 | 7.04 | 6.28 |
| `mcc` | 41.2% | 2.90 | 3.28 | 2.59 |
| `amount` | 41.2% | 2.77 | 3.21 | 2.64 |
| `device_id` (hashed) | 55.9% | 7.96 | 0.42 | 0.63 |
| `app_version` | 55.9% | 3.58 | 0.42 | 0.32 |
| `counterparty` (hashed) | 4.3% | 8.09 | 4.80 | 9.07 |
| `campaign_id` | 2.5% | 8.32 | 2.63 | 19.67 |

The model beats both floors on merchant, mcc, amount, and app_version. It does
not reach the copy floor on device_id, and it is worse than either floor on
counterparty. Every `campaign_id` after the cutoff is a value unseen in
training (100% UNK), so its loss measures that shift, not the model.

## Engineering findings

- **One hash range per key.** With the three hashed keys sharing one bucket
  range, a bucket stood for a merchant, a device, and a counterparty at once.
  The output layer is tied to the input embedding, so it could not predict
  any of them well. Each hashed key now has its own range
  (`VAL_<key>_h<i>`), the cache identity records the layout, and caches from
  the shared layout refuse to load.
- **4,096 buckets per key, not 8,192.** At a 25k vocabulary, the full-vocabulary
  next-event softmax over batch 128 × 128 events did not fit next to the
  prepared book in 121 GB of unified memory, and the run was killed. The
  proxy also computes per-key cross-entropy in 8,192-token slices.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on a GB10.** Without
  it, the first 4,500-step run was killed. On variable-length batches the
  caching allocator's reservation grew to about 95 GB against a 21.6 GB
  allocated peak. With it, the reservation stayed near 25 GB.
- **Early stopping on log-loss, not PR-AUC.** A ranking metric plateaus while
  scores are still flipping sign, so a PR-AUC stop kept an inverted head in a
  unit test. `fraud_task.finetune.validation_fraction` holds out whole
  training entities.

## Settings carried to `config.pragmatiq.yaml`

The ~100M config (d=512, depths 3/16/12, 256-event rows, 50,000 users) was
not run. It takes the settings that transfer without a search:

| setting | lab (searched) | ~100M (transferred) | why |
|---|---|---|---|
| identity keys | hashed, 4,096 per key | same | the byte-level merchant was not learned |
| `app_version` | categorical | same | |
| `max_tokens` | 40 (cache identity) | 16, was 24 | events are at most 9 tokens once hashed |
| corpus | every 128-event window, all users (cache) | every 256-event window of the sample | the cache holds no labels, so it requires the spliced fraud task; the 100M config keeps the book's own labels |
| Muon lr | 0.01 | 0.01 | Muon's update is normalized per matrix |
| AdamW lr | 1e-3 at d=256 | 5e-4, was 2e-4 | halved for twice the width |
| precision, clipping, warmup | bf16, 1.0, 5% | bf16 (was fp32), 1.0, 5% | |
| fine-tuning | LoRA r8 α8, qkv/out/MLP, early-stopped | same | |

## Limitations

- **Synthetic book.** pragmatiq's personas and injected episodes are
  simulated, and the fraud task is a splice (ADR 0015). Real transactions
  differ in ways this cannot show.
- **One pretraining seed.** The three seeds vary the probe split and the
  probe's training rows, not pretraining. The 1×/3× comparison is also one
  seed each.
- **Selection on a proxy.** Variants were chosen on the proxy's single split,
  which shares the fraud evaluation side with the final evaluation. The
  final numbers are therefore not a clean held-out test of the selection.
- **The readout was chosen on the proxy too.** `last_surprise` was added
  after stage 1 because it scored well. It lives in `benchmarks/proxy_eval.py`,
  not in the recipe's evaluation, so `report.py` on the tuned config reports
  the last-slot readout, which carries much less of the fraud lift.
- **Churn lift is partly architecture.** Random-init features recover up to
  65% of the churn gap.
- **Fine-tuning numbers are single-run.** They rank variants; they are not a
  lift with a spread.
- **Counterparty and campaign tokens are unresolved:** counterparty is worse
  than its unigram floor, and every post-cutoff campaign id is unseen.
- **Not run:** the tuned lab config end to end through `report.py`, and
  anything at ~100M. Peak host memory was not recorded per stage.
