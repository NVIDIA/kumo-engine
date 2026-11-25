# Datasets

The repository reads four kinds of data. Each has one job, and results from
different datasets are never combined. If
you are new, start with TabFormer through the `blueprint_v1` notebooks;
use pragmatiq when you move to Blueprint v2.

Which recipe reads which dataset is the `data:` field of
[`recipes/catalog.yaml`](../../recipes/catalog.yaml). Nothing here is
downloaded by CI, and no dataset file is committed
([data governance](../governance/data-governance.md)).

| | Generated fixtures | IBM TabFormer | pragmatiq book | PersonaLedger |
|---|---|---|---|---|
| role | tests, smoke runs, CPU notebooks, the v2 lab's smoke mode, generated systems cost controls | Blueprint v1 reproduction; newcomer teaching series | Blueprint v2: every v2 task on the same users; the v2 lab notebooks | planned external benchmark (not implemented) |
| what it is | seeded events from `tfm.data.synthetic` | public synthetic card transactions from IBM | a synthetic bank, generated locally by pragmatiq 1.1.0 | LLM-generated card ledgers from Capital One Research |
| entities | configurable, small | 2,000 users, about 24.4M transactions | 100,000 users, 128.5M events (`synth_100k_gen2_m31_seed0`) | about 23,000 users, 30M transactions |
| channels | card transactions | card transactions | card and account transactions, app sessions, trading, communications, transfers, profile | card transactions and payments |
| tasks | fraud (smoke) | fraud, event-level | churn, LTV, default, AML (user-level); spliced identity theft (event-level) | identity theft (event-level); illiquidity (user-level) |
| split | configured per test | v1's: 80% / 90% of cumulative daily transactions | from the label calendar: `train_end` 2024-06-01, `validation_end` 2024-09-01 | published train/test, not yet verified |
| licence | this repository's | IBM/TabFormer repository: Apache-2.0 | pragmatiq: Apache-2.0; the book is generated output | data CC BY 4.0; code Apache-2.0 |
| local path | none | `TFM_TABFORMER_ROOT` | `TFM_PRAGMATIQ_ROOT`, or the lab notebooks' download (`tfm.runtime.lab`; URL a placeholder until hosted) | none yet |

## Never mix datasets

A foundation model's claim is that one representation of a person, learned
from all of their channels, serves many tasks. That can only be shown on one
population. So:

- A recipe config reads one dataset.
- A checkpoint is evaluated on the dataset it was pretrained on.
- A results record under `benchmarks/results/` covers one dataset and names it
  in its header.
- Numbers from different datasets are not averaged, pooled, or placed in one
  table unless every row names its dataset and the text says they are not
  comparable.

TabFormer and pragmatiq describe different people. A TabFormer fraud score and
a pragmatiq churn score are two models' results, not one model's.

## IBM TabFormer

Public synthetic card transactions ([IBM/TabFormer](https://github.com/IBM/TabFormer)):
one row per card transaction with user, card, time, amount, merchant, MCC,
location, chip use, errors, and a fraud flag. It is one channel and one task.
The v1 Blueprint was built on it, and `blueprint_v1` rebuilds v1's
flow on it with the SDK (Blueprint v1 guide).

- **Get it:** download `transactions.tgz` from the IBM/TabFormer repository,
  then `uv run tfm prepare-tabformer --root ~/data/TabFormer` and
  `export TFM_TABFORMER_ROOT=~/data/TabFormer`. The recipe README has the
  details.
- **Split:** v1's. The book is sorted by date and cut where the cumulative
  daily count reaches 80% and 90% (2017-05-14 and 2018-10-12). The split is by
  date across all users, so a user's history can span all three partitions;
  1,584 of the 2,000 users have training rows.
- **Labels:** the book's own fraud flag, one per transaction (about 0.12% of
  training rows). The recipe's task is transaction-level; few users have a
  fraud among their last 128 transactions, so the entity-level fraud probe is
  thin on this book.

## pragmatiq books

A pragmatiq book is a synthetic bank generated locally by
[pragmatiq](https://github.com/dynamiq-ai/pragmatiq), an independent
implementation by Dynamiq inspired by the PRAGMA paper (arXiv 2604.08649); it
is not affiliated with or endorsed by Revolut. It has several channels per
user (transactions, app sessions, trading, communications), a profile, a
transfer graph, and labels for several tasks on the same users. It is the only
dataset here that can show one representation reused across tasks.

- **Get it:** from a pragmatiq 1.1.0 checkout, generate the book with the world
  config this repository commits; the command is in the header of
  [`configs/data/pragmatiq/world.yaml`](../../configs/data/pragmatiq/world.yaml)
  (about 77 seconds with 16 workers). Then
  `export TFM_PRAGMATIQ_ROOT=~/data/pragmatiq/data/synth_100k_gen2_m31_seed0`.
- **Used unmodified.** This repository does not change pragmatiq's code. A
  book is regenerated from a released pragmatiq and the committed world
  config.
- **Split:** pragmatiq issues its forecast labels on fixed dates set by the
  world config: `default_12m` on 2024-07-01 and `churn_6m` and `ltv_positive`
  on 2025-01-01. Pretraining must end before the earliest of them
  ([probe split rules](../concepts/evaluation-protocol.md)), so the
  cutoffs come from that calendar, not from an 80% rule. `train_end`
  2024-06-01 and `validation_end` 2024-09-01 leave 40%, 15%, and 45% of the
  book's events in the three periods. An 80% cut would fall on 2025-02-22,
  after every forecast label's issue date. Probes are entity-disjoint.

### Tasks on a pragmatiq book

| task | level | source | notes |
|---|---|---|---|
| `churn_6m`, `ltv_positive` | user | pragmatiq's labels | issued 2025-01-01, 6 months of outcome |
| `default_12m` | user | pragmatiq's labels | issued 2024-07-01, 12 months of outcome |
| `aml` | user | pragmatiq's labels | about 62 positive users per 100,000; too few to report |
| spliced identity theft | event | built by this repository from the book | the v2 fraud task (below) |
| pragmatiq's own `fraud` | event | pragmatiq's labels | smoke label only: one transaction identifies it |

### Why pragmatiq's own fraud labels are not a fraud result

On `synth_100k_gen2_m31_seed0`, a classifier on the labelled transaction's own
fields scores 1.000 ROC-AUC. Every labelled fraud event is an online payment
abroad, against 3% abroad and 30% online among the labelled negatives
([`pragmatiq_fraud_separability.md`](../../benchmarks/results/pragmatiq_fraud_separability.md)).
The generator has no setting that changes this. A task that one transaction
answers cannot show what a model learns from history.

### Spliced identity theft

The v2 fraud task follows PersonaLedger's method on an unmodified pragmatiq
book. For a victim and a donor, two users with no pragmatiq fraud label, the
donor's card payments and ATM withdrawals and app sessions on one day are
copied into the victim's history at their original times. Those copied
transactions are fraud; the victim's own card transactions in the days before,
and on that day, are legitimate. By default the donor has the same home
country as the victim and a similar typical amount, so each stolen
transaction is an ordinary transaction for someone like the victim.

On the same book, with matched donors, the transaction's own fields and the
profile score at chance (0.50 ROC-AUC); history features lift it to 0.84–0.85.
The victim's history is what gives the theft away: 80% of stolen payments go
to a merchant the victim has never paid, against 29% of the victim's own.

Leakage rules (dataset strategy, decision 5): the victim and donor are in the same
entity split; the theft day lies inside that side's label window; pretraining
uses unspliced histories; forecast tasks are scored on unspliced histories.

Select it with `data.fraud_labels: spliced_identity_theft` on a pragmatiq
book; `data.identity_theft` sets the victim share (0.2 of each probe side),
the negative window (7 days), how close to the side's cutoff theft days fall
(28 days, so they stay inside the most recent `sequence_length` events), and
matched or random donors. Users with the book's own fraud take no role and
join neither probe side, and the sample is drawn uniformly rather than
stratified by the book's fraud. Each run writes the victims, donors, theft
days, and label counts to `identity_theft_manifest.json`. The generated
pragmatiq fixture is too small to splice; tests write their own book.

## PersonaLedger

[PersonaLedger](https://github.com/CapitalOne-Research/PersonaLedger)
([arXiv 2601.03149](https://arxiv.org/abs/2601.03149),
[Hugging Face](https://huggingface.co/datasets/capitalone/PersonaLedger)) uses
an LLM conditioned on NVIDIA Nemotron-Personas to write each simulated day,
with a rule engine that keeps balances, bills, and credit limits consistent.
Each event has five fields: timestamp, merchant name, merchant type, card
present, and amount. Its identity-theft task is the method the spliced
pragmatiq task follows.

**Status: planned, not implemented.** Its role is an external benchmark: the
v2 model scored on data this repository did not generate, reported in its own
records. It is one channel with two tasks, so it cannot replace pragmatiq.
Before an adapter lands:

- **Split:** the published task files carry a sequence id but no user id, so
  whether train and test share users cannot be checked from them. The raw
  generation files are the place to check.
- **Timestamps** in the rows we inspected fall on whole and half hours, as
  the LLM wrote them; a time encoding that assumes seconds-level gaps will see
  ties.
- **Licence:** the data is CC BY 4.0, which requires attribution in any
  record or notebook that uses it.

## Experimental graph recipe views

`gnn_encoder` and `gnn_sequence` default to generated transaction/transfer
fixtures, retaining the generated-fixture role above. They can read a local
pragmatiq book as a separate configured run, retaining that book's role. They
add no dataset or external download. An induced graph contains only selected
owners, their event rows, and transfers whose endpoints both belong to that
population. User roots have a constant bias; static profile attributes are not
consumed in the initial graph recipes.

FutureSubgraph is a self-supervised future-activity target, independent of
fraud labels. Its windows close by the configured pretraining boundary with
explicit source coverage and label delay. The downstream task is the existing
any-available-positive fraud sequence label on a later, truncated history.
Pretraining uses unspliced observations, and these recipes do not apply the
identity-theft splice. Root-owner probe splits allow cutoff-bounded transfer
context across owners and must not be described as isolated graph components.


Graph recipe binary arms require a nonempty pragmatiq fraud label source and
available source coverage for each sampled owner by the task cutoff. Missing,
empty, unmatched or unavailable label sources are rejected before training.
The numeric sequence-label rule remains any available positive in the truncated
history; known all-negatives and app-only retained histories remain valid. Owner
sampling stays independent of labels. This requirement belongs to the binary
recipe arms; generic unlabeled graph sampling and FutureSubgraph helpers do not
require fraud labels.
