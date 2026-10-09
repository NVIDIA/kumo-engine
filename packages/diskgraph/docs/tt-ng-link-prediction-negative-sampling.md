# TT-NG Link-Prediction: Global Negative RHS Sampling (engine side)

> **Audience.** A principal engineer who knows ML broadly but has *not* read this
> codebase. This document explains, from the ground up: what link prediction (LP)
> training needs, why the *sampling engine* (not Python) must produce negatives,
> how the legacy system did it, what TT-NG does instead, the exact APIs and data
> contracts, the code/data flow, performance, testing, and the boundaries of this
> change. This design describes the original implementation; §13 links to current
> sources. Where a decision could go either way, the rationale and any caveat
> is recorded in §12.

---

## 1. What problem this solves

### 1.1 Link prediction, in one paragraph
A link-prediction task asks: *given a left-hand-side entity (LHS, e.g. a `User`),
which right-hand-side entities (RHS, e.g. `Item`s) will it link to?* Training data
is a set of observed positive links `(u, v)` — "user `u` interacted with item `v`."
A model that only ever sees positives learns nothing discriminative (it could score
everything 1.0). It must also see **negatives**: `(u, w)` pairs that are *not* links.
For each positive `(u, v)` we draw `M` negatives `(u, w₁…w_M)`; the loss contrasts
the positive's score against the negatives'.

### 1.2 Why the *engine* must sample negatives (not Python)
Kumo trains several LP model families. The one that drives this work is the **deep
two-tower**: it encodes each RHS node by running a GNN over that node's sampled
neighborhood (it does *not* keep a static per-RHS embedding table — that's the
*shallow* model). Consequently a negative `w` is not just an id: its **neighborhood
must be sampled and pushed through the GNN** to get its embedding. So negatives have
to be **sampling seeds**, exactly like positives. The two-tower cannot afford to
GNN-encode the entire RHS catalog per batch, so it relies on a bounded `M` global
negatives per positive, drawn fresh each batch.

Contrast the **shallow-RHS** model: it has an `nn.Embedding` table over all RHS and
computes a (sampled) global softmax — it never seeds negative *nodes*, so it gets
"global negatives" for free from the table. **This feature is therefore for the deep
two-tower path.** The engine produces negatives generically; only the two-tower
consumes them as seeds.

### 1.3 Temporal correctness (the leakage trap)
For *temporal* LP, a negative `w` drawn for a positive observed at time `t` must
have **existed at `t`** (you cannot contrast against an item created in the future),
and `w`'s own sampled neighborhood must not contain edges newer than `t`. Both are
causal-leakage guards. This is why negatives carry the positive's timestamp as their
sampling **anchor**, and why we additionally validate the negative *node itself*
against `t`.

---

## 2. What the legacy (PyG) system did

In v2.21, PyG triplet negative sampling applied to the non-default `TWO_TOWER`
module, not link prediction generally. `SHALLOW_RHS`, `ID`, and `HYBRID` used
the LHS node loader. See the release's
[loader selection](https://github.com/kumo-ai/kumo/blob/v2.21.0/kumo/train/datamodule/util.py)
and [module definitions](https://github.com/kumo-ai/kumo/blob/v2.21.0/kumo/config/trainer.py):

- **Distribution: uniform, global.** Negatives are drawn uniformly at random over the
  RHS id space — no degree/popularity weighting. (`neg_sampling=dict(mode='triplet', amount=M)`.)
- **Count: `M = rhs_sample_size or 1`** — defaults to **1**. PyG draws `ceil(B·amount)`.
  (`rhs_sample_size` is overloaded — for the shallow model it means the sampled-softmax
  pool size; for the deep triplet path it is `M`. We surface it as an explicit `amount_m`.)
- **No temporal guard on negative draws.** Kumo's
  [v2.21 sampler](https://github.com/kumo-ai/kumo/blob/v2.21.0/kumo/engine/neighbor_sampler.py)
  called `edge_sample(..., neg_sampling=neg_sampling)` without `node_time`.
  In the [pinned Linux PyG revision](https://github.com/pyg-team/pytorch_geometric/blob/370181c7751892e017844382deb0298cc8c67782/torch_geometric/sampler/neighbor_sampler.py),
  `node_time=None` bypasses temporal rejection/resampling. The draw could
  therefore select a negative that did not exist at the anchor time. This is
  separate from temporal filtering during neighborhood traversal.
- **Consumption is index-based.** The model gathers
  `rhs_emb[dst_pos_index]` / `rhs_emb[dst_neg_index]`. The *physical order* of RHS nodes
  in the batch does not matter to the model — the index tensors define which rows are
  positive vs negative. **So the layout is ours to define** (see §5.3).
- **Negative sharing is a head concern**; candidate temporal validity is a
  separate sampler concern (see §6).

TT-NG's candidate checks and resample-then-mask behavior are stricter than
v2.21 when anchors and RHS temporal metadata are available. This is a behavior
change, not temporal parity; two-tower metrics can legitimately differ.

---

## 3. The TT-NG sampling substrate (what this builds on)

### 3.1 One unified, multi-type request
TT-NG resolves an LP seed row — a *pair* (`ENTITY1 → lhs`, `TARGET1 → rhs`) — into a
single sampling request with multi-type seeds:
`seed_nodes = { lhs: [u₀…u_{B-1}], rhs: [v₀…v_{B-1}] }`, row-aligned (the positive pair
is the diagonal). `sample()` roots a hop-0 frontier at **every** seed type. **One
request, never two loaders.** (`engine.rs` `sample_runtime_batch` builds `seed_nodes`;
`sample()` roots all types.)

### 3.2 "Unified traversal" ≠ "entangled roots"
There is one BFS loop over one frontier and one set of `state.type_states` (keyed by
**node type id**), with output pooled by type into one HeteroData (PyG
`LinkNeighborLoader` style). But each node carries provenance `(seed_type_id,
seed_index)`, and the **disjoint dedup key** `LookupKey { seed_type_id, seed_index,
global_id }` keeps different roots' subtrees as **separate instances**. So the merged
output equals `(lhs-only sample) ⊎ (rhs-only sample)` with zero shared instances —
*provably equivalent to two independent samples*, assembled in one pass for efficiency.

> **Foundation: PR #305.** Before #305 the dedup key was `{seed_index, global_id}` and
> the anchor was looked up via `seed_nodes.keys().next()` (an arbitrary type). Because
> `seed_index` resets per seed type (lhs#0 and rhs#0 both = 0), a node reachable from
> both an lhs and an rhs root *merged*, entangling the two towers; and the anchor could
> be read from the wrong type's timestamp array (silently leaking once arrays differ in
> length). #305 added `seed_type_id` to `NodeRecord`/`RawEdge`/`LookupKey` and the anchor
> lookup. **Negative sampling depends on this**: negatives are extra rhs roots whose
> arrays make `rhs` longer than `lhs`, which is exactly the case #305 made correct.

### 3.3 The temporal anchor
Each seed carries a timestamp ("anchor") in `seed_timestamps[type][seed_index]`. When
expanding a node, the `TemporalMostRecent` sampler drops edges newer than that node's
anchor (resolved via the node's own `seed_type_id` — the #305 fix). Negatives must
replicate their positive's anchor so their neighborhoods are sampled at the same
causal time.

---

## 4. The idea in one sentence

> After the engine resolves the positive seeds, append `M` uniformly-drawn, temporally-
> valid global RHS ids **per positive** to `seed_nodes[rhs]` (anchor-major), replicate
> each positive's anchor into `seed_timestamps[rhs]`, and let the existing `sample()`
> machinery GNN-encode them as ordinary seeds — then expose `M`, `B`, and the rhs type
> name through the FFI so the kumo adapter can build `dst_pos_index`/`dst_neg_index`.

`sample()` itself is **unchanged**; negatives are pre-expanded into the request.

---

## 5. Design & data contracts

### 5.1 The config — `NegativeSamplingSpec`
`diskgraph-core/src/sampling.rs`:
```rust
pub struct NegativeSamplingSpec {
    pub rhs_node_type: String,  // RHS type to draw from. Caller passes it explicitly
                                //   (we do NOT infer which FK is the target).
    pub amount_m: usize,        // M: negatives per positive. Total negatives = B·M.
    pub rnd_seed: u64,          // Deterministic RNG seed (replay-safe; no thread_rng).
}
```
It is **not** on the wire `BatchSamplingRequest` — it is passed as an optional param to
`sample_runtime_batch`, because negatives are expanded *before* `sample()` runs. `None`
⇒ byte-identical to pre-feature behavior.

### 5.2 Injection point
In `sample_runtime_batch`, *after* `seed_nodes` and the per-type `seed_timestamps` map
are built and *before* the `BatchSamplingRequest` is assembled. `task.seed_node_ids` is
snapshotted from the positives **before** the append, so negatives never leak into the
task projection (only into the sampled subgraph).

### 5.3 Layout contract: `[B positives, then B·M negatives]`, anchor-major
`seed_nodes[rhs]` ends as `[v₀…v_{B-1}, n_{0,0}…n_{0,M-1}, n_{1,0}…, …]` — outer loop
over positive rows, inner over the `M` draws ("anchor-major"). `seed_timestamps[rhs]` is
extended in lockstep with each row's anchor replicated `M×`.

**Why this order, when the model is index-based?** Because the *adapter* (kumo, Task 2.x)
must build the index tensors, and anchor-major makes that trivial:
`dst_pos_index = arange(B)`, `dst_neg_index = arange(B, B+B·M).view(B, M)` (no transpose).
It also matches PyG's own hetero layout. **This is a documented FFI contract, not a model
requirement** — see §6 for why the model doesn't care.

**The seed-prefix invariant.** Seeds are inserted into their `type_states` first, at hop
0, in seed order, and distinct `seed_index` values mean negatives are never deduped
against positives (or against each other / lhs-reachable nodes). Therefore in the output,
`graphs["main"].node_ids[rhs][0:B]` are exactly the positives and `[B : B+B·M]` exactly
the negatives, anchor-major. This is pinned by a Rust test driving `finalize`
(`test_finalize_preserves_anchor_major_seed_prefix`) and by Python e2e.

### 5.4 FFI surface (what crosses to Python)
`TaskBag` (`diskgraph-engine/src/output.rs`) gains:
| field | type | meaning |
|---|---|---|
| `amount_m` | `u32` | `M` (0 when negative sampling is off) |
| `num_positives` | `u32` | `B`, the positive count (always set) |
| `rhs_seed_type` | `Option<String>` | the RHS type negatives were drawn from (`None` when off) |

The pybind dict (`diskgraph-pybind/src/runtime_batch_iterator.rs`) surfaces these under
`batch["task"]["amount_m" | "num_positives" | "rhs_seed_type"]`. The
`RuntimeBatchIterator` constructor accepts optional `neg_rhs_node_type` / `neg_amount_m`
/ `neg_rnd_seed`; when set it threads a `NegativeSamplingSpec` into `sample_runtime_batch`
(default `None` ⇒ unchanged). The Python wrapper `make_runtime_batch_iterator` forwards
the same kwargs.

Together with the §5.3 layout invariant, Python has everything to build the index tensors
from `node_ids[rhs_seed_type]` without any further engine support.

---

## 6. Static vs temporal: what the sampler must (and must NOT) do

This is the subtlety most likely to confuse a reader. The temporal/static split lives in
**two different layers**, and only one is the sampler's job.

- **Sharing is a *head* concern, NOT the sampler's.** Temporal LP (`share_negatives=False`)
  scores each anchor element-wise against *its own* `M` negatives → `neg_score [B, M]`.
  Static LP (`share_negatives=True`, `link_pred_head.py`) does `rhs_neg_x.view(-1, d)` then
  `lhs_x @ rhs_neg_x.t()` → `[B, B·M]`, so every anchor sees *all* negatives. **Both consume
  the same sampled negative set**; the difference is a tensor reshape + matmul on the model
  side. The engine emits the same `B·M` anchor-major negatives either way; the head decides
  element-wise vs shared. Our `[B, M]` layout supports both (`view(-1)` → `[B·M]` for static).
  **We therefore do NOT implement sharing, and we deliberately emit `B·M` (not a deduped
  pool of N)** — the static head's `view(-1)` assumes a full `B·M`.
- **The one task-type difference the sampler owns = temporal validity of the negative node.**
  Gated by `do_temporal = seed_timestamps.is_some() && (create_col || end_col present)` on
  the RHS type. Static-LP RHS with no time columns ⇒ the gate is off ⇒ the filter no-ops.
  (Caveat in §12: a static-LP RHS that *incidentally* carries a `time_col` would be
  filtered against the static anchor — documented, revisit if it bites.)

---

## 7. Code flow (end to end)

```
Python (kumo trainer, Task 2.x/3.x)
  make_runtime_batch_iterator(..., neg_rhs_node_type=, neg_amount_m=M, neg_rnd_seed=)
        │
        ▼
RuntimeBatchIterator (#[new], runtime_batch_iterator.rs)
  stores Option<NegativeSamplingSpec>
        │  next_batch()
        ▼
DiskGraphEngine::sample_runtime_batch(reader, …, negatives: Option<&NegativeSamplingSpec>)
  1. resolve seed table → seed_nodes {lhs:[B], rhs:[B]}, seed_timestamps (flat, per row)
  2. build per-type seed_timestamps map (clone flat ts into each type key)
  3. SNAPSHOT primary_ids = seed_nodes.values().next()   ← positives-only, for task.seed_node_ids
  4. if Some(spec):
        rhs_type_id ← node_type_to_id[spec.rhs_node_type]
        (lo, hi)    ← shard min_id..=max_id for that type        (dense id range)
        (c_idx,e_idx) ← reader.table(rhs).temporal_col_indices()  (resolve ONCE)
        do_temporal ← seed_timestamps.is_some() && (c_idx|e_idx)
        is_valid    ← closure → TableReader::is_node_temporally_valid (one reused scratch)
        append_global_negatives(seed_nodes, seed_timestamps, …, is_valid)   ← §7.1
  5. assemble BatchSamplingRequest (seed_nodes now includes negatives)
  6. sample(&request, …)        ← UNCHANGED; roots every seed (incl. negatives) at hop 0
  7. finalize() → RuntimeBatch  (node_ids[rhs] = [B pos, B·M neg] + neighbors)
  8. set task.{amount_m, num_positives, rhs_seed_type}; task.seed_node_ids = snapshot
        │
        ▼
runtime_batch_to_pydict → batch["task"][...], batch["graphs"]["main"]["node_ids"][rhs]
        │
        ▼
kumo adapter (Task 2.x): dst_pos_index=arange(B), dst_neg_index=arange(B,B+B·M).view(B,M)
two-tower (Task 3.x): element-wise (temporal) or view(-1)+matmul (static)
```

### 7.1 `append_global_negatives` (the pure core)
`diskgraph-engine/src/engine.rs`. Signature (temporal validity injected as a closure so
the function is pure and unit-testable without disk I/O):
```rust
pub(crate) fn append_global_negatives<F>(
    seed_nodes: &mut HashMap<String, Vec<u64>>,
    seed_ts: &mut Option<HashMap<String, Vec<i64>>>,
    rhs_node_type: &str, amount_m: usize, rnd_seed: u64,
    lo: u64, hi: u64, b: usize,
    mut is_valid: F,                    // (candidate_id, anchor_ts) -> Result<bool>
) -> Result<()> where F: FnMut(u64, i64) -> Result<bool>
```
Logic:
- `amount_m == 0` ⇒ no-op. Validates `seed_nodes[rhs].len() == b` and (if temporal)
  `seed_ts[rhs].len() == b`.
- For each row `i in 0..b` (anchor `a_i = seed_ts[rhs][i]`): clear the per-row validity
  memo (validity depends on the anchor, fixed within a row), then for `m in 0..amount_m`
  seed a per-`(i,m)` `ChaCha8Rng` (stable mix `rnd_seed + i + (m<<32)`), draw
  `cand = gen_range(lo..=hi)`. If temporal, accept `cand` if valid else resample up to
  `NEG_RESAMPLE_BUDGET = 5` more times; **on exhaustion the slot is MASKED** (`neg_mask=false`)
  rather than emitting a temporally-invalid negative — the last-drawn `cand` is kept as a **dense
  placeholder** id (still GNN-encoded, but the loss ignores it via the mask). Valid candidate (or
  non-temporal) ⇒ `neg_mask=true`. Push `cand` to a buffer; push `a_i` to a timestamp buffer; record
  the mask bit.
- Extend `seed_nodes[rhs]` and (if temporal) `seed_ts[rhs]` from the buffers, in lockstep; the
  parallel `neg_mask: Vec<bool>` (length `B·M`, anchor-major) is surfaced via `TaskBag.neg_mask` and
  the pybind dict (`batch["task"]["neg_mask"]`). The `[B,M]` layout stays **dense** (placeholders in
  masked slots), so `dst_neg_index` reshape is unaffected; the loss applies the mask.

### 7.2 Temporal validity (the live check)
The closure calls `TableReader::is_node_temporally_valid(node_id, observation_ts_secs,
create_idx, end_idx, &mut scratch)` (`diskgraph-index/src/graph.rs`). It decodes the
candidate's create/end time from that node's rowpack and returns `created ≤ anchor &&
(end is null || end > anchor)`; returns `true` immediately if both column indices are
`None`. Cost is one shard lookup + a partial rowpack decode per *distinct* candidate;
the per-row memo makes resampled repeats free, and a single `RowPackDecodeScratch` is
reused across all candidates (no per-call allocation). We deliberately do **not** prefetch
the whole RHS range (it can be millions of rows); for the bounded `B·M` draws lazy
per-candidate decode + memo is cheaper.

---

## 8. Worked example

`B = 1`, `M = 2`, lhs=`User`, rhs=`Item`. Positive pair `(User5, Item9)` at anchor `t=100`.
Item catalog ids `[0..50]`; items `45,46` have `create_time = 150` (created *after* `t`).

1. Seeds: `seed_nodes = {User:[5], Item:[9]}`, `seed_timestamps = {User:[100], Item:[100]}`.
2. Inject negatives for the one positive row (anchor `100`): draw `M=2` uniform ids in
   `[0,50]`. Suppose draws are `46` then `12`. `46` fails temporal validity
   (`create_time 150 > 100`) → resample → `12` (valid). Second draw `12`… say `30` (valid).
   → negatives `[12, 30]`.
3. After injection: `seed_nodes[Item] = [9, 12, 30]` (`[pos, neg, neg]`),
   `seed_timestamps[Item] = [100, 100, 100]` (anchor replicated).
4. `sample()` roots all three Item seeds (seed_index 0,1,2) + the User seed. Each Item's
   neighborhood is sampled at anchor `100` (so no future edges leak). Disjoint keys keep
   them separate instances even if, say, `Item12` is also reachable from `User5`.
5. Output `node_ids[Item] = [9, 12, 30, <neighbors…>]`; `task.amount_m=2`,
   `num_positives=1`, `rhs_seed_type="Item"`, `seed_node_ids=[5]` (positives only).
6. Adapter: `dst_pos_index=[0]`, `dst_neg_index=[[1,2]]`. Two-tower scores
   `User5·Item9` (pos) vs `User5·Item12`, `User5·Item30` (neg).

---

## 9. Performance

- **Allocation discipline.** One reused `RowPackDecodeScratch`; negatives accumulated into
  pre-`reserve`d buffers then a single `extend_from_slice`. The per-row validity memo
  avoids re-decoding repeats during resampling.
- **Subgraph growth.** `B·M` extra rhs neighborhoods per batch (bounded by `M`, usually 1).
  This is the intended cost of GNN-encoding negatives in the two-tower.
- **Temporal check.** O(distinct candidates) partial rowpack decodes, not O(RHS). No full
  RHS prefetch.
- **`sample()` untouched** ⇒ no regression on the hot traversal path for non-LP / no-neg
  batches (`None` is a zero-cost early-out; `seed_node_ids` snapshot is one clone that was
  already happening).
- **Determinism.** `ChaCha8` seeded per `(rnd_seed, row, m)` ⇒ replay-safe and parallel-safe
  (no `thread_rng`).

---

## 10. Testing

- **Rust unit (helper, pure):** `append_global_negatives` — length `B+B·M`, positives
  preserved in `[0:B]`, negatives in range, timestamps replicated anchor-major,
  determinism, `M=0`/`None` no-op, temporal rejection via stubbed closure, bounded-resample
  fallback, `B`-mismatch error.
- **Rust output invariant:** `test_finalize_preserves_anchor_major_seed_prefix` — `finalize`
  preserves `node_ids[rhs] = [B pos, B·M neg]` ordering.
- **Python e2e (real built wheel):**
  (a) negatives present, non-temporal — asserts `amount_m`/`num_positives`/`rhs_seed_type`,
  the rhs prefix split, and that `seed_node_ids` stays length `B`;
  (b) negatives absent — regression against the existing contract (`amount_m==0`, etc.);
  (c) **temporal RHS** — items created after the anchor; asserts drawn negatives are all
  temporally valid. **(c) is the test that exercises the live
  `is_node_temporally_valid` decode path** the helper's stubbed tests cannot.
- **Backward-compat:** all existing engine/core/pybind/Python tests pass; the 6 ingest
  integration callers pass `None`.

---

## 11. Boundaries — what this PR does NOT do

- **No index tensors / HeteroData assembly.** Building `dst_pos_index`/`dst_neg_index` is the
  kumo adapter's job (Task 2.x). The engine stops at emitting the ordered rhs seeds + `M`/`B`.
- **No model changes.** Element-wise vs shared (static) scoring lives in the two-tower head
  (Task 3.x) and already exists.
- **No true-negative (false-negative) filtering — DEFERRED.** A sampled negative may coincidentally
  be a true link of the LHS; this PR does not filter it (PyG does none either). True-negative
  filtering via an `lhs→rhs` union positives index is deferred. This PR's correctness
  scope is **temporal validity (via masking)**, not non-link filtering.
- **Shallow / hybrid models** get global negatives from their embedding table and do not use
  this path.

---

## 12. Decisions & rationale (with caveats)

1. **Spec as a `sample_runtime_batch` param, not on `BatchSamplingRequest`.** Negatives are
   expanded before `sample()`, so the wire request and `sample()` stay untouched; smaller blast
   radius, no serialization churn.
2. **Caller passes `rhs_node_type` explicitly** — the engine does not guess which FK is the
   target. Avoids a heuristic that would silently break on multi-FK seeds.
3. **Anchor-major `[B pos, B·M neg]` layout** chosen to make the adapter trivial and match PyG;
   justified because model consumption is index-based (order is free to define) — see §6.
4. **Emit `B·M`, never a deduped shared pool** — required so the static head's `view(-1)`
   reshape lines up (§6).
5. **Temporal gating derived from metadata + timestamp presence**, not an explicit flag.
   *Caveat:* a static-LP RHS that incidentally carries a `time_col` would be filtered against
   the static anchor. Verified that gating is correct for genuinely-timeless RHS; if a static
   task ships a temporal RHS and this matters, add an explicit `temporal: bool` to the spec
   (it is OK to revisit this with new information).
6. **Bounded resample (K=5) then MASK** — on exhaustion the slot is masked (`neg_mask=false`,
   dense placeholder kept) so a temporally-invalid negative is **never emitted**; the loss ignores
   masked slots. (Earlier this "accepted the last draw," which could admit an invalid negative —
   replaced by masking for temporal correctness.) Masking the loss (vs a ragged tensor) keeps `[B,M]`
   dense → no reshape break; masked placeholders get zero gradient → leak-safe, only wasted FLOPs.
7. **Deterministic ChaCha8 seeded per `(rnd_seed, row, m)`** — replay-safe and worker-fork-safe.
8. **`seed_node_ids` stays positives-only** (snapshot before append) — preserves the existing
   single-type contract; the LP-specific meaning of `seed_node_ids` is finalized in Task 2.x.

---

## 13. Pointers
- Config: [`diskgraph-core/src/sampling.rs`](../diskgraph-core/src/sampling.rs) (`NegativeSamplingSpec`).
- Core logic: [`diskgraph-engine/src/engine.rs`](../diskgraph-engine/src/engine.rs) (`append_global_negatives`, the injection in
  `sample_runtime_batch`).
- Temporal check: [`diskgraph-index/src/graph.rs`](../diskgraph-index/src/graph.rs) (`is_node_temporally_valid`,
  `temporal_col_indices`).
- FFI: [`diskgraph-engine/src/output.rs`](../diskgraph-engine/src/output.rs) (`TaskBag`),
  [native iterator](../diskgraph-pybind/src/runtime_batch_iterator.rs),
  [Python `RuntimeIndex`](../diskgraph-pybind/python/diskgraph/runtime_batch_iterator.py).
- Kumo integration: [runtime adapter](https://github.com/kumo-ai/kumo/blob/master/packages/kumo-core/src/kumo_core/diskgraph/runtime_adapter.py),
  [two-tower module](https://github.com/kumo-ai/kumo/blob/master/packages/kumo-core/src/kumo_core/train/link_gnn_module.py),
  [prediction head](https://github.com/kumo-ai/kumo/blob/master/packages/kumo-core/src/kumo_core/modeling/heads/link_prediction.py).
- Tests: [Rust](../diskgraph-engine/src/tests_unit.rs),
  [Python](../diskgraph-pybind/python/tests/test_runtime_batch_iterator_e2e.py).
- Foundation: [DiskGraph PR #305](https://github.com/kumo-ai/kumo-diskgraph/pull/305)
  (per-type anchor + `seed_type_id` in `LookupKey`).
- Background: [TT-NG architecture and migration](tt-ng-architecture-and-migration.md).
