# Experimental relevant-walk attribution

Engine provides an experimental reference implementation inspired by [GNN-LRP](https://arxiv.org/pdf/2006.03589).
It assigns signed relevance to layer-indexed routes through a captured computation.
It is not a causal explanation, an uncertainty estimate, or a replacement for numerical derivatives.

## Supported reference profile

The current profile requires evaluation-mode HeteroGNN, ReLU or identity activation, no normalization, sum or mean aggregation, and NodePredHead.
It handles linear prelayers, learned or identity skips, and biases through an explicit accounting ledger.
The Engine prediction and reconstruction adapters retain their CPU scalar MSE, one-worker, one-seed restrictions.
Other task heads, GPU, multi-seed batches and concurrent scorers are not qualified by this implementation.
The retained supply-chain model uses GELU, LayerNorm and multiple aggregation operators and is unsupported for this method.

```python
from kumo.engine import EvidenceConfig, WalkAttributionConfig

config = WalkAttributionConfig(gamma=0.0, top_k=20, max_walks=10000)
result = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    batch_size=1,
    num_workers=1,
    evidence=EvidenceConfig(walk_attribution=config, on_unsupported="record"),
)
```

Use the existing prediction-evidence retention API to retain the result before cleaning up runtime directories.
On a loaded evidence bundle, call `bundle.attribute_walks(model=restored_trainer, config=config, output=new_directory)` to produce a separate immutable derived bundle.
The loaded Trainer must be trusted and match the retained fingerprint.
Offline reconstruction checks the emitted prediction before attributing it and never resamples the graph.

Original activation results carry `original_forward_attribution`; offline results carry `retained_graph_recomputation`.
Failed attribution does not erase completed predictions.
Unknown operations remain unsupported instead of being silently skipped.

## Interpretation

The selected scalar is an encoded model output; attribution ends at encoded node features.
These scores do not claim raw-source column contributions or decoded decision-unit contributions.
Rounding, flooring and nonlinear target decoding are not included.
The gamma rule enhances positive local source contributions, including signed-input behavior explicitly implemented by the reference backend.
The default gamma is zero; nonzero gamma changes the rule and is retained with the explanation.
This is a reference rule profile, not a claim that every Engine model satisfies the paper's exact Taylor assumptions.

The ledger records displayed walk relevance, bias relevance, stabilization residual and omitted signed relevance.
Signed cancellation means omitted signed relevance is not a bound on omitted absolute relevance.
Top-k truncation and route-budget exhaustion are explicit; returned walks are not renormalized to imply full coverage.
The route budget counts expansion work rather than only completed walks.
Self routes are computational operations and are never fabricated database relations.
Row/edge incidence summaries of overlapping walks are not independent additive contributions.

## Qualification and extension gates

The current implementation is the reference and integration stage of the broader feasibility plan.
An interface or a passing schema check does not qualify a new method, device or task family.

| Extension | Required work before enabling |
| --- | --- |
| GELU, LayerNorm, min/max/std | Derive explicit rules; test degenerate cases and faithfulness independently. |
| Forecasting, classification, link heads | Generalize frozen scorer reconstruction and selected-output identity; validate each head. |
| GraphTransformer | Distinguish attention, positional and database routes; qualify attention and normalization rules. |
| GPU, multiple seeds/workers | Preserve actual batch context, qualify numerical parity and prove bounded publication. |
| Faster search | Match exhaustive oracle; identify node-level approximations and report omitted work. |
| Raw source/decoded attribution | Specify encoder/decoder rules and their reference semantics. |

[Efficient subgraph aggregation](https://proceedings.mlr.press/v162/xiong22a.html) and [relevant-walk search](https://proceedings.mlr.press/v202/xiong23b.html) are separate scaling candidates.
[AttnLRP](https://proceedings.mlr.press/v235/achtibat24a.html) is a candidate for attention extensions, not evidence that Engine's graph transformer is supported.

## Validation and release

Methods tests use hand-calculated scalar graphs, multilayer path multiplicity, bias accounting, negative outputs and genuine-gradient preservation.
Integration tests check source identities, unsupported operators, corruption, duplicate IDs, ledger validity and failure isolation.
The native gate trains a fresh small fixture, saves/reloads the model, and compares disabled/enabled predictions and original/offline attribution.
Finite differences validate ordinary derivatives; they are not an oracle for LRP's relevance redistribution.

See [installed-wheel validation](evidence-validation.md) for the committed-source build and rollback process.
The new standalone candidate installer and probes under `scripts/xai` avoid the stress harness's incompatible fat-wheel installer.
Stress qualification must use manifest-bound probe bytes and dedicated admitted hosts; no cloud workload is launched merely by adding the probes.
Repeated-batch performance, GPU qualification and empirical faithfulness reports are distinct gates and must not be inferred from unit-test results.

The prototype wheel, environment, active run, evidence and inspected holdout remain unchanged.
