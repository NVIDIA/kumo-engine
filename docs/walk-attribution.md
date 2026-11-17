# Experimental relevant-walk attribution

Engine provides an experimental implementation of walk attribution inspired by [GNN-LRP](https://arxiv.org/pdf/2006.03589) (Schnake et al.).
For one prediction it assigns signed relevance to ordered message-passing walks of the captured scoring forward, from a source record's encoded features to the explained output.
Three methods are registered: the legacy contribution-gamma LRP rule, a paper-style weight-gamma GNN-LRP rule, and the degree-T GNN-GI decomposition of bias-free models.
Scalar regression and binary node classification are supported.
It is not a causal explanation, an uncertainty estimate, or a replacement for numerical derivatives, and relevance is never a probability contribution or the effect of removing a record.

## Supported profile

The explained module must be an evaluation-mode `HeteroGNN` with a `NodePredHead`.
Activations must be ReLU or identity, with no normalization, one sum or mean aggregation, registered relation and root operators, no prediction-time encodings, no distribution head and no `torch.compile` wrapper.
Linear pre-message-passing layers, learned or identity skips, and eval-mode dropout are handled.
Supported tasks are scalar regression with MSE and binary node classification with binary cross-entropy and a categorical target with exactly two labels.
Multi-class, multi-label, link and forecasting tasks are refused.
Capture runs on CPU with one scorer and one seed per batch (`batch_size=1`, `num_workers=1`); GPU, several seeds or workers, and concurrent scorers are not qualified.
Models with GELU, LayerNorm or several aggregation operators, such as the retained supply-chain prototype model, are refused.

Model-level checks run before any hook is registered, and graph facts of one row, such as a primary-key gather with several incoming edges, make that row `unsupported`.
With `on_unsupported="raise"` an unsupported model raises `UnsupportedWalkAttribution`; with `"record"` the row is `unsupported` with a reason and predictions are unchanged.
Unknown operations are refused instead of being skipped.

## Methods

| Method ID | Rule at each linear (input `h`, weight `w`, frozen mean denominator `d`) | Models | Decomposition | Bags |
| --- | --- | --- | --- | --- |
| `hetero_lrp_gamma_reference_v1` | Contribution gamma: `z = h w / d`, then `z + gamma max(0, z)` | Profile above, with or without biases | `conservative_lrp` | No |
| `hetero_gnn_lrp_weight_gamma_v1` | Weight gamma: `h (w + gamma max(0, w)) / d` (Table I, Eq. 9, with the Engine definitions below) | Profile above, with or without biases | `conservative_lrp` | No |
| `hetero_gnn_gi_degree_t_v1` | Nested Gradient x Input over layer gates (Proposition 2) on the captured linear piece | Profile above with no pre-message-passing, message-passing or head bias | `exact_degree_t` when claimed, else `piecewise_gnn_gi` | Yes |

Both LRP rules treat a bias as the weight of a constant input 1 (`b + gamma max(0, b)`), which absorbs relevance reported by kind, and stabilize each denominator with an absolute epsilon added with the sign of the total (`+epsilon` when the total is 0).
v2 rows of every method are computed in float64 from captured tensors with the exact epsilon; v1 rows keep their original arithmetic.
An unknown method ID raises `ValueError` and never falls back to another rule.

## Configuration

```python
from kumo.engine import WalkAttributionConfig

WalkAttributionConfig(
    gamma=0.0,                # LRP gamma; GNN-GI requires 0
    epsilon=1e-9,             # LRP stabilizer; GNN-GI requires the default
    max_walks=10000,          # budget of message-passing route expansions
    top_k=20,                 # displayed walks, by absolute relevance
    output_index=0,
    method='hetero_lrp_gamma_reference_v1',
    feature_vectors=False,    # per-walk relevance over the encoded source features
    bags=False,               # GNN-GI only
    selected_class=None,      # binary only; None explains the encoded positive label
)
```

`feature_vectors` and `bags` must be `bool`, and `selected_class` must be `None` or a `bool`, `int` or `str` whose type matches the label type exactly, so `1` never selects `True`.
A default configuration serializes exactly as before through `to_dict()`, and `WalkAttributionConfig(**config.to_dict())` reproduces it.
`inspect_walk_capability(module)` returns the unchanged legacy record; `inspect_walk_capability(module, config=config)` reports the method's qualification codes and the architecture read from actual parameters, without registering hooks.

## Usage

```python
from kumo.engine import (EvidenceConfig, PredictionEvidenceBundle, WalkAttributionBundle,
                         WalkAttributionConfig)

config = WalkAttributionConfig(method='hetero_gnn_lrp_weight_gamma_v1', feature_vectors=True,
                               top_k=64, selected_class=True)
result = trainer.predict(
    prediction_table,
    snapshot=snapshot,
    batch_size=1,
    num_workers=1,
    evidence=EvidenceConfig(walk_attribution=config, numerical_sensitivity=None,
                            on_unsupported='record'),
)
result.evidence.export(retained_directory)

bundle = PredictionEvidenceBundle.load(retained_directory)
bundle.attribute_walks(model=trusted_trainer, config=other_config, output=new_directory)
derived = WalkAttributionBundle.load(new_directory)
```

Native rows are written into each prediction's evidence record during the original forward (origin `original_forward_attribution`).
`attribute_walks` derives any registered method from the retained graphs into a separate immutable bundle (origin `retained_graph_recomputation`), so several methods can explain one capture.
The loaded trainer must be trusted and must match the retained fingerprint; replay checks the emitted prediction before attributing it and never resamples the graph.
A method that does not qualify for the captured model yields `unsupported` rows, and failed attribution never erases completed predictions.

## Binary classification

Binary evidence requires a `BinaryCrossEntropyLoss` model, a categorical target encoder with exactly two labels, one head output, and `EvidenceConfig(numerical_sensitivity=None)`, because numerical sensitivity differentiates a decoded regression output and remains regression-only.
The explained scalar is the selected-class logit `s * z`, where `z` is the raw head output before calibration and `s` is `+1` when the selected label is the one the fitted target encoder maps to class 1, otherwise `-1`.
Orientation is verified at run time from the fitted encoder and the loss, never inferred from output column names.
Every method realizes `s` by folding it into the selected output row of the last head linear before its rule applies; for LRP with `gamma > 0` this differs from negating an explanation of `z`.
Each row records the label mapping, the target encoder's reverse flag (for audit only), the calibration (`platt` with its raw and effective weight and bias, or `none`), the model probability `sigmoid(s * z)`, and the emitted probability of the selected class, which equals the emitted `True_PROB` or `False_PROB` bit for bit.
Relevance is in logit units, and the calibrator is outside every explained function.
Engine's record removal (`PredictionEvidenceBundle.perturb`) remains regression-only: on a binary bundle it fails inside the removal step after creating a staging tree, so binary interventions must be implemented outside Engine.

## Bias options

`Trainer(model_plan, accelerator='cpu', bias_options=BiasOptions(gnn=False, head=False))` trains without additive biases in the `HeteroGNN` linears (pre-message-passing linears, relation convolutions and learned root skips; `gnn`) or in the `NodePredHead` linears (`head`).
Encoder and normalization parameters are never affected.
These switches are Engine-local and are not part of the public `ModelPlan`; promoting them to plan fields is a recorded follow-up.
`None` and `BiasOptions()` build exactly the default modules, and older bundles without the flag load with biases and predict identically.
Non-default options are recorded in the bundle's `training-spec.json`, the trained flags in `model/config.json`, and `Trainer.load` rejects a bundle whose two records disagree.
A warm start with `bias_options=None` inherits the source checkpoint's flags, read from its `model.` and `head.` parameters; an explicit option that differs raises `ValueError` before training.
Switches the selected model or head cannot honor, such as `gnn=False` with a graph transformer or `head=False` with another head, raise instead of being ignored.
A bias-free bundle loaded by an Engine without this option fails at strict state-dict loading.

## Named runs

`Trainer(model_plan, ..., run_name='churn-v1')` replaces the random run ids; the name must match `[a-z0-9][a-z0-9-]{0,47}`.
The trainer's scratch directory becomes `trainers/<run_name>`, the k-th `fit` (from 0) runs as `<run_name>-fit-<k>`, and its train-table extension is `tt-<run_name>-fit-<k>` in the snapshot index.
`fit` raises `FileExistsError` before any work when that run directory or extension already exists, so each named run needs a fresh scratch directory (`kumo.engine.configure(scratch_dir=...)`).
The check precedes the creation of the run directory, so two processes starting the same name in one scratch at the same moment are not excluded.
Prediction run ids stay random for named trainers, so prediction ids never repeat across predict calls, and bundles do not record the name.

Tested: with a run name, caller-supplied training rows (`TrainingTable.from_source_table`), a fresh scratch directory and `torch.use_deterministic_algorithms(True)` around `fit`, fits in separate fresh processes produced bitwise-identical model states (every parameter and buffer, including the calibrator) on the development host and data-loading layout.
Under the same conditions, with the base run id fixed to the same name, default training on this branch matched the base `1909c5b9` wheel bit for bit.
Inferred, not tested inside the native code: earlier run-to-run differences came from random extension names entering DiskGraph's training shuffle and from nondeterministic CPU scatter-add gradients.
`PredictiveQuery.generate_training_table` can return its rows in a different order on every call, which changes the trained model even with a run name, so reproducible training needs caller-supplied rows.
The binary walk gate generates its table this way and trains without a run name, so every gate run trains a different model.
Reproducibility across hosts, data-loading layouts, Python or Torch versions has not been tested.

## Evidence schemas

A result is `walk_attribution/v1` only for the legacy method on a regression task with the five original settings (`gamma`, `epsilon`, `max_walks`, `top_k`, `output_index`); every other result is `walk_attribution/v2`.
On fixed inputs, v1 rows, derived walk-bundle manifests and legacy-shaped native records keep their exact bytes; on new bundles v1 values can change because replay now applies the recorded precision (see Replay precision).
A v2 row lists the displayed walks with exactly `T` steps in forward order, indexed by `mp_layer`: relation steps carry the relation, its direction (`fk_to_pk` or `pk_to_fk`), the operator (`mean`, `sum` or `gather`), the frozen denominator and the retained edge occurrence, and self-update steps carry the destination occurrence.
It also lists every source occurrence with totals over all its enumerated walks, the per-walk and per-source encoded-feature vectors when `feature_vectors=True`, bags when requested, the ledger, coverage, exactness, numerics, diagnostics and the output block.
Head and pre-message-passing operations are recorded as operations, never as walk steps.
Statuses are `available`, `incomplete`, `unsupported` and `failed`; failed rows name a reason such as `accounting_error_exceeded`, `nonfinite_relevance` or `replay_precision_unavailable`.
Every v2 number is finite JSON; context numbers of failed or unsupported rows that are not finite, such as an infinite logit, are written as null.
Coverage is `complete`, `complete_top_k` (every route expanded, only `top_k` walks displayed) or `budget_exhausted` (the only incomplete state); an exhaustive search is never reported as truncated.
The budget counts message-passing route expansions, and a destination's incoming routes are expanded all or none.
Walks are ranked by absolute relevance with ties broken by walk key, but only among exactly equal floats, so walks equal in exact arithmetic can be ordered by rounding.
When the retained sample records its edges, the v2 validator also binds every displayed relation step to an edge of its layer, every mean denominator to the retained in-degree and every primary-key gather to one incoming edge, and it binds the explained class to the row's own `selected_class`.
Legacy-method v2 rows written before commit `5a48f774` do not validate, because that rule now requires the head-inversion record; bundles containing them are refused and must be recaptured, while bundles with only v1 rows are unaffected.

## Accounting and tolerance

Every v2 row satisfies `score = displayed walks + omitted displayed walks + omitted search mass + bias relevance + stabilizer residual + precision residual + accounting error`.
A row with an available or incomplete status must satisfy `abs(accounting_error) <= tol(A) = 1e-9 * max(1, A)`, or it is emitted as `failed` (`accounting_error_exceeded`).
`A` adds the absolute ledger items, the absolute mass of every complete walk's encoded-feature vector (`ledger.walk_vector_abs_mass`) and, for the two LRP rules, the absolute mass of every intermediate relevance vector and of every split's shares (`ledger.intermediate_state_abs_mass`), so the tolerance of every row can be recomputed.
Float64 rounding scales with these magnitudes, which can far exceed the score when relevance cancels near a zero denominator.
Including them kept the reviewed cancellation cases from failing as `accounting_error_exceeded`, but it is a tolerance, not a proof that every correct row closes.
The validator bounds the vector component from the displayed vectors, but the intermediate-state component is the rule's own report and cannot be checked from the row; an inflated value widens the tolerance (in a review mutation, a recorded state mass of 1e6 hid a 1e-4 mismatch).
The check loses power under heavy cancellation: on reviewed real rows, a dropped share and a leak of 1e-6 of a share were always caught, a leak of 1e-7 on 86 of 96 rows and a leak of 1e-8 never, which is below the float32 capture noise of the walk values themselves.
Read ledger closure as weak evidence when `min_denominator_margin` is small or `intermediate_state_abs_mass` is large relative to `abs(score)`.
Rows written before commit `17c0ef84` lack both components and keep the smaller mass, which only makes their check stricter.
Omitted search mass is the signed relevance that reached unexpanded route prefixes; for LRP it includes bias and stabilizer terms that would have been absorbed below, and for GNN-GI the sum of absolute prefix masses is only a lower bound on the omitted absolute walk relevance.
Signed omissions can cancel, so a zero or small net omission is not evidence that omitted walks are irrelevant.

## Weight-gamma GNN-LRP

`hetero_gnn_lrp_weight_gamma_v1` redistributes the selected score backward through the captured forward, one linear map at a time.
At each destination neuron an input `h_j` of a route with weight `w` and frozen mean denominator `d` receives the share `h_j (w + gamma max(0, w)) / d` of the neuron's relevance, divided by the stabilized total of all such terms and the gamma-modified biases.
The same rule serves relation messages, self updates (an identity skip is the weight `I`, so its share is `(1 + gamma) h`), pre-message-passing linears and head linears.
The paper states the rule for a GCN with one weight matrix per layer, a normalized adjacency and no bias or stabilizer; relation-specific weights, biases as constant-input weights, the absolute epsilon, the identity skip as `I`, the rule at pre-message-passing and head linears, and sign folding are Engine and contract definitions.
`precision_residual` is 0 by construction because the method redistributes the emitted score itself, but walk values still follow the captured float32 activations.
The stabilizer residual carries the epsilon effect and the float64 rounding of the shares; epsilon bounds each factor by the relevance over epsilon, yet individual walks can still be large with opposite signs.
Rows carry `diagnostics['hetero_gnn_lrp_weight_gamma_v1']` with `head_output_denominator`, `head_output_flipped`, `relevant_neurons`, `denominator_sign_flips` (also by kind), `min_denominator_margin` (`abs(S) / sum abs(terms)`, small values meaning large opposing shares) and `max_preactivation_mismatch` (captured against float64 recomputed pre-activations, which shows the capture precision regime); they describe numerical conditions and are not quality scores.

The legacy rule applies gamma to contributions and this rule applies it to weights.
They are mathematically identical whenever every input of every step is nonnegative and numerically equal up to float64 rounding, often bitwise.
Every Engine ReLU model feeds nonnegative inputs to every rule step, because the encoder output, the message-passing outputs and hidden head layers are post-ReLU, so comparing the two rules on such models is a null comparison.
They differ on signed inputs, which occur with identity activations in directly built modules.
The paper's model has `h >= 0` throughout; applied literally to signed inputs its Appendix C step gives the contribution form, while the Table I weight form assumes `h >= 0`; neither has been validated on signed inputs, so their difference there is a definitional one, not evidence that one rule is right.

Neither LRP rule is the degree-T decomposition, and LRP rows never claim it.
Their walk relevances are not `T`-th order derivatives with respect to graph gates and need not satisfy the homogeneity identity.
On a bias-free model at `gamma = 0` they approximate GNN-GI walk scores only up to the stabilizer, with a relative deviation of order `epsilon / abs(pre-activation)` per layer that grows without bound near zero pre-activations (reviewer fixtures: 2.8e-8 relative at `T = 2`, 6.7e-7 at `T = 3`); with biases, part of the score is attributed to biases instead of walks.

## Head sign inversion

Legacy and weight-gamma v2 rows report `head_output_denominator`, the folded, gamma-modified total `S` of the selected output before epsilon, and `head_output_flipped`, which is true when the selected score is nonzero and `S` has the other sign (`S = 0` counts as positive).
A flipped row has every share of its output step inverted, so inputs that raised the logit receive negative relevance, and it should not be read directionally.
With `gamma > 0` this can happen for negative selected logits; on the small binary models of the test gate it happened for 0 to 2 negative selected logits per run at `gamma = 0.25`, a count that varies because the gate retrains its models.
At `gamma = 0` the flag can fire only when the captured logit lies within the capture's rounding of zero, because `S` is recomputed in float64 from the captured head inputs.
The flag is truthful whenever it fires; the validator checks its consistency with `S` and the score but does not recompute `S`.

## Degree-T GNN-GI decomposition

`hetero_gnn_gi_degree_t_v1` computes the nested Gradient x Input walk relevances of Schnake et al. (Proposition 2) for one prediction of a qualified bias-free model on the captured scoring forward.
The explained value is the selected-class logit `s * z` before calibration for binary tasks, or the encoded output for regression.
The encoded node features (the encoder output, including encoder biases, categorical embeddings and relative-time encodings) and any pre-message-passing outputs are held fixed.
The variables are gates: one per message-passing layer, relation and retained edge occurrence, and one per layer and destination occurrence for the self-update branch.
Mean denominators stay frozen at the sampled in-degrees, the model is the point where every gate equals 1, head layers carry no gates, and `T` equals the number of message-passing layers.

A walk is an ordered sequence of `T` gates from a source occurrence to the seed, and its relevance is the `T`-th mixed partial derivative of the explained value with respect to its gates at the model point, on the linear piece fixed by the captured ReLU masks.
It is computed in float64 from captured tensors as a product of step matrices (relation weight divided by the frozen denominator, root weight or identity) and ReLU masks, ending at the source's encoded features; captured intermediate activations are used only for the masks.
`input_relevance` splits a walk's relevance over the source's encoded channels.
With `bags=True`, walks are grouped by the multiset of their layer-free variables, and a bag's relevance, the sum of its walks, equals its tied-gate derivative divided by `alpha!`.
Variables are per retained edge occurrence, which is finer than the paper's node-pair entries when two sampled occurrences join the same pair of records.
With two layers almost every bag holds a single walk, and bags are mainly informative at `T = 3`.

The method refuses, before any hook, every model outside the supported profile and every model with a bias parameter in a pre-message-passing, message-passing or head linear, reading actual parameters: the modules may hold only the weights the decomposition uses, and convolution bias flags must agree with their parameters.
A model trained with Engine's default biases is therefore refused (`gnn_bias_present`); explain it with GNN-LRP instead.
The claim `exact_degree_t` additionally requires gathers with at most one incoming edge, complete enumeration, no relevant ReLU at an exactly zero captured pre-activation with a nonzero input (a boundary neuron), agreement between the captured masks and the float64 recomputation, and `abs(precision_residual) <= 1e-5 * max(1, abs(score))` under `highest` precision or `<= 1e-2 * max(1, abs(score))` otherwise.
Then the walk relevances are the unique degree-`T` coefficients of the float64 function of the captured piece, they sum to its value `f_piece`, and `f(q Lambda) = q^T f(Lambda)` holds for every `q > 0`.

The claim is an algebraic identity about the captured piece of the qualified model under this definition of graph variables, and nothing more.
The emitted float32 score differs from `f_piece` by `precision_residual`, which is always reported beside the claim.
Under `medium` precision with bf16 matmul internals a claimed row can carry a residual up to that bound (up to about 9.8e-3 was measured on synthetic rows with 16 neighbors), which is comparable to individual walk relevances, so the claim is about `f_piece`, not about the emitted logit.
The claim does not extend to raw columns or to models with biases.
A walk relevance is not the effect of deleting a record or an edge: gating keeps denominators frozen, while record removal removes every occurrence and recomputes means (in the reference fixture, gating one edge at both layers gives 4.5 and deleting its transaction gives 8, from 9).

When the claim is withheld the row stays a conservative decomposition of `f_piece` (`piecewise_gnn_gi`) and lists its reasons in this order: `enumeration_incomplete`, `boundary_piece`, `mask_disagreement`, `precision_residual_exceeded`.
`piece: boundary` means not certified interior; `boundary_neurons` and `mask_disagreements` say why.
The boundary rule is conservative: an exactly zero relevant pre-activation with a nonzero input counts as a boundary even when the weights make the zero structural.
There is no near-zero margin threshold; the diagnostics report `f_piece`, the claim tolerance and its precision mode, and per-layer margins (`min abs(a~)`) so that small margins stay visible.
On synthetic rows shaped like a two-layer, eight-neighbor, 32-channel model, every linear saw at most eight rows and `medium` kept every probed row claimable (residuals at most 1.6e-7); with 12 neighbors bf16 internals gave residuals up to 2.1e-3 and mask disagreements in 10 of 24 rows, which correctly withheld the claim.
On the binary gate fixture (about ten events per user, retrained on every run) between 15 and 23 of 24 rows were claimed in four runs; these counts are not a stable claim rate.

## Replay precision

Engine sets the float32 matmul precision to `medium` process-wide whenever a trainer creates its Lightning trainer for fitting, validation, testing or prediction.
On hosts with bf16 matmul support `medium` uses bf16 internals above a size threshold: on the development host (Intel Xeon with AMX), 32-to-32 linears matched `highest` up to 8 rows and differed by about 2.3e-3 to 3.4e-3 from 9 rows on; the threshold depends on host and shape.
Native evidence manifests record `numerics` (the precision mode, the derived oneDNN fp32 setting, thread count, CPU capability and Torch version).
Offline replay applies the recorded mode, requires the derived oneDNN setting to match the record (`none` is read as `ieee`), fails a row with `replay_precision_unavailable` otherwise, and restores the process settings afterwards, also after exceptions.
CPU capability and thread count are recorded, not enforced; thread counts 1, 2 and 4 did not change `medium` results on the one probed shape.
Bundles produced before this record replay at the process precision as before and may fail in a fresh process.
Batch size also changes results under `medium` (by up to 1.0e-3 in the logit at module level as measured during design), so evidence runs use batch size 1, and predictive metrics should come from the same batch-size-1 run or report the measured difference.
On a 30-day churn example with two layers of eight neighbors at batch size 1, claimed degree-T rows carried precision residuals of at most 1.54e-7, the float32 rounding level.

## Interpretation

The explained scalar is the encoded model output for regression and the selected-class logit for binary classification, and attribution ends at the encoded node features.
Relevance is in those units: it is not a probability contribution, not a deletion or intervention effect, and not a cause.
Rounding, flooring and nonlinear target decoding are not included, and no raw-source column or decoded decision-unit contributions are claimed.
Any column-level breakdown is a separate encoder attribution step outside Engine and never an exact raw-feature degree-T expansion.
A record usually lies on several walks, and removing it also changes mean denominators, so row and edge incidence summaries of overlapping walks are not independent additive contributions.
Self routes are computational operations and never fabricated database relations.
Exactness, conservation and ledger closure are accounting properties; they are not evidence of faithfulness, causality or predictive quality, which need separate experiments with their own out-of-distribution limits.

## Qualification and extension gates

An interface or a passing schema check does not qualify a new method, device or task family.

| Extension | Required work before enabling |
| --- | --- |
| GELU, LayerNorm, min/max/std | Derive explicit rules; test degenerate cases and faithfulness independently. |
| Multi-class, multi-label, link, forecasting and distribution heads | Generalize frozen scorer reconstruction and selected-output identity; validate each head. |
| GraphTransformer | Distinguish attention, positional and database routes; qualify attention and normalization rules. |
| GPU, multiple seeds/workers | Preserve actual batch context, qualify numerical parity and prove bounded publication. |
| Faster search | Match the exhaustive oracle; identify node-level approximations and report omitted work. |
| Raw-source or decoded attribution | Specify encoder and decoder rules and their reference semantics inside Engine. |
| Public bias plan fields | Change the shared plan schema and regenerate its generated artifacts. |
| Binary record removal | Define structural removal with probability outputs for classification bundles. |

[Efficient subgraph aggregation](https://proceedings.mlr.press/v162/xiong22a.html) and [relevant-walk search](https://proceedings.mlr.press/v202/xiong23b.html) are separate scaling candidates.
[AttnLRP](https://proceedings.mlr.press/v235/achtibat24a.html) is a candidate for attention extensions, not evidence that Engine's graph transformer is supported.

## Validation and release

Shared plumbing tests cover every label ordering, type-strict class selection, Platt calibration (clamped, fitted and unfitted), bitwise binding of both probability columns, sign folding, rejection of every unsupported profile before hooks, the bounded enumerator against toy rules, validator mutations, default modules identical to the base `1909c5b9` wheel, and v1 outputs regenerated byte for byte against bytes produced by the installed base wheel.
The legacy rule re-implemented on the shared enumerator matches the original walk by walk within 1e-12 relative on float64 fixtures.
Weight-gamma tests use hand-computed fixtures (signed inputs, bias signs, zero and near-zero denominators, sign folding, head inversion, budgets and top-k), equality with the legacy rule on nonnegative inputs, Engine's original capture, and an independent autograd implementation of the paper's forward rewrite on signed and three-layer graphs; an independent reviewer reference matched 38,352 walks within 5.2e-14 of `max(1, A)`.
Degree-T tests compare every walk with an independent gated reimplementation using nested autograd at `T = 2` and `T = 3` (within 4e-16 relative), bags with tied-gate derivatives divided by `alpha!`, source vectors with first-order gradient-times-input, homogeneity jointly and per layer, and kinks, mask disagreements and the `s = -1` ledger.
Finite differences validate ordinary derivatives; they are not an oracle for LRP's relevance redistribution.

`scripts/xai/walk_binary_native_gate.py --development --output <new directory> --methods <ids> --gamma 0 0.25` trains a small binary fixture with and without biases.
It checks class orientation, exact prediction parity with and without evidence, the encoder boundary bit for bit, native-versus-offline row equality including diagnostics, fresh-process replay, rejection of shifted mean denominators, and replay under the recorded precision; with `--wheel` instead of `--development` it also checks that installed modules equal the wheel's bytes.
`scripts/xai/walk_native_gate.py` remains the regression-only v1 gate.
Development and review runs of this feature used the committed sources overlaid on an installed base `1909c5b9` wheel, whose native extension is built from the same unchanged native source; source-overlay runs are development evidence, not installed-wheel acceptance.
The [installed-wheel procedure](evidence-validation.md) was written for `supply-chain-xai` and its script refuses other branches; installed-wheel acceptance of this feature is recorded with its release evidence.
Repeated-batch performance, GPU qualification and empirical faithfulness reports are distinct gates and must not be inferred from unit-test results.
