# G-ARTIFACT-001: a written artifact reloads in a fresh process to the same batches

## Rule

`task.to_artifact().save(directory)` writes four files: the task's contract
(`contract.json`), its fitted state (`fitted-state.json`), its fingerprints and
output schema (`task.json`), and its manifest (`manifest.json`), which also
records each of the four files' SHA-256 digest. `Artifact.load(directory)` reads
them back in any process, reading no data and fitting nothing. The loaded
artifact equals the writing task's `to_artifact()`: its contract, fitted state
and fingerprints equal the writing task's, and its bound
contract, the contract with the encodings that the fitted state records pinned,
equals the writing task's `bound_contract()`.

Preparing the loaded contract over the same rows and materializing the same
anchors with `fitted=artifact.fitted_state` fits nothing and reads no fitting
anchors, and produces the same batches as the writing task for the same batch
arguments: the same inputs, targets and metadata, including the contract and
artifact fingerprints. The task's fingerprints equal the writing task's.

Fitted target state is part of the artifact. A reloaded label vocabulary is the
trained one; a label the training data never had encodes as the unknown ID
rather than being fitted.

## Boundaries

The dataset and engine may differ from the ones the artifact was written with;
the fitted state is reused as written. State is refused under a contract whose
input fingerprint differs (a different schema, sample or inputs), and complete
state is refused under a contract whose full fingerprint differs. The input
half, `Artifact.load(directory).fitted_state.inputs`, is reused under a contract
with the same input fingerprint and other targets, which fits its own targets.
The artifact is the only on-disk form of fitted state.

`Artifact.load` refuses three things, and nothing is refitted to fill a gap:

- a missing file;
- a file that does not match the digest `manifest.json` records for it, named
  in the refusal (the manifest's own digest covers everything in it but the
  digests), or a manifest that records none, as one written before digests;
- fitted state that does not reproduce exactly: every encoder, ID space and
  binding, rebuilt from `fitted-state.json` under the encodings the contract
  declares, must serialize back to it exactly, and each ID space must number
  exactly the IDs its members produce;
- a file that disagrees with the contract and fitted state, named with its
  field: target state in `fitted-state.json` for other targets than the
  contract's fitted ones; a manifest other than the one the contract and
  fitted state give, or a `task.json` whose output schema or recomputable
  fingerprints (contract, inputs, input and target state, output schema,
  task) differ from theirs. The dataset and anchors fingerprints can't be
  recomputed without data and are taken as written.

The digests catch files not written together or edited independently; they are
not a signature against deliberate tampering. The `fit` record in
`fitted-state.json` is informational and is not checked.

`save_weights(path, model_state, fitted)` records the contract fingerprint and
the fitted-state fingerprint beside the weights; `load_weights(path, fitted)`
refuses weights recorded for any other contract or fitted state.

## Scope

The golden case below fixes raw entity values, which the conformance harness
checks on the DiskGraph engine through `prepare` → `materialize` → `preview`.
The golden format cannot express a fresh process or encoded batches, so those
are covered by tests in
[test_inference_lifecycle.py](../../../../tests/test_inference_lifecycle.py):
the golden case reloaded in a child interpreter, whose decoded values must equal
the answer below; the same batches from a child interpreter on the DiskGraph
engine, for a columns/events/graph contract with labels and a key/value token
contract with masking; every fitted-state kind the reference recipes use; label
vocabularies; and each refusal above. Tests in
[test_fitted_state_persistence.py](../../../../tests/test_fitted_state_persistence.py)
also cover the contract, fitted state and artifact pickled into `spawn`
DataLoader workers on the DiskGraph engine, giving the same batches; the loaded
artifact equal to the writing task's `to_artifact()` and saving back to the same
files; the loaded artifact's input half reused under a contract with the same
inputs and other targets, on the DiskGraph engine; and the informational `fit`
record in `fitted-state.json`, kept through reuse and outside the fitted-state
fingerprint that weights are tied to.

Equality is exact on CPU. GPU placement and other PyTorch versions are outside
this rule. Model weights are the caller's; this package reads and writes them only
through the two helpers above.

## Golden examples

- [C-ARTIFACT-001](golden/C-ARTIFACT-001.json): three users at time 10 whose
  countries and event counts are US and 2, FR and 0, US and 1. The reloaded
  artifact gives the same counts, and its country vocabulary decodes the IDs
  back to US, FR and US.
