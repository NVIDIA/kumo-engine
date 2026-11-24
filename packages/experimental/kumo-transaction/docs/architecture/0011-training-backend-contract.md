# 0011 — Training backends execute the model's semantics; they do not define them

Status: accepted

*Update, 2026-10-03:* `training.backend: nemo` is removed in 2.0.0a2; `fsdp2` trains the same HuggingFace Llama.

## Context

v1's published checkpoint learned the identity map (ADR 0017). Its dataset set
`labels = input_ids`, expecting the loss to shift them; the training framework's
loss expected the dataset to have shifted already. Each side assumed the other
owned label alignment, and nothing checked.

The SDK is about to gain backends it does not write: NeMo AutoModel driving its
own recipe and distributed checkpointer, and Megatron Bridge with tensor and
pipeline parallelism. Each has its own loss, data loader, and checkpoint format,
so each is a new place for v1's defect to come back. Until now `build_trainer`
was a string `if`/`else`, backends declared nothing about what they support,
and incompatible settings surfaced mid-run, if at all.

## Decision

**The SDK owns semantics; a backend owns execution.**

| owned by the model (SDK) | owned by the backend |
|---|---|
| objective and label alignment | device placement |
| attention, padding, and segment masks | data parallelism and sharding |
| batch contract and tokenization | tensor and pipeline parallelism |
| loss computation | precision policy |
| | executing the optimizer and schedule |
| | checkpoint I/O |

A backend calls the composed model's forward with `compute_loss=True` and
backpropagates the loss it returns. It never computes its own loss, and never
builds labels.

### Registry

`training.backend` names are resolved by `tfm.training.backends`, which follows
the tokenizer registry's pattern. `@register_backend("name")` registers a class;
`register_lazy_backend("name", "module:Class")` registers an optional extra
without importing it, so `import tfm` stays free of extras (ADR 0006).
Registered today: `eager`, `ddp` (replicated data parallelism, added with
ADR 0012), `fsdp2`, and `nemo`, a deprecated alias for `fsdp2` with the model
built through NeMo AutoModel. `shards_parameters` distinguishes FSDP-style
backends, whose ranks each hold a slice of the weights, from `ddp`, whose
ranks each hold a full replica.

### Capabilities

Every backend class declares `capabilities: BackendCapabilities`
(`tfm.contracts.capabilities`), and its `name` must equal the registered name:

- `backbones` and `batch_layouts` it can train;
- `requires_cuda` and `multi_process`;
- `segment_masks`, whether packed rows keep their block-diagonal mask, which
  `training.packing` requires;
- `consolidated_hf_export`, `distributed_checkpoint`, `tensor_parallel`,
  `pipeline_parallel`;
- `precisions`, the `training.precision` values it executes as declared.
  `bf16` means bf16 compute over fp32 master weights, gradients, and optimizer
  state. `eager` and `ddp` declare `fp32` and `bf16`; `fsdp2` declares only
  `bf16`, which is what its mixed-precision policy computes. A backend that
  cannot honor the configured precision refuses it rather than silently
  running another.

The runtime checks the configuration against them twice: before data
preparation, which is the slow step at book scale, and again once the
tokenizer's layout is known. A mismatch raises `UnsupportedCombinationError`
naming every unmet requirement. Backends keep their own checks as well; the
declaration makes refusal early, it does not replace refusal.

### Checkpoint contract

- **Required.** A backend that trains the `huggingface` backbone writes a
  consolidated HuggingFace checkpoint (`save_pretrained` layout, full
  unsharded weights) to the `checkpoint_dir` it is given, and reports its path
  as `stats["hf_checkpoint"]`. This is the format the probe, the notebooks, and
  every downstream consumer read. The eager backend does this too.
- **Optional.** Sharded, resumable distributed checkpoints, declared as
  `distributed_checkpoint`. No backend declares it yet; NeMo AutoModel's
  checkpointer is the intended first. It never replaces the consolidated
  export.

### Contract suite

`tests/contract/test_training_backends.py` runs every registered backend
against eager training on the same tiny model and batches:

1. **Parity.** First and final loss match eager over a fixed number of steps
   from identical weights (tolerance allows bf16 compute). Every backend that
   declares `bf16` is also run in bf16 against fp32 eager, and must keep fp32
   parameters.
2. **No copying.** After training on sequences whose next token is always
   predictable, next-token loss must be below half the same-token loss. A
   negative control confirms the check flags a copying model. This is the test
   that would have caught v1.
3. **Masks.** Batches are not mutated, and replacing every masked-out token
   with random ids changes no step's loss.
4. **Checkpoint round-trip.** The consolidated export reloads with
   `AutoModelForCausalLM` and reproduces the trained model's logits.
5. **Determinism.** Two runs from the same weights and seed agree.

A backend that cannot run in the test environment is skipped by its declared
capabilities (`requires_cuda`, missing extra), never by catching its failure.
On CPU CI the GPU backends skip; on a GPU host they run.

### Weights are the model's

A backend may change how the model is constructed, as the `nemo` alias does,
but not its weights: it carries the SDK-built state dict across strictly. A
rebuild that re-initializes would silently discard
`model.huggingface.pretrained_path` and break parity.

## Consequences

- Adding a backend is: a class with `capabilities`, a constructor taking
  `checkpoint_dir`, `fit`, one registration line, and a green contract suite.
  Phase 4's `nemo_automodel` and `megatron_bridge` backends are held to exactly
  this.
- `Trainer.fit` gains `shuffle` and `precision` keywords and the protocol a
  `capabilities` attribute. Third-party trainers must add all three.
- The published v1 defect class, a loss that disagrees with the model about
  label alignment, now fails a test rather than a benchmark.

## Alternatives rejected

- **Let each backend bring its own loss.** It is the arrangement that produced
  v1's defect.
- **Declare capabilities in config instead of on the class.** A config can
  claim what the code does not do; a class attribute is checked against the
  code that implements it by the contract suite.
- **Skip failing backends in the contract suite with `xfail`.** A backend that
  cannot pass is not a supported backend.
