# AML test models

These models belong to the repository testing suite, outside the installed
`kumo.pytorch` package. They require no Arena checkout or imports.

| Workload | Consumer | Native inputs | Objective |
| --- | --- | --- | --- |
| account-involvement-next24h | GraphSAGE | graph | Binary cross-entropy |
| account-involvement-next24h | nuFormer | token history | Binary cross-entropy |
| account-involvement-next24h | combined | graph, token history, twelve counts | Binary cross-entropy |
| masked-value-pretrain | PRAGMA | nested field history, calendar, elapsed time | Masked-value cross-entropy, smoothing 0.1 |
| next-token-pretrain | nuFormer | token history | Next-token cross-entropy |

[aml.py](aml.py) builds a consumer from a fitted Kumo task and an entry in
[the AML selection](../workloads/aml.json). Vocabulary sizes come from fitted
state. The small default dimensions are for the prototype.

```python
import torch
from test.kumo_pytorch.models.aml import build_model

model = build_model(task, selection, "graphsage")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for batch in task.batches(split="train", batch_size=32):
    optimizer.zero_grad(set_to_none=True)
    result = model.training_step(batch)
    if result.weight:
        result.loss.backward()
        optimizer.step()
```

The CPU training step preserves the input batch and excludes padding and invalid
targets. Kumo supplies masking and already-shifted next-token targets. GraphSAGE
and combined require v2 selections and original `prediction_time_ns` request
metadata; sampling uses the shifted execution cutoff. Combined keeps all three
branches in the same request order. Accelerator execution is not validated here.

The [AML prototype runner](../docs/aml-prototype.md) prepares downloaded sources
and anchors and passes the resulting tasks to these consumers. Complete-population
execution remains pending in [the plan](../PLAN.md). The tiny
[model checks](../model_checks/test_aml.py) share the prototype's input declarations.

## Copied source

Model Python modules were copied from Arena revision
`b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`, under its `models/` directory.
The `graph_sequence_aggregates/` family implements **combined**. Standalone CLI
smokes, platform locks and Arena orchestration were omitted. `aml.py` adapts the
model construction and objectives from `profiling/execution/model_execution.py`.
Its lifecycle adapter also applies a binary involvement head to PRAGMA's user
representation during finetuning. The local nuFormer `forward_kumo` now removes
its internal empty-history padding position from emitted logits/hidden states,
preserving alignment with zero-width Kumo next-token targets. Fusion uses an
encoder-only GraphSAGE: no standalone prediction head or final non-root updates
are registered, because neither contributes to its root-based prediction.
Earlier graph layers remain intact. Standalone GraphSAGE is unchanged; the
copied RelBench class bodies remain unchanged. Fusion checkpoints produced
before this change contain extra parameters and need to be regenerated.
Test fixtures adapt `kumo_specs/fixtures.py` and selected `typed_contracts.py`
builders. Historical Arena validation is not evidence for this port.

GraphSAGE retains [RelBench provenance](graphsage/UPSTREAM.json) and
[license](graphsage/LICENSE.relbench). PRAGMA retains its licenses and notices;
see [its source record](pragma/REFERENCE.md). nuFormer and combined retain
Arena's implementations with the local adaptations described above. Optional PRAGMA text encoders are not used by AML.
