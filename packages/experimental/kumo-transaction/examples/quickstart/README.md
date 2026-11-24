# Quickstart example

The smallest composition of the core surface. `uv run tfm smoke` trains and
scores the same stack on generated data on CPU
(`recipes/blueprint_v1/config.smoke.yaml`); [`recipes/`](../../recipes) has
the reference architectures.

```python
from tfm import TrxFM, Flat, Decoder, NoPE, EagerAttention, LastTokenPool, NextEvent, Binary

model = TrxFM(
    granularity=Flat(Decoder(depth=2, mask="causal")),
    position=NoPE(),
    attention=EagerAttention(),
    objective=NextEvent(),
    readout=LastTokenPool(),
    heads={"fraud": Binary()},
)
```
