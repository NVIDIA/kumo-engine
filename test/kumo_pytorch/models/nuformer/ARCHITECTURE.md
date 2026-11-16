# nuFormer transaction decoder

This is Arena's independent reference implementation inspired by
[Your Spending Needs Attention](https://arxiv.org/abs/2507.23267).
It supplies a randomly initialized transaction backbone, not pretrained weights
or a reproduction of the paper's predictive results.

The decoder uses causal PyTorch attention, no positional embeddings, pre-layer
normalization, GELU feedforward blocks and a final layer normalization. Input
and output embeddings are tied by default. The last valid token represents the
account; empty histories produce zero account representations. Padding is
excluded from attention and pooling.

`encode_kumo` supplies representations without vocabulary logits.
`forward_kumo` also produces next-token logits. Kumo's next-token targets are
already aligned; the AML wrapper applies cross-entropy without another shift.
For zero-width histories, the encoder uses an internal masked position, while
`forward_kumo` returns zero-width logits and hidden states to match emitted inputs.
For account involvement, the wrapper adds a binary classification head.

The named tiny/small/large presets are preserved, but the AML prototype selects
smaller explicit dimensions. See [the model guide](../README.md) for source
provenance, usage and validation limits.
