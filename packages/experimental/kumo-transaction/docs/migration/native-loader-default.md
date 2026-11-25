# Native loader default after 2.0.0a4

Recipe configurations now default to `data.loader: kumo_pytorch`. This applies
to every sequence composition and to the experimental graph recipes. Existing
model APIs, tokenizers, label rules and objectives are unchanged. To reproduce
the previous Python loader explicitly:

```yaml
data:
  loader: python
```

Install with `uv sync --frozen --extra dev --extra native` (add `--extra kumo` for the GNN).
The `native` extra installs the optional external backend; select the Python
loader explicitly to use the kit without it. The lockfile pins Kumo Engine main at
`5c3a22cacc43e8de99531693e3b6612f0b34d637`, including its experimental
`kumo-pytorch` package and matching DiskGraph runtime. This is a source build:
install Rust through rustup, a C/C++ compiler and the usual Linux build tools
first. A cold native build can take tens of minutes. The published
`kumo-engine==1.0.1` bundle is incompatible with this loader; do not install it
on top of the source packages, because it also owns the `diskgraph` namespace.
The GNN extra now selects the matching `kumo-core` source package.

A configured token cache still selects its existing batch builder. Packing also
retains the existing packer, operating on natively retrieved histories. Neither
setting changes the tokenizer or fits native vocabularies. Run manifests name
the history loader; graph manifests additionally identify the native source
revision and separate Python future-target builder. Changing loader changes
resolved-config identity: use a new run directory, and do not reuse an old
managed checkpoint under a different resolved configuration.

Native training collation pads within each minibatch rather than retaining the
previous whole-corpus padding width; valid tokens, masks and labels keep the
same meaning. This can change stochastic training trajectories, so fresh runs
are required for comparisons.

Native sequence selection preserves the inclusive observation start and exact
cutoff, including subsecond events and tied timestamps. Graph sampling has a
narrower schema/clock contract and its own recent-fanout tie ordering; selecting
`python` is required outside that contract. See the
[worked integration guide](../../examples/kumo-engine/README.md#canonical-recipes)
for commands and limitations. These reference integrations still import a
bounded object corpus for validation, tokenization, targets and probes; they do
not establish full-book out-of-core training or native future-aggregate parity.
