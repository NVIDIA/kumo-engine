# PRAGMA source record

Copied from Arena revision
`b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`, `models/pragma/`.
Usage and current validation scope are in [the test model guide](../README.md).

The neural core derives from Dynamiq's pragmatiq 1.1.0, revision
`eb6928c6b33d3c6010f04a4286aa52426654a78d`, with the historical Kumo overlay at
`31e90aa60757f62e2d696636cb88cf688b2f3ec0`. Arena replaced custom attention and
loss kernels with ordinary PyTorch operations while retaining the packed
architecture and parameter names. `native.py` and `_kumo.py` provide the native
Kumo batch interface. The neural core's hash guard pins the local source. The
suite's license-header and whitespace cleanup updated that pin after verifying
that the core's Python AST was unchanged. Upstream licenses and notices remain.

Retained attribution: [LICENSE](LICENSE), [NOTICE](NOTICE),
[overlay notice](NOTICE.overlay.md) and [historical Liger license](LICENSE.liger).
The Liger-derived kernel itself is not included.

PRAGMA uses shared embeddings, event/profile/history encoders, time/calendar
features and a tied masked-value head. Attention is bidirectional within each
segment; Kumo must restrict inputs to history before the prediction cutoff.
Optional frozen text encoders remain available in the copied code but are not
part of the AML workload. No pretrained weights are included.
