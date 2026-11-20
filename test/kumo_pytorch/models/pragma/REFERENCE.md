# PRAGMA source record

Copied from Arena revision
`b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`, `models/pragma/`.
Usage and current validation scope are in [the test model guide](../README.md).

The neural core derives from Dynamiq's pragmatiq 1.1.0, revision
`eb6928c6b33d3c6010f04a4286aa52426654a78d`, with the historical Kumo overlay at
`31e90aa60757f62e2d696636cb88cf688b2f3ec0`. Arena replaced custom attention and
loss kernels with ordinary PyTorch operations while retaining the packed
architecture and parameter names. `native.py` and `_kumo.py` provide the native
Kumo batch interface. The hash guard pins the reviewed local token model. Unused
frozen-text encoders, projection/reconstruction branches and metadata have been
removed; the retained token architecture and parameter names are unchanged.
Unused size presets and the standalone loss helper are also omitted; shared task
consumers own loss calculation. Upstream licenses and notices remain.

Retained attribution: [LICENSE](LICENSE), [NOTICE](NOTICE),
[overlay notice](NOTICE.overlay.md) and [historical Liger license](LICENSE.liger).
The Liger-derived kernel itself is not included.

PRAGMA uses shared embeddings, event/profile/history encoders, time/calendar
features and a tied masked-value head. Attention is bidirectional within each
segment; Kumo must restrict inputs to history before the prediction cutoff.
S01 text fields use Kumo-produced BPE token IDs through the same token model.
No frozen text encoder or pretrained weights are included.
