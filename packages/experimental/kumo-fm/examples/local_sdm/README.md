# Optional local SDM example

This example keeps SDM and its Arrow compatibility handling beside the
notebook. Install matching DiskGraph, `kumo-pytorch`, and `kumo-fm` wheels
from one engine revision, then the pinned SDM environment in
`requirements.lock`. The model may download the
`nvidia/Kumo-Relational` v1.0.1 checkpoint; the prior validation used
snapshot `2bd603d3d8f25f67a7aaa8579908595e20567e22`.

The example includes `fixture/bundle/` and `fixture/workload.json`. Run the
notebook from this directory to use them by default. Set `FM_SDM_FIXTURE` to
another directory containing `bundle/` and `workload.json` to use your own
eligible `native_multihop` workload. Set `FM_SDM_BUNDLE` when its bundle has
another path. The notebook opens or creates `fm-index/` in the fixture directory, samples one
batch, converts its retained Arrow tables through public
`TableTensor.from_arrow` and `RelatedTables`, and exercises both model
call forms. The helper only repairs zero-chunk columns and read-only string
buffers when SDM requires it. The same batch is encoded as arrays and IPC.

Regenerate the included smoke case with
`python benchmark/fm-payloads/make_fixture.py /tmp/fm-case` from the repository
root, then copy `bundle/` and `workload.json` into this directory's `fixture/`.
You can also set `FM_SDM_FIXTURE=/tmp/fm-case` to use the generated copy.
The notebook includes the local execution output: both model call forms
agreed with a maximum absolute difference of 0.0 on this fixture.

Direct conversion avoids HTTP serialization; it does not promise that SDM
tensor construction or inference is zero copy. The source baseline is
fm-dataloader PR #27 at
`b4516394e98929c5b4752745866d5e3138276255`.
