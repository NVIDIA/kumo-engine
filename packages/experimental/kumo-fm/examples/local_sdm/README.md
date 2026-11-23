# Optional local SDM example

`connector_to_prediction.py` is the primary connector-to-model example. Run it
from this directory after installing matching `kumo-engine`, DiskGraph,
`kumo-pytorch` and `kumo-fm` wheels from one engine revision, plus the pinned
SDM checkout in `requirements.lock`. `kumo.engine.datasets.shop()` supplies
the reproducible 1,000-user shop Parquet sample; the script writes producer and
consumer state under `scratch/connector-to-prediction/` here. The public
`KumoRelational` classifier downloads its pretrained checkpoint on first use;
set `HF_HOME` to a writable model-cache directory. A CUDA device is used when
available, otherwise CPU. The example's 30-day future label window ends well
before the November query cutoff. Its one sampling hop is independent of the
model's two-hop setting. No SDM package is imported by `kumo.fm` itself.

```bash
cd packages/experimental/kumo-fm/examples/local_sdm
python connector_to_prediction.py
```

The separate `walkthrough.ipynb` uses the included frozen bundle fixture and
keeps SDM and its Arrow compatibility handling beside the notebook. Install
matching DiskGraph, `kumo-pytorch`, and `kumo-fm` wheels
from one engine revision, then the pinned SDM environment in
`requirements.lock`. The model may download the
`nvidia/Kumo-Relational` v1.0.1 checkpoint; the prior validation used
snapshot `2bd603d3d8f25f67a7aaa8579908595e20567e22`.

The notebook includes `fixture/bundle/` and `fixture/workload.json`. Run it
from this directory to use them by default. Set `FM_SDM_FIXTURE` to
another directory containing `bundle/` and `workload.json` to use your own
eligible `native_multihop` workload. Set `FM_SDM_BUNDLE` when its bundle has
another path. The notebook uses `fm-index/` for its FM workspace and configures
`kumo-scratch/` for managed storage, both in the fixture directory. Each opening
builds a fresh base; repeated requests in that loader reuse it. Base indexes and
caches remain after closing, with no automatic garbage collection. It samples
one batch, converts its retained Arrow tables through public
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
