# Optional local SDM example

`connector_to_prediction.py` creates deterministic PyArrow user/order tables,
writes Parquet source files with `fixture.py`, builds a connector graph snapshot,
and prepares an FM payload through public PyTorch plans.

Install the candidate merged `kumo-engine[fm]` wheel. Calling
`build_index_and_load_batch()` uses only those packages. For model inference, additionally install
the pinned SDM environment documented at the top of
[`walkthrough.ipynb`](walkthrough.ipynb) and run:

```bash
cd packages/experimental/kumo-fm/examples/local_sdm
python connector_to_prediction.py
```

`walkthrough.ipynb` uses the same `build_index_and_load_batch()` helper to write
source files, build or reuse the graph index, and load an FM batch. Run it from this folder;
it writes under `scratch/notebook/`, while the script uses
`scratch/connector-to-prediction/`. `source/` holds regenerable Parquet files;
`producer/` holds managed base indexes and derived tasks/caches. Repeated runs
reuse completed snapshots; closing tasks retains bases/caches without automatic
collection. The 30-day context label window ends before the November query.

`adapter.py` converts the retained Arrow batch into public SDM `TableTensor`
and `RelatedTables` inputs. Only that conversion and model inference need SDM.
The classifier may download a pretrained checkpoint; use `HF_HOME` for a
writable cache. The script uses CUDA when available and CPU otherwise. SDM's
model hop setting is independent of the loader's sampling depth. Both arrays
and Arrow IPC encode the same retained batch after model prediction.

Notebook outputs are cleared; they are not claims of a fresh model run.
