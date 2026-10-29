# Small PR172 inputs

`prepare.py` reuses the three synthetic generators from the pinned revision in
`provenance.json`. It generates 556,127 adtech rows, 1,442,968 delivery rows,
and 74,416 commerce rows. It trains no models and defines no quality limits.

The upstream source is acquired separately. It is not redistributed here.
A trusted CI job needs a token with read access to that source repository.
The default token for this public repository cannot read a private repository.
Use a checkout at the exact `source_revision`, then pass its `scale-bench`
directory to this command:

```sh
python3.12 packages/kumo-engine/test/pr172/prepare.py \
  --source-dir /path/to/pinned-upstream/scale-bench \
  --output-dir /path/to/pr-inputs
```

Use a separate generator environment with the four dependency versions in
`provenance.json`. The worker environment can resolve its own dependencies.
The command returns `data_root` and `source_root` for the GPU task worker.
It keeps the source copy and all generation and validation logs under the
output directory. An incomplete output directory fails without overwriting it.

The wrapper checks source fingerprints before running copied code. It builds
and checks the original placeholder profiles before data generation. It then
runs the existing structural validator with full file checksums and the
recorded small-profile settings. The wider distribution checks match those
small inputs. They do not allow model score drops. Exact table counts,
configuration identity, and file fingerprints remain strict.

A complete output can be cached using the provenance file hash and generator
dependencies. Reuse checks the copied source, generated profiles, manifests,
validation stamps, complete table inventory, and every table file hash.
Generated Parquet and acquired source must stay outside the committed tree.
