# Small PR172 fixtures

The committed archive contains the fixed synthetic inputs used by the eight
PR task checks. It contains 556,127 adtech rows, 1,442,968 delivery rows, and
74,416 commerce rows. These are source table rows across all tables in each
family. The task adapters derive their training and prediction rows from them.

Preparation reads local files only. It needs no token, private repository,
generator, or download. Run it with the candidate environment, which already
contains PyArrow:

```sh
python packages/kumo-engine/test/pr172/prepare.py --output-dir /path/to/pr-inputs
```

The result reports `data_root` for the extracted data and `source_root` for
the committed graph builder in this directory. Preparation trains no models.
It sets no learning score threshold. To check the complete archive without
keeping extracted data, use `--check` instead of `--output-dir`.

`fixtures.json` records the compressed archive hash and every retained file's
size and hash. Extraction rejects changed bytes, unsafe paths, duplicate
members, links, and unexpected files. It verifies the original manifests and
validation stamps, exact table counts, and actual Parquet row counts. Reuse
of extracted data repeats these checks. A partial or modified directory fails
without replacing it.

The archive keeps 23 table Parquet files, three original manifests, and three
original validation stamps. It excludes hidden generator state, private
Python source, generation logs, and machine paths. The retained table bytes
match the measured inputs exactly. The original validator ran before this
archive was made. Its retained stamp is historical evidence, while the loader
independently checks the files and row counts used by the PR worker.

`provenance.json` and `config.json` retain the original generator revision,
source fingerprints, configuration, and expected counts for audit. These
records do not create a dependency on the original source repository at run
time. A fixture change requires updating the archive, integrity metadata,
and provenance together. Keep generated training outputs and model bundles
outside this directory.
