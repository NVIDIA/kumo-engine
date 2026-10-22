# In-tree local SDK example

This example uses the local `kumo.engine` interface in
[packages/kumo-engine](../../packages/kumo-engine/README.md) from the current
checkout. It creates a four-row CSV file, infers a table schema, validates a
binary-classification query, and builds and verifies a real DiskGraph snapshot.
It uses temporary source and scratch directories, which are removed on exit.
No Kumo service, account, connector credentials, or external dataset is needed.

## Run with prepared runtime dependencies

Use a [development environment](../../docs/development/setup.md) with the
runtime dependencies already installed, including DiskGraph and its native
ingest executable. Follow the [Faiss selection guide](../../docs/development/setup.md#choose-the-core-faiss-dependency)
to choose `kumo-core[faiss-cpu]` or `kumo-core[faiss-gpu]` for retrieval.
Bare `kumo-core` supports other tasks and compatible preinstalled Faiss.
From the repository root:

```bash
PYTHONPATH=packages/kumo-engine:packages/kumo-api:packages/kumo-pql:packages/kumo-core/src \
  .venv/bin/python examples/sdk-local/check.py
```

The command selects the local SDK, API schemas, parser, and core runtime from
this checkout. The script rejects a `kumo.engine` package from another location
before importing it.

Expected result:

```text
Local SDK: CSV ingest, query validation, and snapshot verification passed.
```

This example does not train a model or make predictions. Its successful run
does not establish a production CPU or GPU support profile.
