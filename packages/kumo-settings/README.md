# kumo-settings

Process-wide scratch and compute settings shared by the Kumo packages. Importing
`kumo.settings` and configuring scratch space use only the standard library:

```python
import kumo.settings as settings

settings.configure(scratch_dir="/mnt/nvme/kumo")
settings.scratch_dir()  # PosixPath('/mnt/nvme/kumo')
```

Scratch defaults to `~/.cache/kumo-engine` and must be a local filesystem path.
Passing `scratch_dir=None` keeps its current value; configuration creates no
scratch directory.

For remote artifact preparation, install `kumo-engine[compute]` and initialize
Ray first. Compute setup lazily loads Engine's worker-pool implementation:

```python
settings.configure(
    cluster_config=settings.ComputeClusterConfig(
        worker_root="/mnt/nvme/kumo",
        resources={"artifact_worker": 1},
    ),
    artifact_root="s3://<bucket>/artifacts",
)
```

`artifact_root` accepts a URI/path, `ArtifactRoot(location, connector)`, or a
`(connector, path)` pair. Connector authentication passes unchanged to Engine.
Replacing compute settings releases the old pool and invalidates its handles,
while preserving worker caches and the caller's Ray runtime.

Omitting `cluster_config` preserves the pool, including on scratch-only calls.
Pass `cluster_config=None` to release it and return to local execution. The
`scratch_dir` override stays applied if worker startup fails.

`kumo.engine.configure(...)` delegates to `settings.configure(...)` for
compatibility. Engine re-exports `ComputeClusterConfig` and `ArtifactRoot`.
The older `kumo.engine.init(...)` also delegates, with its existing behavior of
selecting local execution when `cluster_config` is omitted.

From the monorepo root, run the tests with:

```bash
pytest packages/kumo-settings/tests -q
```
