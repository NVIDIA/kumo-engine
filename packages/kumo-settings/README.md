# kumo-settings

Process-wide settings shared by the Kumo packages, in the standard-library-only
module `kumo.settings`.

```python
import kumo.settings

kumo.settings.configure(scratch_dir="/mnt/nvme/kumo")
kumo.settings.scratch_dir()  # PosixPath('/mnt/nvme/kumo')
```

The scratch directory resolves, first match wins, from a `kumo.settings.scoped(scratch_dir=...)`
block, then `configure(scratch_dir=...)`, then the `KUMO_SCRATCH_DIR` environment variable, then
`~/.cache/kumo-engine`. It must be a local filesystem path; `configure` creates it if missing.

From the monorepo root, run the tests with:

    pytest packages/kumo-settings/tests -q
