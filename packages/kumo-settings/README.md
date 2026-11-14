# kumo-settings

Process-wide settings shared by the Kumo packages, in the standard-library-only
module `kumo.settings`.

```python
import kumo.settings

kumo.settings.configure(scratch_dir="/mnt/nvme/kumo")
kumo.settings.scratch_dir()  # PosixPath('/mnt/nvme/kumo')
```

`configure` is the only way to change a setting. The scratch directory defaults to
`~/.cache/kumo-engine`. It must be a local filesystem path; whatever writes under it creates it.

From the monorepo root, run the tests with:

    pytest packages/kumo-settings/tests -q
