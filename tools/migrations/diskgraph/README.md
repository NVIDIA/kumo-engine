# DiskGraph source import

`packages/diskgraph` is the buildable mirror of `kumo-ai/kumo-diskgraph` at the
revision recorded in `source_revision.txt`. The initial import rewrites that
repository beneath `packages/diskgraph` and merges the rewritten ancestry into
Kumo so the original file history remains reachable.

The imported tree is byte-identical to the frozen source revision. Run the
verification from a checkout that can read the source repository:

```bash
python tools/migrations/diskgraph/verify_import.py \
  --source-repo /path/to/kumo-diskgraph
```

CI verifies the committed mirror against the manifest without requiring
cross-repository credentials:

```bash
python tools/migrations/diskgraph/verify_import.py
```

The generated manifest records every source path, destination path, mode, and
blob identifier. Subsequent synchronization is one-way from
`kumo-diskgraph/main`; Kumo builds the native wheel from the mirrored source.
Monorepo build integration lives outside the mirrored directory.

Do not edit `packages/diskgraph` in Kumo pull requests. Merge the change into
`kumo-diskgraph/main`; its dispatch workflow opens or updates the mechanical
Kumo synchronization pull request and credits the upstream authors.

The mirrored tree retains DiskGraph's formatting and lint policy. Kumo's
repository-wide format, lint, and type checks exclude only
`packages/diskgraph`; Kumo integration code remains covered by those checks.

The history-bearing import pull request must be merge-committed. Squashing or
rebasing it would discard the imported ancestry.
