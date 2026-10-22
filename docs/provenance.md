# Code provenance

kumo-engine began on 2026-10-09 as a copy of `kumo-ai/kumo` at `2ed61536f649ca542cb3c90e83bad3cae905a520`, without its Git history. Every path that existed then has the same path in that commit, so earlier history, including the original authors, pull requests, and reviews, is in kumo-ai/kumo. Components that kumo-ai/kumo imported from other repositories have import commits that name their source repository and commit. Changes later ported from kumo-ai/kumo name their source in the commit message. These repositories are private. Third-party code and data are covered by the [third-party notices](../3rdparty/README.md).

## Viewing earlier history

With read access to kumo-ai/kumo, attach its history to this repository's first commit so that `git log` and `git blame` continue into it:

```bash
git remote add kumo <kumo-ai/kumo clone URL>
git fetch kumo master
git replace --graft 1c2090ef9028a1e03a273c0b1a02476ed40a7685 2ed61536f649ca542cb3c90e83bad3cae905a520
```

The replacement is local and is not pushed. Remove it with `git replace -d 1c2090ef9028a1e03a273c0b1a02476ed40a7685`. Pull request numbers in that history refer to kumo-ai/kumo, or to the source repository for imported history. Use `git blame -C` to follow code that kumo-ai/kumo moved between files.
