# FM correctness testing

Run from the repository root with matching DiskGraph, kumo-settings,
kumo-pytorch, kumo-fm and pytest installed. See [source installation](README.md#source-installation)
and the repository [setup guide](../../../docs/development/setup.md).
The native module must include `diskgraph.encode_fm_rows`; an older wheel is
not a substitute for building the candidate's DiskGraph. No cloud credentials,
GPU, SDM model, or downloaded dataset is required by the new tests.

## Fast suite

```bash
uv run --no-sync --package kumo-fm python -m pytest -ra packages/experimental/kumo-fm/tests
```

The default excludes `dev` campaigns. The `kumo-fm` workflow runs the fast suite
against both source and finished wheels on Linux with Python 3.11, 3.12 and
3.13. It also checks pure Python FM artifacts, a fresh environment without
DuckDB, dependency resolution, and installed connector/engine interoperability.
Automatic triggers are limited to `packages/experimental/kumo-fm/**` and the
workflow itself; other dependency changes alone do not trigger it. Manual
`workflow_dispatch` remains available for checking a changed native stack.
This matrix does not establish Windows, macOS, or Python 3.14 coverage.

`test_correctness.py` builds real native indexes from shuffled Parquet row
groups. `_correctness.py` computes expected histories and labels directly from
source rows, without FM's selection, assembly, or encoding helpers. It tests:

- Both declared relation orientations and all self-relation traversal directions.
- Context windows `(anchor, anchor + horizon]`, history at or before the anchor,
  and query cutoffs exactly at or before the end of label observability.
- Multiple internal batch sizes, repeated entity/time pairs, nonmonotonic IDs,
  IDs shared across splits, int64 extremes and Unicode string entity keys.
- Missing history, nullable numeric/string/bool features, regression and binary
  labels, query-only requests, and payloads retained after request/loader close.
- Exact arrays cells and Arrow IPC round trips against the retained Arrow tables.

The contract tests additionally cover invalid populations, timezone offsets,
class maps, budgets, feature projections, corrupt labels, native ownership,
endpoint bounds, anchor permutations, and failed preparation cleanup.
`test_native_json_buffers.py` checks unaligned Arrow buffers and independently
chunked columns sliced across null-bitmap boundaries. These run alongside the
existing float-format, multihop, connector, and SDM-adapter tests.

## Developer campaigns

```bash
uv run --no-sync --package kumo-fm python -m pytest -ra -m dev packages/experimental/kumo-fm/tests
```

These are correctness tests with assertions, not throughput benchmarks. The
randomized campaign uses three fixed seeds, both key types, 24 entities and
1,000 events per populated entity, fanout 50, and batch sizes 1, 7 and 4096.
It compares every selected row and label with the Python oracle and verifies
both wire encodings. Event times are unique within each entity, so recent fanout boundaries are
unambiguous.

The four-hop campaign uses 512 repeated occurrences in each split and reopens
loaders three times. It checks every selected edge, occurrence ownership,
labels, encoding after close, and the base index's SHA-256 inventory. Files are
created under pytest's temporary directory; use `--basetemp=/path/on/local/disk`
to select a disk. Allow several minutes and a few GiB of available memory/disk;
there are no machine-specific latency or RSS thresholds.

Scale the campaigns explicitly (positive integers):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FM_DEV_ENTITIES` | 24 | Entities per randomized index, minimum 4 |
| `FM_DEV_EVENTS` | 1000 | Events per populated entity |
| `FM_DEV_OCCURRENCES` | 512 | Occurrences per split in the four-hop case |
| `FM_DEV_REPEATS` | 3 | Reopened four-hop loaders |

For a quick check of the campaign harness:

```bash
FM_DEV_ENTITIES=4 FM_DEV_EVENTS=15 FM_DEV_OCCURRENCES=8 FM_DEV_REPEATS=1 \
  uv run --no-sync --package kumo-fm python -m pytest -ra -m dev packages/experimental/kumo-fm/tests
```

Use `-k randomized` or `-k four_hop` to isolate a campaign. Pytest node IDs contain
the random seed and key type, so a failure can be replayed without new data.

## Review regressions and remaining gaps

Links refer to [PR #391](https://github.com/NVIDIA/kumo-engine/pull/391).
Existing fixed regressions remain ordinary passing tests. Confirmed outstanding
bugs have narrowly scoped `xfail(strict=True, raises=...)` markers with review
IDs. A fix becomes an XPASS failure: remove its marker and keep its assertions.
Do not turn an unrelated failure into an xfail. Use `--runxfail -k TEST_NAME` to
see the current failure directly.

| Review finding | Executable coverage |
| --- | --- |
| Self-relation endpoint inversion; declared forward/reverse (4674512212, 4687200471, 4687361954) | Existing snapshot API matrix; new native self-relation and PK/FK-left oracle tests |
| `instance_id` PK/feature collision (4674512220, 4687200483, 4687361973) | Existing snapshot API rejection tests |
| Generated `__fm_edge_*` collision (4687200483) | Strict xfail: `test_generated_edge_feature_name_is_reserved` |
| Unprojected int32/float32/decimal columns block FM (4687200488) | Four strict xfails for unprojected columns (also binary); two for lossless projected narrow-scalar widening |
| Population allocated before checking limit (4674512222, 4687200497) | Existing budget-before-validation and bounded-generator tests; new malformed populations |
| Candidate cap changes with batch size (4687200497) | `test_candidate_budget_is_independent_of_batch_size`: large batch rejects; single-row batches strict xfail |
| Unknown budget names (4687200524) | Strict xfail: `test_unknown_budget_key_is_rejected` |
| Temporary anchors retained until close (4687200529) | Two strict xfails (both key types): `test_repeated_load_does_not_retain_anchor_files` |
| Default strategy rejected (4687200537) | Strict xfail: `test_default_strategy_matches_explicit_recent` |
| Occurrence join integrity and binary-only classification (4687200541) | Corrupt-label rejection matrix, task aliases, invalid classes, both wire encodings |
| Extension ABI, platform flags, unaligned reads (4687200564, 4687200570) | Existing pure Python artifact checks; Python CI matrix; new unaligned-buffer tests |
| Wheel dependency/install path and stale lock (4687200550, 4674512227) | Existing locked install, fresh-wheel `pip check`, connector/engine wheel install |

Architecture and editorial requests (adapter removal, internal typing, no-op
phases, public annotations, version stamping, metadata, wording, fixture setup,
change map, and PR splitting) require source/design review. Passing correctness
tests do not resolve those comments. This testing PR does not change runtime
behavior, widen supported projected dtypes, or certify platform compatibility.
