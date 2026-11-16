# Polars timing reference

`polars.PolarsEngine` is a test-owned Kumo engine for local Parquet tables. It
uses Polars joins, filters and aggregates with the shared Kumo contract,
materialization and batching API. It is independent of AML and model families.
The nightly uses it only for CPU loader timing; it is outside the installed
package and requires no Arena checkout.

Source: Arena revision `220aa187921ddb1d846985bec8f8edff1d3581c9`,
`engine-comparison/engines/`. The port includes its shared source/anchor/type/
graph helpers and `polars/{engine,store,temporal}.py`. Standalone AML loaders,
DuckDB, comparison orchestration and Arena package initializers were omitted.
Changes are local initializers, the engine name, formatting and import ordering.
The reference's execution semantics and exceptions are preserved.

Install the test-only dependency `polars==2.1.0`. Set `POLARS_MAX_THREADS` before
starting Python, matching the selected engine thread count; an already initialized
pool with a different size is rejected. Nightly workers use four threads on both
engines and record the environment and dependency versions.

The AML adapter uses ephemeral resident Polars tables, so it measures
`reference.load` and has no DiskGraph index build/reopen. File caches are
uncontrolled in both loader-timing campaigns. This reference supports its
explicit contract/type subset; unsupported combinations fail rather than changing
engines or silently falling back. See the [benchmark interface](../docs/benchmarking.md#polars-loader-timing-comparison).
