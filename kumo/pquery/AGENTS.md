# Query orchestration agent guidance

Read [architecture](ARCHITECTURE.md), [development constraints](DEVELOPMENT.md),
and [testing](TESTING.md) for this service module.

- Parsing/semantic validation and AST definitions live in `packages/kumo-pql`
  and `packages/kumo-api`. Coordinate their consumers within this repository.
- Compiler semantics live in `packages/diskgraph`; the Python compiler is `diskgraph-pybind/python/diskgraph/compiler.py` within that workspace. Follow the [DiskGraph package guidance](../../packages/diskgraph/README.md#python-wheel-builds); update affected service/core adapters here.
- Preserve temporal boundaries, split semantics, generated table schemas,
  weight handling, cached statistics, and serialized query compatibility.
- Use pre/post generation validators for data-dependent checks; do AST-level
  checks in the parser package. Validate at the earliest applicable stage.
- Shared training execution lives in `packages/kumo-core`; service training
  and task adapters remain in `kumo/train` and `kumo/task`.
- Select focused tests from the testing guide. Grammar changes also affect
  UI generation and autocomplete; follow the root generated-artifact guide.
