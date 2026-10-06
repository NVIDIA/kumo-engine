# statedb package layout

`statedb` contains persisted data models and storage helpers. Keep modules in the package that owns their domain instead of using `statedb.config` as a catch-all.

## Packages

- `statedb.config`: Common persisted definitions shared across REST, ML, and other runtime code. These modules should not import REST-specific code.
- `statedb.rest`: REST metadata storage definitions and REST-owned persisted shapes. Modules here may depend on common `statedb.config` definitions.
- `statedb.ml`: ML-specific persisted definitions, including batch-prediction jobs.
- `statedb.db`: Database configuration, schema, bootstrap, repositories, and services.
- `statedb.utils`: Shared error types and other small statedb utilities.

## Dependency Rules

- Common definitions in `statedb.config` must remain REST-agnostic and ML-agnostic.
- REST-specific definitions should live in `statedb.rest`, not `statedb.config`.
- ML-specific definitions should live in `statedb.ml`.
- Shared functionality belongs in `statedb.utils`; shared type definitions belong in `statedb.config`.
- Avoid compatibility forwarding modules when moving unshipped code. Update imports to the owning package directly.
