# Repository development

Run commands from the repository root unless a component says otherwise.
Use the prepared environment's Python interpreter with `dev`:

```bash
.venv/bin/python dev layout-check
.venv/bin/python dev test development/testing/test/scripts
.venv/bin/python dev lint apps/studio/rest
.venv/bin/python dev types
.venv/bin/python dev pre-commit run --all-files
```

`dev` supplies the source paths and relocated shared configuration. Existing
`make` targets delegate to [Makefile](Makefile). Prepare dependencies using the
[setup guide](../docs/development/setup.md); select Studio explicitly with
`uv sync --package kumo --extra dev --extra cpu`.

- `tooling/`: repository maintenance, builds, release and validation programs.
- `testing/`: tests spanning components and their isolated environments.
- `benchmarks/`: system workload generation and measurements.
- `config/`: shared tool configuration and test scheduling data.
- `third-party/`: dependency attribution inputs and controlled build artifacts.

Package-owned checks, examples and benchmarks stay with their packages.
The [migration manifest](repository-moves.json) records the previous paths.
