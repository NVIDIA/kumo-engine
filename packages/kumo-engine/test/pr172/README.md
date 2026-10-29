# Small PR172 fixtures

The GPU smoke action downloads the versioned S3 bundle recorded in
`inputs.json`. Its SHA-256 pins the complete compressed archive. The action
checks that hash before extraction and uses the repository's existing AWS role.
No private GitHub repository or fixture-access token is required.

The bundle retains the same 29 files from the approved generate-once inputs:
23 Parquet table files, three manifests, and three validation stamps. It adds
`graphs.json` with public table, column, key, time, type, and edge metadata.
The tables contain 556,127 adtech rows, 1,442,968 delivery rows, and 74,416 commerce
rows across their respective graph families. The original 29 files remain
byte-identical. No generator source, hidden answers, or model outputs are included.

The extracted root contains `adtech/pr`, `delivery/pr`, `commerce/pr`, and
`graphs.json`. The standalone runner reads those paths with public Engine APIs.
It uses native static-link splitting for smoke coverage. The previous quality
control payload and source projection are not part of this check.

A fixture update requires a new versioned S3 object and an updated manifest
hash. Keep the frozen reports from previous runs unchanged. The
[Engine PR guide](../../../../docs/development/engine-pr-ci.md) describes
normal runs, numeric qualification, and local feedback.
