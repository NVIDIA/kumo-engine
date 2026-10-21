# Synthetic commerce generator

This is the commerce portion of the original scale benchmark generator with an invented placeholder profile.
See the enclosing benchmark README for commands and measurement boundaries.
The source algorithm uses NumPy, PyArrow and SciPy, all present in the Kumo runtime dependency graph at engine revision `392c25ed`.
No dependency pins or installers are supplied.
`config.json` selects the placeholder profile; `smoke-config.json` uses the same profile and reduces market size for a small local check.

`verify.py` verifies generated files against their manifest using SHA-256, declared Arrow schemas and row counts.
It does not implement the original DuckDB validator's distribution, temporal or relationship checks.
The manifest records the actual output and library versions; a rung name is an approximate size target.
Do not join generated `_ground_truth` hidden state into model features.
