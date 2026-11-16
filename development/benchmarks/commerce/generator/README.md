# Synthetic commerce generator

This is the commerce portion of the original scale benchmark generator with an invented placeholder profile.
See the enclosing benchmark README for commands and measurement boundaries.
The source algorithm uses NumPy, PyArrow and SciPy, all present in the Kumo runtime dependency graph at engine revision `392c25ed`.
`requirements.txt` pins the published generator versions: NumPy 2.2.6, SciPy 1.15.3 and PyArrow 18.1.0.
Install them in a separate generator environment as shown in the enclosing benchmark README.
Commerce uses `degree_drop_sub1=True` and `tail_stretch_max=4.0`.
Both configurations write Snappy Parquet directly; no transcode step is needed.
Exact published bytes also require the original profile, configuration and file layout, which are not supplied.
`config.json` selects the placeholder profile; `smoke-config.json` uses the same profile and reduces market size for a small local check.

`verify.py` verifies generated files against their manifest using SHA-256, declared Arrow schemas and row counts.
It does not implement the original DuckDB validator's distribution, temporal or relationship checks.
The manifest records the actual output and library versions; a rung name is an approximate size target.
Do not join generated `_ground_truth` hidden state into model features.
