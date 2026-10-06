# Licensing source and packages

Kumo-owned source covered by `scripts/license_header_policy.json` must use the NVIDIA SPDX header defined there. Apache-2.0 is the default; `packages/kumo-api/**` remains MIT-licensed. Generators must emit the applicable header. Do not add NVIDIA headers to unmodified third-party source; instead, record its scope and provenance in the policy.

Rust headers belong before inner crate attributes and module documentation. Shebangs, Docker parser directives, and required `"use client"` directives may precede a header.

Policy exclusions and derived-source declarations must identify existing files that need the exception. The checker rejects stale, overlapping, or repository-stamped exclusions.

Run the source and Python package checks locally with:

```bash
python scripts/check_license_headers.py
python scripts/check_python_license_metadata.py
pytest -q --confcutdir=test/scripts test/scripts/test_check_license_headers.py test/scripts/test_python_license_metadata.py
```
