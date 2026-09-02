# DiskGraph diagnostic protocol

DiskGraph publishes one bounded `.diagnostic.json` record when a subprocess fails terminally. The record carries operational detail for Kumo logs and support bundles, plus an optional `public` projection for an explicitly approved user-facing message. Raw subprocess output is a separate, untrusted channel and is never copied into the record.

## Logging policy

Returned errors are visible in the operational projection by default. This preserves useful first-party failures without requiring an allowlist or module registry.

Code that crosses a boundary known to return provider-controlled text, customer values, credentials, or sensitive paths must wrap the error in `RestrictedDiagnostic`. The wrapper retains the original source for local control flow, while the record contains only its fixed safe summary and classification. Once a restricted marker is present, neither the marked source nor outer context is rendered.

Log severity is not a confidentiality control. Credentials and secrets must never be emitted at any level; raw subprocess output remains restricted even when Kumo makes it available at `DEBUG`.

When adding or changing an error boundary:

- Keep the error ordinary unless there is a concrete sensitive-data path.
- Apply `RestrictedDiagnostic` at the narrowest call site that understands the boundary.
- Choose a stable operation, category, code, retry decision, and data-independent safe summary.
- Test with a distinctive sensitive canary and assert that it is absent from the serialized diagnostic.

## Transport invariants

The worker writes `.diagnostic.worker.json` atomically. The supervisor promotes only the terminal attempt to `.diagnostic.json`, removes stale records on retry or success, and creates a metadata-only fallback when a worker cannot return an error.

Version 1 has an exact schema and a 32 KiB encoded limit. Text, cause chains, and frames are bounded; control characters are normalized; and restricted fingerprints never depend on raw sensitive detail.

Only the optional `public` projection may supply REST, Temporal, or UI text. Operational summaries and causes are for internal logging only. A missing or invalid public projection must produce a generic user-facing failure.

Consumers must read only the exact filename without following symlinks, enforce the size and schema limits, and log only validated fields. Malformed, stale, oversized, or missing records must not change the underlying retry classification.
