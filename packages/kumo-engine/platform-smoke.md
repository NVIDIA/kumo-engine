# Installed-wheel platform qualification

The manual [wheel build workflow](../../.github/workflows/kumo-engine-wheel.yml)
builds the candidate once and passes those artifacts to the reusable
[platform smoke workflow](../../.github/workflows/kumo-engine-platform-smoke.yml).
Select **Kumo Engine - Build and publish wheel**, **Run workflow**, and the
candidate branch. Manual runs upload artifacts; they do not publish a release.

Qualification covers Python 3.11, 3.12, 3.13 and 3.14 on native Windows x86-64,
Windows through WSL x86-64, and Linux ARM64: twelve independent CPU jobs.
Native Windows jobs provide experimental build and runtime evidence only.
For this release, Windows support is limited to WSL2; native Windows wheels
are retained as CI artifacts but excluded from public release attachments and
publication. A passing native job does not establish supported native Windows
distribution.
WSL installs the same Linux x86-64 wheel as ordinary Linux; it is not a
separate wheel format. GPU availability is not a qualification requirement.
The producer also retains the macOS ARM64 artifact and its existing checks;
this smoke does not add macOS runtime qualification.

Each consumer creates a fresh virtual environment, installs the downloaded
wheel and public CPU Torch dependencies, and runs dependency consistency
checks. It copies only the smoke test and standalone environment auditor
outside the checkout. The auditor rejects checkout imports, records origins
for engine/core/API/PQL/DiskGraph, verifies the packaged ingestion executable,
and records the wheel filename, SHA-256, candidate revision, Python, platform,
and Torch version. Native Windows and Linux ARM64 must run without Faiss;
WSL retains the wheel's existing Linux x86-64 Faiss dependency.

[smoke_platform.py](test/smoke_platform.py) creates a synthetic Parquet table,
performs real snapshot ingestion, and selects CPU through the public
`Trainer(..., accelerator='cpu')` API. Acceptance requires four optimizer
steps with finite CPU gradients, finite training metrics, and all 100 expected
finite predictions. Synchronous DataLoader execution limits Windows coverage
to zero worker processes; Unix worker pools and native Windows GPU execution
are outside this qualification. Subprocess diagnostics are printed only for
this generated fixture, preserving production error privacy.

The WSL job checks the Linux guest kernel. A Windows host shell or ordinary
Linux x86-64 runner cannot substitute for WSL. Consumer jobs install no Rust
compiler and rebuild no native artifacts: missing executables or libraries
must fail the downloaded-wheel check. Logs, environment evidence and pytest
results distinguish packaging failures from training failures.

The [installed-wheel qualification run](https://github.com/NVIDIA/kumo-engine/actions/runs/41101558418)
at `8f6c496f` passed all twelve jobs, each with four CPU optimizer steps and
100 finite predictions. Windows native and Linux ARM64 passed without Faiss.
This establishes runtime evidence for those artifacts, including experimental
native Windows coverage; it does not change the WSL-only Windows release
boundary or publish a package. Later runtime or packaging changes require
qualification of their candidate artifacts.

PR checks remain a later decision after manual qualification is reliable.
This change does not require an ARM GPU runner grant or change publication
approval policy.
