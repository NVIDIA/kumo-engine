# Installed-wheel platform qualification

The [wheel workflow](../../.github/workflows/kumo-engine-wheel.yml) builds the
candidate once and gates release attachment on the reusable
[platform smoke](../../.github/workflows/kumo-engine-platform-smoke.yml) and
[desktop reload](../../.github/workflows/kumo-engine-desktop-reload.yml) checks.
For a rehearsal, select **Kumo Engine - Build and publish wheel**, **Run
workflow**, and the candidate branch. Manual runs save artifacts without
attaching or publishing a release.

Python 3.11–3.14 consumers install the exact candidate wheel in fresh virtual
environments outside the checkout. The auditor records the commit, wheel
SHA-256, module origins, packaged ingestion executable, platform and dependency
versions. Native Windows requires importable CPU Faiss; Linux ARM64 runs
without Faiss, and WSL uses the Linux x86-64 wheel and its Faiss dependency.

The twelve [synthetic CPU smoke](test/smoke_platform.py) jobs cover native
Windows, WSL and Linux ARM64. They require four optimizer steps, finite metrics,
100 finite predictions, and matching predictions after bundle reload. These
jobs use synchronous DataLoader execution.

The shared [desktop harness](test/desktop_reload_qualification.py) additionally
gates every build on four native Windows CPU jobs. Binary and static-link tasks
train, export models and snapshots, delete the producer's temporary storage,
and predict in fresh consumers. Retrieval must perform CPU Faiss searches.
Windows installs the bare wheel and exercises automatic CPU selection.
Native CUDA and restricted accounts are outside this qualification. Evidence is
saved in `engine-wheel-reload-windows-native-cpu-*` artifacts.

See the [installation requirements](README.md#installation) for the long-path
and Visual C++ runtime prerequisites.

Consumer jobs rebuild no native artifacts. WSL checks the Linux guest kernel;
an ordinary Linux runner cannot substitute for WSL. The existing Linux CUDA
and macOS build checks remain separate. See the
[release guide](../../docs/development/releasing-kumo-engine.md) for the broader
manual desktop matrix and publication approval procedure.
