# Python Third-Party Dependencies & Lockfiles

This directory contains all third-party Python dependencies for our monorepo and their corresponding Pants lockfiles. This document explains how to manage dependencies, regenerate lockfiles, and deal with platform-specific builds.

## 1. Lockfiles Basics

Lockfiles ensure deterministic builds by pinning exact versions and SHA256 hashes of all Python dependencies. Benefits:

- Reproducible builds across developers and CI.
- One-time dependency resolution during lockfile generation.
- `pex_binary` packages use only pinned artifacts.

We currently use a single default resolve (`python-default`), but some dependencies, like `torch`, vary by GPU vs CPU (`CUDA` environment variable), requiring parametrized lockfiles.

---

## 2. Regenerating Lockfiles

Whenever you modify a dependency in `3rdparty/BUILD` or make changes to `pants.toml`, the lockfiles may become out of sync.

Our GitHub workflow handles lockfile regeneration automatically:

1. When you create a PR that touches `3rdparty/BUILD` or `pants.toml`, the `pants_check` job will run first.
2. If `pants_check` fails due to a lockfile mismatch, the workflow automatically:
   - Detects the lockfile error (other `pants_check` failures won't trigger regeneration).
   - Posts a comment on your PR letting you know regeneration is in progress.
   - Runs `pants --python-enable-resolves=false check ::` to catch non-lockfile errors before regeneration.
   - Generates updated lockfiles for CPU (`CUDA=cpu`) and CUDA 13 GPU development (`CUDA=cu130`), then creates the update PR; this job does not recheck the generated locks.
   - Opens a PR titled **"[ACTION REQUIRED] Merge lockfile updates into \<branch\>"** targeting your dev branch.
   - Posts a follow-up comment on your PR with a link to the lockfile PR and next steps.
   - If regeneration fails, posts a comment with a link to the job logs.
3. Review and merge the generated lockfile PR into your dev branch (usually just a rubber-stamp).
4. Pull changes locally:
```bash
git pull origin <dev_branch>
```

### Manual / local regeneration

The auto-regeneration above only fires when `pants_check` **fails** on a lockfile mismatch. Pants validates that the requirements each target consumes are a **subset** of what the lockfile was generated with, so a change that only **removes** a requirement (or loosens a constraint) leaves every lockfile still valid — CI stays green and no lockfile PR is opened. In that case the stale pins linger until you regenerate by hand.

Each `generate-lockfiles` run regenerates exactly one lockfile, selected by platform, CPU architecture, and `CUDA` (see `.pants.bootstrap`). Regenerate the CPU and native-architecture CUDA locks on a Linux host:

```bash
for cuda in cpu cu130; do
  rm -rf ~/.cache/pants/
  CUDA=$cuda pants --pants-workdir=/tmp/pants-workdir-$cuda -lwarn generate-lockfiles
done
```

On x86_64 this writes `3rdparty/default_cpu.lock` and `default_cu130.lock`. On ARM64 the CUDA invocation writes `default_cu130_aarch64.lock`; the x86_64 CUDA lock remains unchanged. The ARM resolve uses `default_cu130_aarch64.constraints.txt`, derived from the x86_64 CUDA lock with `faiss-gpu` omitted, so both architectures retain the same package versions wherever both support them.

---

## 3. Platform-specific lockfiles (macOS)

`.pants.bootstrap` selects the active lockfile from the host platform and the `CUDA` env var:

| Lockfile | Host | `CUDA` |
|---|---|---|
| `default_cpu.lock` | Linux | `cpu` |
| `default_cu130.lock` | Linux x86_64 | `cu130` (GPU) |
| `default_cu130_aarch64.lock` | Linux ARM64 | `cu130` (GPU; FAISS installed out of band) |
| `default_cpu_macos.lock` | macOS | `cpu` (forced) |

The standard lock workflow regenerates the CPU and x86_64 CUDA locks. The ARM image workflow verifies the constraints snapshot and checks the ARM64 CUDA lock on its native runner. **`default_cpu_macos.lock` is never touched by CI** — it exists purely so Pants can resolve on Apple-silicon dev machines. Regenerate it out-of-band on a Mac:

```bash
pants generate-lockfiles          # Darwin forces CUDA=cpu → default_cpu_macos.lock
```

When a `3rdparty/BUILD` change lands, regenerate the macOS lock in the same window as the Linux locks so all supported lockfiles share one upstream resolve snapshot.

---

## 4. Third-party notices

The root `THIRD-PARTY.txt` is generated from `3rdparty/notices/inventory.json` and `3rdparty/notices/policy.json`. The inventory maps components copied into Kumo source or artifacts to the artifact that carries them. Dependencies installed as separate Python distributions normally retain their own legal files instead of being duplicated in the generated notice. A dependency is included when its installed distribution omits required upstream legal text.

Validate the inventory and committed notice:

```bash
python scripts/gen_third_party.py --check THIRD-PARTY.txt
python -m unittest test.scripts.test_gen_third_party
```

Validate the DiskGraph artifact manifest after populating Cargo's source cache:

```bash
cargo fetch --locked --manifest-path packages/diskgraph/Cargo.toml
python packages/diskgraph/scripts/generate_artifact_manifest.py --repo-root packages/diskgraph --check packages/diskgraph/third_party/artifact-manifest.json
```

After reviewing an inventory or policy change, regenerate the notice:

```bash
python scripts/gen_third_party.py --output THIRD-PARTY.txt
```

The fat-wheel builder validates collected wheel identities, native members, and legal metadata before generating its artifact-specific notice. Its GloVe input is separately hash-pinned. Set `THIRD_PARTY_NOTICE` only to supply a reviewed alternative.
