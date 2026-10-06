# Multi-devserver requirements and design contract

## Goal

Run multiple independent Kumo devservers on one machine without collisions in local state, DiskGraph state, Temporal execution, or CPU/RAM allocation. Normal multi-devserver launches are resource-fenced; unfenced launches require an explicit opt-out.

The coordinator owns child-devserver identity, dynamic port allocation,
lifecycle, and resource planning. The existing one-devserver launcher is an
internal primitive.

## Public interface

`make devserver` is the only public entry point. It has two deliberately
separate configuration layers:

1. **Devserver configuration** selects what each server runs: the existing
   `variant`/`DEPLOY_ENV_CONFIG_FILE` and shared settings such as
   `DISABLE_AUTH`, `PHASE1B_DEVSERVER_MINIMAL`, and
   `DEVSERVER_MONITORING`. The coordinator derives a non-overridable,
   child-specific isolation overlay from that one selected configuration.
2. **`CGROUP_CONFIG`** allocates only CPU and RAM to those children. It does
   not select a variant, set paths, change Temporal, or carry arbitrary
   devserver settings.

The first version intentionally has no user-authored per-child devserver
configuration format. Differing variants/runtime flags are a future explicit
`INSTANCE_CONFIGS` feature; it must be separately validated and cannot
override generated identity, ports, roots, Temporal settings, or cgroup
limits.

```bash
# N identically sized, resource-fenced instances.
make devserver NUMS=2 CGROUP_CONFIG=auto DETACHED=1

# Exact unequal CPU/RAM allocations. Keys are stable child IDs.
make devserver CGROUP_CONFIG='{"train-a":{"cpus":32,"memory_gib":128},"train-b":{"cpus":32,"memory_gib":128}}' DETACHED=1

# Explicit opt-out of CPU/RAM fencing.
make devserver NUMS=2 UNFENCED=1 DETACHED=1
```

`make devserver` without `CGROUP_CONFIG` or `UNFENCED=1` fails with usage.
`CGROUP_CONFIG=auto` and `UNFENCED=1` require a positive `NUMS`.
`UNFENCED=1` is mutually exclusive with `CGROUP_CONFIG` and prints a
resource-contention warning. `NUMS > 1` requires detached operation.

For a JSON `CGROUP_CONFIG`, its object keys are the child devserver IDs and
therefore define the devserver count. `NUMS` may be omitted or must match that
count; an optional `DEVSERVER_IDS` must have the same normalized IDs. For
`auto` and `UNFENCED`, an optional `DEVSERVER_IDS=a,b` supplies stable readable
IDs and otherwise the coordinator creates them. `NUMS` has no implicit default
of one.

`BASE_PORTS` is independent of both configuration layers. It may be omitted
or `auto`, or it must be a comma-separated list exactly as long as the
devserver count. `DEVSERVER_IDS` and `BASE_PORTS` are both validated before
any child starts; duplicate child IDs (after normalization) and duplicate
ports are rejected.

## Identity and ports

Ports are transport addresses, not persistent identity. Each child devserver has a
stable child ID. `DEVSERVER_IDS=a,b` supplies readable IDs; otherwise the
coordinator generates collision-free IDs and records them in its manifest.
Child IDs are used in tmux sessions, systemd units, roots, labels, Temporal
queues, and command arguments, so their normalized form is ASCII lowercase
and must match `^[a-z][a-z0-9-]{0,47}$`. Input is Unicode-NFKC-normalized and
case-folded before validation; collisions after normalization are rejected.
The coordinator logs the normalized ID and never interpolates an unvalidated
ID into a path, unit name, label, or shell command.

`BASE_PORTS` replaces public `BASE_PORT`: an explicit comma-separated list supplies one base port per child; omitted or `auto` dynamically selects one free contiguous port block per child; an explicit list must exactly match the devserver count.

The coordinator checks every TCP port in the complete nonlegacy block, holds
an allocation lock, and records a reservation in its manifest before launch.
It validates block/range overflow, duplicate offsets, and the IPv4/IPv6 bind
behavior expected for every service. A child repeats preflight, then the
coordinator verifies every expected listener belongs to that child after
launch. A lock and preflight cannot eliminate a third-party bind race:
explicit blocks fail loudly, while an auto block may be retried only after
stopping and proving ownership of that failed child. The coordinator never
stops, cleans up, or retries over an unknown conflicting process.

Before launch, print child IDs, final Temporal identities, port blocks and
URLs, roots, resource limits, and fence status. This is a launch plan, not an
additional confirmation protocol; `make` continues unless `PLAN_ONLY=1` is
requested.

## CPU and memory

`CGROUP_CONFIG=auto` means CPU and memory only. It does not choose a variant, invent paths, allocate GPUs, or throttle disk I/O.

Auto allocation uses effective parent-cgroup capacity: affinity/cpuset and finite ancestor `cpu.max` bound CPU; finite ancestor `memory.max` otherwise physical memory bounds RAM. It reserves documented host/shared-service headroom, subtracts live coordinator-managed allocations, divides the remainder equally, rounds down conservatively, and leaves rounding remainder unused. It uses capacity, never transient `MemAvailable` or page cache, and fails instead of producing too-small limits.

Each fenced child re-execs before any fork through `systemd-run --user --scope` with `CPUQuota` and `MemoryMax`; it must not write directly under `/sys/fs/cgroup`. DiskGraph #375 and Kumo #29932 are required dependencies, so ingest sees the inherited boundary and Kumo preserves its resolved thread count.

GPU and I/O isolation are not supplied by `CGROUP_CONFIG`. The coordinator
must state that fact, but multi-devservers are allowed to share the machine's
GPU(s), including the common single-GPU host. It must neither set
`CUDA_VISIBLE_DEVICES` nor reject GPU jobs merely because another child
devserver exists. Concurrent GPU training can contend, slow down, or OOM; that
is an explicit shared-resource tradeoff reported in the launch plan, not an
isolation guarantee. A future `GPU_CONFIG` may offer verified disjoint device
assignments, but it is not required for normal multi-devserver operation. SSD
paths are isolated but SSD bandwidth remains shared.

## Storage isolation

One raw devserver config is selected through the existing `variant` mechanism.
Before deploy-config resolution, each child receives an automatic overlay:
child-specific `KUMO_HOME`, DB/data-persist, temporary, cache, and model
roots; a child-specific `DEVSERVER_INSTANCE_SUFFIX` appended to mutable
data-plane, artifact, DiskGraph `data_root`, and `local_ingest_root` paths;
and child-specific tmux server/session, Docker label, ports, monitoring home,
and AutoRPC service prefix.

The coordinator creates every private temporary/cache/model directory with
owner-only permissions before setting `TMPDIR`, XDG, HF, or Torch variables.
It canonicalizes local paths, verifies they remain inside the child-owned
manifest sandbox, and rejects symlink traversal or root escape. A config
schema must classify each remote artifact/data-plane URI as read-only input or
writable output. Read-only inputs may be shared; known writable outputs are
child-suffixed; an unknown or intentionally shared writable remote destination
is rejected rather than silently rewritten.

Read-only sources and connector credentials may remain shared. Only
schema-classified writable data-plane and artifact destinations are
suffix-scoped; no remote destination is assumed mutable merely from its URI.

## Shared Temporal control plane

An operator-configured Temporal namespace (or one local Temporal service) is shared by default. Sharing is correct only when workflow IDs, task queues, AutoRPC queues, and workflow/activity descendants are child-devserver-scoped.

The parent raw machine identity is preserved: `host-a + train-a ->
host-a-train-a`. If an operator supplies `INSTANCE_ID`, it is the
raw parent identity and is preserved verbatim as the prefix; the coordinator
only appends `-<normalized-child-id>`. `DeployEnvConfig` remains the sole
place that prepends the username, exactly once. The plan must print the raw
parent identity and the resolved final child identity.

Thus identical submitted jobs from `train-a` and `train-b` receive distinct workflow IDs and task queues and both execute in the shared namespace. Temporal UI search uses this full final `instance_id`.

Required invariants: a client and server for one child resolve the same AutoRPC queue; two children resolve different AutoRPC queues; the singlehost daemon resolves `diskgraph-daemon-tq-<child-suffix>` on the wire; and top-level workflow IDs, child workflow IDs, and activity queues retain the full child suffix.

Separate Temporal endpoints/namespaces are an advanced future override for control-plane blast-radius isolation or incompatible versions, not an automatic cgroup feature.

## Atomic lifecycle and recovery

The coordinator holds a global launch/lifecycle lock and an exclusive lock per
normalized child ID. It atomically writes and fsyncs a manifest with phases
`planned`, `launching`, `running`, and `stopping`; it contains child and final
Temporal identities, ports, roots, cgroup scope, Docker label, selected raw
config, expected listener set, owner tokens, and start time. It rejects
duplicate IDs, live owned scopes/sessions, occupied port blocks, and root
ownership conflicts.

On a stale manifest, recovery first proves whether every recorded resource is
still manifest-owned using owner tokens, cgroup/session/container metadata,
and listener PID ancestry. It may only resume, stop, or delete proven-owned
resources. An ambiguous resource leaves the manifest intact and fails with
manual-remediation information; it is never killed or cleaned up by name,
port, or path alone.

Plain stop never deletes state. Reset or purge must terminate only that child's open Temporal workflows; poll until zero opens are confirmed for its full identity; stop its systemd scope, tmux server, and labeled containers; verify no owned processes/listeners remain; then delete only manifest-listed local roots. Failure to confirm zero opens retains local state and fails loudly. Completed Temporal history may remain.

## Required validation

Unit coverage must validate identity derivation and normalization, JSON and
cardinality validation, dynamic port plans/range bounds/bind-family behavior,
resource bounds, cgroup command construction, private-root anti-escape
checks, unchanged CUDA visibility, atomic manifest transitions/recovery, and
Temporal client/server queue parity.

Acceptance E2E starts two nonminimal fenced singlehost devservers against the same Temporal endpoint/namespace. It submits the same small graph refresh/ingest job concurrently through both REST APIs, verifies distinct workflow IDs/task queues/pool workers/roots and successful completion, then runs short training with normal shared-GPU behavior. Logs must show the expected cgroup and resolved DiskGraph ingest thread count. Cleanup must prove scoped Temporal termination and zero open workflows before deletion.
