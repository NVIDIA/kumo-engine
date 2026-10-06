# Pool pod CPU and memory

`kumoConfig.diskgraph.poolResources` sets CPU/memory requests and limits on
runtime-created pool containers, including their child processes. It does not
change long-running application Deployments (`workloads.*.resources`), GPU
allocation, storage sizing, or job concurrency settings.

| Group | Applies to | When omitted |
|---|---|---|
| `daemon` | DiskGraph daemon, including snapshot creation/restore | Requests: 1 CPU, 2 GiB; no CPU/memory limits |
| `activityDefaults` | Defaults for all activity pods, both cache-backed and ephemeral | No CPU/memory requests or limits |
| `trainingOverrides` | Overrides for `TRAIN` activity pods only | Inherits `activityDefaults` |
| `predictionOverrides` | Overrides for `PREDICT` activity pods only | Inherits `activityDefaults` |

Training-table generation and task materialization use `activityDefaults`,
including when part of a training or prediction workflow. Snapshot work inside
the daemon uses `daemon`. These are **per-pod**, not per-job or pool-wide budgets.

For one shared activity budget, configure only `activityDefaults`. Add the
optional type-specific overrides only when training or prediction needs a
different budget. `daemon` remains independent and does not inherit activity
defaults.

Example syntax (illustrative values, not a sizing recommendation):

```yaml
kumoConfig:
  diskgraph:
    poolResources:
      daemon:
        requests: {cpu: "2", memory: "16Gi"}
        limits: {cpu: "4", memory: "16Gi"}
      activityDefaults:
        requests: {cpu: "4", memory: "16Gi"}
        limits: {cpu: "8", memory: "32Gi"}
      trainingOverrides:
        requests: {cpu: "8", memory: "64Gi"}
        limits: {cpu: "16", memory: "128Gi"}
      predictionOverrides:
        limits: {memory: "64Gi"}
```

## Defaults and validation

- Omit `poolResources` or set `{}` to retain current behavior. No new limits
  are enabled by default.
- Overrides merge **per resource and per request/limit**. For example,
  `predictionOverrides.limits.memory` replaces only the default memory limit,
  retaining CPU and request fields from `activityDefaults`. Overrides win even
  when smaller, as long as the effective request does not exceed its limit.
  Missing daemon fields retain its legacy defaults. Empty maps do not clear
  inherited fields; remove the setting from `activityDefaults` to stop
  inheriting it.
- Use positive Kubernetes quantities, e.g. `1000m`, `4`, `16Gi`; CPU precision
  must be at least 1 millicore. Numbers are also accepted and normalized to
  strings. Quote quantities (especially large integers) to avoid rounding
  during Helm's YAML loading. Only `cpu` and `memory` are configurable here.
- Helm rejects explicitly configured requests below **1 CPU** or **16 GiB**
  (`16Gi`), for any group. These are configuration guardrails, not a guarantee
  that every dataset fits. Omitted fields stay omitted; the daemon's implicit
  legacy 1 CPU / 2 GiB requests are unchanged.
- Helm also validates shape, positive quantities, the supported numeric range,
  CPU precision, and **effective** request <= limit (including inherited
  values). These checks include overrides supplied through `kumoConfig.extra`.
  There is no workload-sizing floor or cap on limits: `1Ti` is allowed. Choose
  limits for the workload and node capacity. The new request-floor checks are
  Helm-only; application startup validation is unchanged.
- If an activity specifies only a limit, Kumo leaves the request absent.
  Kubernetes normally defaults the request to that limit; namespace
  LimitRanges/admission policies can also supply or constrain values. See
  [Kubernetes request/limit semantics](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#requests-and-limits).

## Capacity and rollout

Requests reserve scheduling capacity; CPU limits can throttle and memory limits
can cause a container OOM kill. Limits do not make undersized jobs succeed or
guarantee protection from node-wide memory pressure. Include child processes,
shared memory and file-cache usage when sizing memory.

For each cache node, budget its daemon **plus all concurrent activity pods**
and system overhead against node allocatable capacity. Activities must colocate
with their cache daemon; spare capacity on another node does not help that
placement. Too-small requests still allow overcommit. Too-large requests leave
pods Pending. On clusters without an autoscaler, provision sufficient capacity
and set job concurrency accordingly; this feature does not add CPU/memory-aware
pool routing, eviction, or autoscaling.

Use both a chart and backend image containing this feature. The chart passes
these values through `data_compute.k8s_pool.pool_resources` in
`DEPLOY_ENV_CONFIG_JSON`; an older controller image will not enforce them.
The runtime group names are `daemon`, `activity_defaults`, `training_overrides`,
and `prediction_overrides` (snake_case, unlike the Helm values).
Restart/roll out the pool controller after changing its configuration. The
settings apply when it creates new daemon Deployments and activity Jobs; they
do not resize existing pods or existing daemon Deployment templates. Plan any
daemon replacement for a maintenance window with no active jobs. Do not delete
cache PVCs just to apply CPU/memory settings.
