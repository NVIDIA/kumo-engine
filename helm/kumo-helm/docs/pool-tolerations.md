# Pool pod tolerations

`kumoConfig.diskgraph.poolExtraTolerations` adds Kubernetes tolerations to every
runtime pool pod: cache daemons, cache-backed activities, ephemeral activities,
and their startup canaries. It is separate from `nodeTargeting.tolerations`,
which configures Helm-managed workloads and hooks.

```yaml
kumoConfig:
  diskgraph:
    poolExtraTolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

The existing built-ins are always retained:

| Pool | Built-in toleration |
|---|---|
| Cache daemon and cache-backed activities | `dedicated=diskgraph-<global.tenant>:NoSchedule` |
| Ephemeral activities | `dedicated=diskgraph-eph-<global.tenant>:NoSchedule` |

Omitting the setting, `null`, and `[]` all mean **no extras**. Exact duplicates
are ignored. Tolerations do not require matching node taints to exist, and the
built-ins have no scheduling effect on nodes without those taints. This setting
does not add/remove node taints, change required node selectors/affinity, or
configure TopoLVM/NVIDIA DaemonSets. Prefer narrowly scoped tolerations; broad
wildcards can permit access to nodes reserved for other workloads or tenants.

Use Kubernetes fields: `key`, `operator`, `value`, `effect`, and optionally
`tolerationSeconds` for `NoExecute`. Helm validates field types, operators,
effects, and field combinations, including the effective raw configuration
under `kumoConfig.extra.data_compute.k8s_pool.pool_extra_tolerations`.

## Changing tolerations: maintenance operation

Stop new job submissions and finish or cancel all running training/prediction
jobs before upgrading. Ensure cached snapshots have usable dataplane backups
or can be rebuilt, and prepare nodes that satisfy the new configuration.

Changing effective cache tolerations uses the same retirement path as removing
an SSD/GPU capacity tier: the controller deletes affected activity Jobs, daemon
Deployments, and local cache PVCs, and clears their placement bookkeeping.
This is **not a graceful rollout** and does not wait for active jobs to finish.
It does not resize or relabel nodes, and new work may require a cold-cache
restore/rebuild. Ephemeral activities created afterward use the new tolerations.

Startup reconciliation runs before canary validation and new placements.
Retired workers cannot be revived by late heartbeats, and stale daemons are
rejected before provisioning work. Changes invalidate the canary result cache;
surviving canary pods must match the requested spec or be replaced before they
can prove capacity.

Failed cleanup is retried while the controller remains running. The existing
retirement path does not persist cleanup retry state: if the controller restarts
after the daemon pod disappears but before teardown finishes, leftover PVCs or
scaled-down Deployments may need manual cleanup. Verify retirement completed
before resuming submissions, especially after a controller restart.

Unchanged configurations, reordered/duplicate entries, and switching between
omitted, null, and [] do not trigger retirement. Admission-added tolerations
are not treated as changes to Kumo's configuration. Legacy unannotated daemons
remain compatible when no effective extras are configured.

This behavior requires both the updated chart and backend images.
