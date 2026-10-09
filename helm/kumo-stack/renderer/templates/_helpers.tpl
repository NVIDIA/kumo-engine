{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{- define "stack.provider" -}}
{{- if eq .Values.platform "eks" }}aws{{ else }}azure{{ end -}}
{{- end -}}

{{- /*
GPU facts used for Karpenter CPU limits and pool-controller labels. Unknown
SKUs fail rather than rendering guessed capacity. g4dn=T4; g6e=L40S.
*/ -}}
{{- define "stack.eksGpuSkus" -}}
g4dn.xlarge: {vcpu: 4, gpus: 1}
g4dn.2xlarge: {vcpu: 8, gpus: 1}
g4dn.4xlarge: {vcpu: 16, gpus: 1}
g4dn.8xlarge: {vcpu: 32, gpus: 1}
g4dn.12xlarge: {vcpu: 48, gpus: 4}
g4dn.16xlarge: {vcpu: 64, gpus: 1}
g6e.xlarge: {vcpu: 4, gpus: 1}
g6e.2xlarge: {vcpu: 8, gpus: 1}
g6e.4xlarge: {vcpu: 16, gpus: 1}
g6e.8xlarge: {vcpu: 32, gpus: 1}
g6e.12xlarge: {vcpu: 48, gpus: 4}
g6e.16xlarge: {vcpu: 64, gpus: 1}
g6e.24xlarge: {vcpu: 96, gpus: 4}
g6e.48xlarge: {vcpu: 192, gpus: 8}
{{- end -}}

{{- define "stack.eksGpuSku" -}}
{{- $skus := fromYaml (include "stack.eksGpuSkus" .) -}}
{{- if not (hasKey $skus .) -}}
{{- fail (printf "unsupported EKS GPU instance type %q; add its vCPU and GPU counts to stack.eksGpuSkus" .) -}}
{{- end -}}
{{- index $skus . | toYaml -}}
{{- end -}}

{{- define "stack.eksVcpu" -}}
{{- (fromYaml (include "stack.eksGpuSku" .)).vcpu -}}
{{- end -}}

{{- define "stack.eksGpuCount" -}}
{{- (fromYaml (include "stack.eksGpuSku" .)).gpus -}}
{{- end -}}

{{- /* The DiskGraph class, and the labels selecting the CPU capacity Kumo runs
on. Both are operator overrides: unset, each resolves to what the platform names
today, so a values file that says nothing about them renders as it always did. */ -}}
{{- define "stack.storageClassName" -}}
{{- if .Values.storage.diskgraph.storageClass.name -}}
{{- .Values.storage.diskgraph.storageClass.name -}}
{{- else if eq .Values.platform "eks" -}}
ebs-gp3-xfs
{{- else -}}
managed-csi-xfs
{{- end -}}
{{- end -}}

{{- define "stack.cpuNodeSelector" -}}
{{- if .Values.cluster.cpuNodeSelector -}}
{{- toYaml .Values.cluster.cpuNodeSelector -}}
{{- else if eq .Values.platform "aro" -}}
{}
{{- else -}}
kumo.ai/node-pool: cpu
{{- end -}}
{{- end -}}

{{- define "stack.validate" -}}
{{- $v := .Values -}}
{{- $provider := index $v.cluster.providers $v.platform -}}
{{- if hasKey $v.cluster "meshNamespaceLabels" -}}
{{- fail "cluster.meshNamespaceLabels is not supported by the default stack; configure advanced mesh transport directly in the leaf charts" -}}
{{- end -}}
{{- if and (eq $v.platform "aks") (ne (len $provider.cachePoolNames) (len $v.gpu.cacheTiers)) -}}
{{- fail "cluster.providers.aks.cachePoolNames must have one name per gpu.cacheTiers entry" -}}
{{- end -}}
{{- if and (eq $v.platform "aro") (ne (len $provider.cacheMachineSets) (len $v.gpu.cacheTiers)) -}}
{{- fail "cluster.providers.aro.cacheMachineSets must have one name per gpu.cacheTiers entry" -}}
{{- end -}}
{{- $tierSkus := dict -}}
{{- range $tier := $v.gpu.cacheTiers -}}
{{- if hasKey $tierSkus $tier.instanceType -}}
{{- fail (printf "gpu.cacheTiers must use distinct instance types; %s is configured twice and they collide as one kumoConfig.diskgraph.nodeTypes key" $tier.instanceType) -}}
{{- end -}}
{{- $_ := set $tierSkus $tier.instanceType true -}}
{{- if eq $v.platform "eks" -}}
{{- $gpus := include "stack.eksGpuCount" $tier.instanceType | int -}}
{{- if ne $gpus (int $tier.gpuCount) -}}
{{- fail (printf "gpu.cacheTiers %q: %s has %d GPUs, not %v" $tier.name $tier.instanceType $gpus $tier.gpuCount) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- /* EKS/AKS require a discriminator their untainted GPU pools never carry.
ARO validates selectors against its adopted MachineSets during preflight. */ -}}
{{- if and (has $v.platform (list "eks" "aks")) $v.cluster.cpuNodeSelector -}}
{{- if ne (get $v.cluster.cpuNodeSelector "kumo.ai/node-pool") "cpu" -}}
{{- fail "cluster.cpuNodeSelector must include kumo.ai/node-pool: cpu on EKS and AKS; it is the one label separating the CPU pool this stack creates from its untainted GPU pools, and any further labels belong alongside it" -}}
{{- end -}}
{{- end -}}
{{- /* Stack-created cloud-disk classes cannot satisfy required
CSIStorageCapacity; that mode requires an adopted capable driver. */ -}}
{{- if and $v.storage.diskgraph.storageClass.create (eq $v.storage.diskgraph.capacityTracking "required") -}}
{{- fail "storage.diskgraph.capacityTracking=required needs a class that publishes CSIStorageCapacity, which a stack-created cloud disk does not; adopt one with storage.diskgraph.storageClass.create=false" -}}
{{- end -}}
{{- /* CPU selectors must not use labels that mark GPU capacity. */ -}}
{{- range $key, $value := $v.cluster.cpuNodeSelector -}}
{{- if or (eq $key "kumo.ai/gpu-node") (hasPrefix "kumo.ai/diskgraph-" $key) -}}
{{- fail (printf "cluster.cpuNodeSelector must not use %s; it marks GPU capacity, which Kumo's CPU workloads must never select" $key) -}}
{{- end -}}
{{- /* These labels identify every node, so they cannot separate CPU capacity. */ -}}
{{- if has $key (list
  "kubernetes.io/os" "kubernetes.io/arch"
  "beta.kubernetes.io/os" "beta.kubernetes.io/arch"
  "topology.kubernetes.io/region" "topology.kubernetes.io/zone"
  "failure-domain.beta.kubernetes.io/region"
  "failure-domain.beta.kubernetes.io/zone") -}}
{{- fail (printf "cluster.cpuNodeSelector must not use %s; every node carries it, GPU ones included, so it tells no capacity apart" $key) -}}
{{- end -}}
{{- end -}}
{{- if and $v.temporal.enabled $v.cloud.postgres.bootstrapTemporal (eq $v.platform "eks") (ne $v.cloud.postgres.admin.mode "aws_iam") -}}
{{- fail "EKS Temporal bootstrap requires cloud.postgres.admin.mode=aws_iam" -}}
{{- end -}}
{{- if and $v.temporal.enabled $v.cloud.postgres.bootstrapTemporal (has $v.platform (list "aks" "aro")) (ne $v.cloud.postgres.admin.mode "azure_entra") -}}
{{- fail "Azure Temporal bootstrap requires cloud.postgres.admin.mode=azure_entra" -}}
{{- end -}}
{{- if and $v.temporal.enabled $v.cloud.postgres.bootstrapTemporal (not $v.cloud.postgres.admin.username) -}}
{{- fail "Temporal bootstrap requires cloud.postgres.admin.username" -}}
{{- end -}}
{{- if and $v.temporal.enabled $v.cloud.postgres.bootstrapTemporal -}}
{{- $identityKey := ternary "eks.amazonaws.com/role-arn" "azure.workload.identity/client-id" (eq $v.platform "eks") -}}
{{- if not (index $v.cloud.postgres.admin.annotations $identityKey) -}}
{{- fail (printf "Temporal bootstrap runs in the cluster and requires cloud.postgres.admin.annotations[%q]" $identityKey) -}}
{{- end -}}
{{- end -}}
{{- range $key := list "global" "image" "exposure" "kumoConfig" "nodeTargeting" "serviceAccounts" "security" "temporal" "postgres" "tenantSecretKey" -}}
{{- if hasKey $v.tenant.extraValues $key -}}
{{- fail (printf "tenant.extraValues.%s is stack-managed and cannot be overridden" $key) -}}
{{- end -}}
{{- end -}}
{{- $workloads := get $v.tenant.extraValues "workloads" | default dict -}}
{{- $poolcontroller := get $workloads "poolcontroller" | default dict -}}
{{- $extraEnv := get $poolcontroller "extraEnv" | default dict -}}
{{- if hasKey $extraEnv "POOL_INIT_CANARY_ENABLED" -}}
{{- fail "tenant.extraValues.workloads.poolcontroller.extraEnv.POOL_INIT_CANARY_ENABLED is stack-managed; use verify gpu-lifecycle" -}}
{{- end -}}
{{- /* Reject overrides that would bypass stack-validated configuration. */ -}}
{{- range $name, $knob := dict
  "POOL_STORAGE_CLASS_NAME" "storage.diskgraph.storageClass.name" -}}
{{- if hasKey $extraEnv $name -}}
{{- fail (printf "tenant.extraValues.workloads.poolcontroller.extraEnv.%s is stack-managed; use %s" $name $knob) -}}
{{- end -}}
{{- end -}}
{{- end -}}
