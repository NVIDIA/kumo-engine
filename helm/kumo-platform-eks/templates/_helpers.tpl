{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{- define "kumo-platform-eks.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: kumo-platform
kumo.ai/platform: eks
{{- end }}

{{- define "kumo-platform-eks.validate" -}}
{{- if or (not .Values.clusterName) (contains "REPLACE_" (upper .Values.clusterName)) -}}
{{- fail "clusterName must name the target EKS cluster" -}}
{{- end -}}
{{- if or (not .Values.karpenter.nodeRole) (contains "REPLACE_" (upper .Values.karpenter.nodeRole)) -}}
{{- fail "karpenter.nodeRole must name the existing Karpenter EC2 node role" -}}
{{- end -}}
{{- $skus := fromYaml (include "kumo-platform-eks.gpuSkus" .) -}}
{{- range $pool := concat .Values.nodePools.cacheTiers (list .Values.nodePools.ephemeral) -}}
{{- range $instanceType := $pool.instanceTypes -}}
{{- if not (hasKey $skus $instanceType) -}}
{{- fail (printf "every GPU pool must use a supported NVIDIA instance type; %q is not one of %s" $instanceType (keys $skus | sortAlpha | join ", ")) -}}
{{- end -}}
{{- if and (hasKey $pool "gpuCount") (ne (int (index $skus $instanceType).gpus) (int $pool.gpuCount)) -}}
{{- fail (printf "cache tier %q declares gpuCount %v, but %s has %v GPUs" $pool.name $pool.gpuCount $instanceType (index $skus $instanceType).gpus) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{- /*
GPU SKUs this chart will build NodePools from. gpus backs the
kumo.ai/diskgraph-gpu-count label the Kumo pool controller schedules on, so an
unlisted SKU fails instead of producing a label no node can honour.
g4dn is NVIDIA T4; g6e is NVIDIA L40S.
*/ -}}
{{- define "kumo-platform-eks.gpuSkus" -}}
g4dn.xlarge: {gpus: 1}
g4dn.2xlarge: {gpus: 1}
g4dn.4xlarge: {gpus: 1}
g4dn.8xlarge: {gpus: 1}
g4dn.12xlarge: {gpus: 4}
g4dn.16xlarge: {gpus: 1}
g6e.xlarge: {gpus: 1}
g6e.2xlarge: {gpus: 1}
g6e.4xlarge: {gpus: 1}
g6e.8xlarge: {gpus: 1}
g6e.12xlarge: {gpus: 4}
g6e.16xlarge: {gpus: 1}
g6e.24xlarge: {gpus: 4}
g6e.48xlarge: {gpus: 8}
{{- end }}
