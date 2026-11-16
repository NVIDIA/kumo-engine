{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{- define "kumo-platform-aro.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: kumo-platform
kumo.ai/platform: aro
{{- end }}

{{- define "kumo-platform-aro.validate" -}}
{{- if and (eq .Values.operators.mode "existing") (or .Values.nfd.create .Values.clusterPolicy.create) -}}
{{- fail "existing operator mode cannot create a NodeFeatureDiscovery or ClusterPolicy" -}}
{{- end -}}
{{- if .Values.machineAutoscalers.enabled -}}
{{- range $name := concat .Values.machineAutoscalers.cacheMachineSets (list .Values.machineAutoscalers.ephemeralMachineSet) -}}
{{- if or (not $name) (contains "REPLACE_" (upper $name)) -}}
{{- fail "machineAutoscalers require existing cacheMachineSets and an ephemeralMachineSet name" -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}
