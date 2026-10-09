{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{/*
Validation. Every mode that creates a resource needs a host, and each provider
mode needs its own TLS input. Failing here beats a half-configured Ingress that
the controller silently never programs.
*/}}
{{- define "kumoEdge.validate" -}}
{{- $mode := toString .Values.mode -}}
{{- $modes := list "none" "alb-acm" "azure-appgw" "openshift" -}}
{{- if not (has $mode $modes) -}}
{{- fail (printf "mode %q must be one of %s" $mode (join ", " $modes)) -}}
{{- end -}}
{{- if ne $mode "none" -}}
{{- if not .Values.host -}}
{{- fail (printf "mode=%s requires host" $mode) -}}
{{- end -}}
{{- if not .Values.target.service -}}
{{- fail "target.service must name the Service to route to" -}}
{{- end -}}
{{- end -}}
{{- if and (eq $mode "alb-acm") (not .Values.alb.certificateArn) -}}
{{- fail "mode=alb-acm requires alb.certificateArn" -}}
{{- end -}}
{{- if and (eq $mode "azure-appgw") (not .Values.appgw.tlsSecretName) -}}
{{- fail "mode=azure-appgw requires appgw.tlsSecretName" -}}
{{- end -}}
{{- end }}

{{- define "kumoEdge.name" -}}
{{- .Release.Name -}}
{{- end }}

{{- define "kumoEdge.labels" -}}
app.kubernetes.io/name: kumo-edge
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- with .Values.extraLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}
