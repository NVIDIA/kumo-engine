{{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}
{{/* Validate the effective list, including values supplied via kumoConfig.extra. */}}
{{- define "kumo.validatePoolExtraTolerations" -}}
{{- $path := "poolExtraTolerations (effective data_compute.k8s_pool.pool_extra_tolerations)" -}}
{{- if not (kindIs "invalid" .) -}}
{{- if not (kindIs "slice" .) -}}
{{- fail (printf "%s must be a list or null" $path) -}}
{{- end -}}
{{- range $i, $t := . -}}
{{- $item := printf "%s[%d]" $path $i -}}
{{- if not (kindIs "map" $t) -}}
{{- fail (printf "%s must be a Kubernetes toleration object" $item) -}}
{{- end -}}
{{- range $field, $value := $t -}}
{{- if not (has $field (list "key" "operator" "value" "effect" "tolerationSeconds")) -}}
{{- fail (printf "%s has unsupported field %q" $item $field) -}}
{{- end -}}
{{- if and (ne $field "tolerationSeconds") (not (kindIs "string" $value)) -}}
{{- fail (printf "%s.%s must be a string" $item $field) -}}
{{- end -}}
{{- end -}}
{{- $operator := $t.operator | default "Equal" -}}
{{- $effect := $t.effect | default "" -}}
{{- if not (has $operator (list "Equal" "Exists")) -}}
{{- fail (printf "%s.operator must be Equal or Exists" $item) -}}
{{- end -}}
{{- if not (has $effect (list "" "NoSchedule" "PreferNoSchedule" "NoExecute")) -}}
{{- fail (printf "%s.effect must be NoSchedule, PreferNoSchedule, NoExecute, or empty" $item) -}}
{{- end -}}
{{- if and (not $t.key) (ne $operator "Exists") -}}
{{- fail (printf "%s requires operator Exists when key is empty" $item) -}}
{{- end -}}
{{- if and (eq $operator "Exists") $t.value -}}
{{- fail (printf "%s must not set value with operator Exists" $item) -}}
{{- end -}}
{{- if and (hasKey $t "tolerationSeconds") (not (kindIs "invalid" $t.tolerationSeconds)) -}}
{{- $seconds := $t.tolerationSeconds -}}
{{- if or (not (or (kindIs "float64" $seconds) (kindIs "int" $seconds) (kindIs "int64" $seconds))) (ne (float64 $seconds) (floor (float64 $seconds))) -}}
{{- fail (printf "%s.tolerationSeconds must be an integer" $item) -}}
{{- end -}}
{{/* Helm encodes numbers through float64; the rounded int64 endpoints can
     serialize outside int64 as well, so exclude those boundary values. */}}
{{- if or (ge (float64 $seconds) (float64 "9223372036854775808")) (le (float64 $seconds) (float64 "-9223372036854775808")) -}}
{{- fail (printf "%s.tolerationSeconds must fit a signed 64-bit integer after Helm number encoding" $item) -}}
{{- end -}}
{{- if ne $effect "NoExecute" -}}
{{- fail (printf "%s.tolerationSeconds requires effect NoExecute" $item) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
