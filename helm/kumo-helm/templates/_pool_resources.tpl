{{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0

Compare positive decimal quantities exactly. `digits` has no leading/trailing
zeroes; `order` is the number of digits before the decimal point (stored as a
string so fromJson cannot round it). Never expand a scientific exponent into
zeroes or convert quantities to float64: adjacent large values must differ.
*/}}
{{- define "kumo.poolQuantityGreater" -}}
{{- $a := .left -}}
{{- $b := .right -}}
{{- if ne $a.order $b.order -}}
{{- gt (int64 $a.order) (int64 $b.order) -}}
{{- else -}}
{{- $width := int (max (len $a.digits) (len $b.digits)) -}}
{{- gt (printf "%s%s" $a.digits (repeat (int (sub $width (len $a.digits))) "0"))
       (printf "%s%s" $b.digits (repeat (int (sub $width (len $b.digits))) "0")) -}}
{{- end -}}
{{- end -}}

{{/* Validate quantities here too: kumoConfig.extra bypasses the value schema. */}}
{{- define "kumo.poolQuantity" -}}
{{- if not (or (kindIs "string" .value) (kindIs "float64" .value) (kindIs "int64" .value) (kindIs "int" .value)) -}}
{{- fail (printf "%s must be a Kubernetes quantity string or number" .path) -}}
{{- end -}}
{{- $text := toString .value -}}
{{- if not (regexMatch "^[+]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+|[numkMGTPE]|[KMGTPE]i)?$" $text) -}}
{{- fail (printf "%s must be a valid Kubernetes quantity" .path) -}}
{{- end -}}
{{- $number := regexFind "^[+]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)" $text -}}
{{- $suffix := trimPrefix $number $text -}}
{{- $parts := splitList "." (trimPrefix "+" $number) -}}
{{- $fraction := "" -}}
{{- if eq (len $parts) 2 -}}{{- $fraction = index $parts 1 -}}{{- end -}}
{{- $digits := regexReplaceAll "^0+" (join "" $parts) "" -}}
{{- if eq $digits "" -}}{{- fail (printf "%s must be positive" .path) -}}{{- end -}}
{{- $power := sub 0 (len $fraction) -}}
{{- if hasSuffix "i" $suffix -}}
{{- /* Decimal digit multiplication keeps all bits for binary quantities up to
2^63-1. Each carry is at most 1023; no int64 multiplication can overflow. */ -}}
{{- $steps := int (index (dict "Ki" 1 "Mi" 2 "Gi" 3 "Ti" 4 "Pi" 5 "Ei" 6) $suffix) -}}
{{- range until $steps -}}
{{- $carry := int64 0 -}}
{{- $product := "" -}}
{{- range reverse (splitList "" $digits) -}}
{{- $digit := add (mul (atoi .) 1024) $carry -}}
{{- $product = printf "%d%s" (mod $digit 10) $product -}}
{{- $carry = div $digit 10 -}}
{{- end -}}
{{- if gt $carry 0 -}}{{- $product = printf "%d%s" $carry $product -}}{{- end -}}
{{- $digits = $product -}}
{{- end -}}
{{- else if regexMatch "^[eE][+-]?[0-9]+$" $suffix -}}
{{- $rawExponent := substr 1 (len $suffix) $suffix -}}
{{- $exponentDigits := regexReplaceAll "^[+-]?0*" $rawExponent "" -}}
{{- $canonical := default "0" $exponentDigits -}}
{{- if and (hasPrefix "-" $rawExponent) (ne $canonical "0") -}}
{{- $canonical = printf "-%s" $canonical -}}
{{- end -}}
{{- $exponent := atoi $canonical -}}
{{- if ne (toString $exponent) $canonical -}}
{{- fail (printf "%s has an unsupported numeric exponent" .path) -}}
{{- end -}}
{{- /* Check before adding: absurd exponents must not wrap int64. The lower
bound matches Python Decimal.MIN_ETINY; it is not a workload sizing floor. */ -}}
{{- if gt $exponent (add (len $fraction) 19) -}}
{{- fail (printf "%s must be <= 2^63-1" .path) -}}
{{- end -}}
{{- if lt $exponent (add -1999999999999999997 (len $fraction)) -}}
{{- fail (printf "%s has an unsupported numeric exponent" .path) -}}
{{- end -}}
{{- $power = add $power $exponent -}}
{{- else -}}
{{- $power = add $power (index (dict "n" -9 "u" -6 "m" -3 "" 0 "k" 3 "M" 6 "G" 9 "T" 12 "P" 15 "E" 18) $suffix) -}}
{{- end -}}
{{- $significant := regexReplaceAll "0+$" $digits "" -}}
{{- $power = add $power (sub (len $digits) (len $significant)) -}}
{{- $quantity := dict "digits" $significant "order" (toString (add (len $significant) $power)) -}}
{{- $maximum := dict "digits" "9223372036854775807" "order" "19" -}}
{{- if eq (include "kumo.poolQuantityGreater" (dict "left" $quantity "right" $maximum)) "true" -}}
{{- fail (printf "%s must be <= 2^63-1" .path) -}}
{{- end -}}
{{- if and (eq .resource "cpu") (lt $power -3) -}}
{{- fail (printf "%s must be a multiple of 1m CPU" .path) -}}
{{- end -}}
{{- toJson $quantity -}}
{{- end -}}

{{- define "kumo.validatePoolRequestLimits" -}}
{{- $limits := .resources.limits | default dict -}}
{{- range $resource, $request := .resources.requests | default dict -}}
{{- if hasKey $limits $resource -}}
{{- if eq (include "kumo.poolQuantityGreater" (dict "left" $request "right" (index $limits $resource))) "true" -}}
{{- fail (printf "%s: effective %s request must not exceed limit" $.path $resource) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "kumo.validatePoolResources" -}}
{{- $path := "poolResources (effective data_compute.k8s_pool.pool_resources)" -}}
{{- if not (kindIs "map" .) -}}{{- fail (printf "%s must be an object" $path) -}}{{- end -}}
{{- $minimums := dict "cpu" (dict "digits" "1" "order" "1") "memory" (dict "digits" "17179869184" "order" "11") -}}
{{- $parsed := dict -}}
{{- range $group, $resources := . -}}
{{- if not (has $group (list "daemon" "activity_defaults" "training_overrides" "prediction_overrides")) -}}
{{- fail (printf "%s: unknown group %s" $path $group) -}}
{{- end -}}
{{- if not (kindIs "map" $resources) -}}{{- fail (printf "%s.%s must be an object" $path $group) -}}{{- end -}}
{{- $normalized := dict -}}
{{- range $field, $quantities := $resources -}}
{{- if not (has $field (list "requests" "limits")) -}}{{- fail (printf "%s.%s: unknown field %s" $path $group $field) -}}{{- end -}}
{{- if not (kindIs "map" $quantities) -}}{{- fail (printf "%s.%s.%s must be an object" $path $group $field) -}}{{- end -}}
{{- $values := dict -}}
{{- range $resource, $value := $quantities -}}
{{- $resourcePath := printf "%s.%s.%s.%s" $path $group $field $resource -}}
{{- if not (has $resource (list "cpu" "memory")) -}}{{- fail (printf "%s: only cpu and memory are supported" $resourcePath) -}}{{- end -}}
{{- $quantity := include "kumo.poolQuantity" (dict "value" $value "resource" $resource "path" $resourcePath) | fromJson -}}
{{- if and (eq $field "requests") (eq (include "kumo.poolQuantityGreater" (dict "left" (index $minimums $resource) "right" $quantity)) "true") -}}
{{- fail (printf "%s: request must be at least %s" $resourcePath (index (dict "cpu" "1 CPU" "memory" "16Gi") $resource)) -}}
{{- end -}}
{{- $_ := set $values $resource $quantity -}}
{{- end -}}
{{- $_ := set $normalized $field $values -}}
{{- end -}}
{{- include "kumo.validatePoolRequestLimits" (dict "resources" $normalized "path" (printf "%s.%s" $path $group)) -}}
{{- $_ := set $parsed $group $normalized -}}
{{- end -}}
{{- $daemon := dict "requests" (dict "cpu" (dict "digits" "1" "order" "1") "memory" (dict "digits" "2147483648" "order" "10")) -}}
{{- $daemon = mergeOverwrite $daemon ($parsed.daemon | default dict) -}}
{{- include "kumo.validatePoolRequestLimits" (dict "resources" $daemon "path" (printf "%s.daemon" $path)) -}}
{{- range $group := list "training_overrides" "prediction_overrides" -}}
{{- $effective := mergeOverwrite (deepCopy ($parsed.activity_defaults | default dict)) (index $parsed $group | default dict) -}}
{{- include "kumo.validatePoolRequestLimits" (dict "resources" $effective "path" (printf "%s.%s" $path $group)) -}}
{{- end -}}
{{- end -}}
