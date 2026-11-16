{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{/*
Common labels for Kumo-owned wrapper resources. The Temporal dependency owns
the labels on its own resources.
*/}}
{{- define "kumo-temporal.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: kumo-temporal
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end }}

{{/*
The wrapper intentionally narrows the upstream chart to the supported Kumo
contract. This prevents a harmless-looking values override from reintroducing
a database, a second credential, plaintext persistence, or another service.
*/}}
{{- define "kumo-temporal.validate" -}}
{{- $policy := .Values.kumoTemporal -}}
{{- $temporal := .Values.temporal -}}
{{- if (index $temporal.server "internal-frontend").enabled -}}
{{- fail "temporal.server.internal-frontend.enabled must remain false" -}}
{{- end -}}
{{- if $temporal.server.config.namespaces.create -}}
{{- fail "temporal.server.config.namespaces.create must remain false; kumo-helm registers tenant namespaces" -}}
{{- end -}}
{{- $persistence := $temporal.server.config.persistence -}}
{{- if ne $persistence.defaultStore "default" -}}
{{- fail "temporal.server.config.persistence.defaultStore must be default" -}}
{{- end -}}
{{- if ne $persistence.visibilityStore "visibility" -}}
{{- fail "temporal.server.config.persistence.visibilityStore must be visibility" -}}
{{- end -}}
{{- $stores := $persistence.datastores -}}
{{- if ne (len $stores) 2 -}}
{{- fail "Temporal persistence must contain exactly the default and visibility PostgreSQL stores" -}}
{{- end -}}
{{- $default := required "Temporal default PostgreSQL store is required" (get $stores "default") -}}
{{- $visibility := required "Temporal visibility PostgreSQL store is required" (get $stores "visibility") -}}
{{- $defaultSql := required "Temporal default store must use sql" (get $default "sql") -}}
{{- $visibilitySql := required "Temporal visibility store must use sql" (get $visibility "sql") -}}
{{- range $name, $sql := dict "default" $defaultSql "visibility" $visibilitySql -}}
{{- if or (hasKey $sql "password") (hasKey $sql "passwordCommand") -}}
{{- fail (printf "Temporal %s store must use existingSecret; password and passwordCommand values are forbidden" $name) -}}
{{- end -}}
{{- if or (not $sql.existingSecret) (contains "REPLACE_" (upper (toString $sql.existingSecret))) -}}
{{- fail (printf "Temporal %s store existingSecret must name a pre-created Kubernetes Secret" $name) -}}
{{- end -}}
{{- if not $sql.secretKey -}}
{{- fail (printf "Temporal %s store secretKey is required" $name) -}}
{{- end -}}
{{- if or (not $sql.connectAddr) (contains "REPLACE_" (upper (toString $sql.connectAddr))) -}}
{{- fail (printf "Temporal %s store connectAddr must be a managed PostgreSQL host:port" $name) -}}
{{- end -}}
{{- if ne $sql.pluginName "postgres12_pgx" -}}
{{- fail (printf "Temporal %s store pluginName must be postgres12_pgx" $name) -}}
{{- end -}}
{{- if $sql.createDatabase -}}
{{- fail (printf "Temporal %s store createDatabase must remain false; the provider layer creates databases" $name) -}}
{{- end -}}
{{- if not $sql.manageSchema -}}
{{- fail (printf "Temporal %s store manageSchema must remain true" $name) -}}
{{- end -}}
{{- if not $sql.tls.enabled -}}
{{- fail (printf "Temporal %s store PostgreSQL TLS must remain enabled" $name) -}}
{{- end -}}
{{- if not $sql.tls.enableHostVerification -}}
{{- fail (printf "Temporal %s store PostgreSQL hostname verification must remain enabled" $name) -}}
{{- end -}}
{{- end -}}
{{- if ne $defaultSql.databaseName "temporal" -}}
{{- fail "Temporal default databaseName must be temporal" -}}
{{- end -}}
{{- if ne $visibilitySql.databaseName "temporal_visibility" -}}
{{- fail "Temporal visibility databaseName must be temporal_visibility" -}}
{{- end -}}
{{- range $field, $values := dict
      "user" (list $defaultSql.user $visibilitySql.user)
      "existingSecret" (list $defaultSql.existingSecret $visibilitySql.existingSecret)
      "secretKey" (list $defaultSql.secretKey $visibilitySql.secretKey)
      "connectAddr" (list $defaultSql.connectAddr $visibilitySql.connectAddr) -}}
{{- if ne (toString (first $values)) (toString (last $values)) -}}
{{- fail (printf "Temporal default and visibility stores must use the same %s" $field) -}}
{{- end -}}
{{- end -}}
{{- $transport := $policy.security.transport -}}
{{- $meshLabels := $transport.meshNamespaceLabels | default dict -}}
{{- $ambient := eq (toString (get $meshLabels "istio.io/dataplane-mode")) "ambient" -}}
{{- $revision := toString (get $meshLabels "istio.io/rev") -}}
{{- if and $transport.encryptionRequired (not $ambient) (not (hasPrefix "asm-" $revision)) -}}
{{- fail "encrypted transport requires Istio ambient mode or an AKS managed-Istio asm-* revision" -}}
{{- end -}}
{{- if and $ambient $revision -}}
{{- fail "meshNamespaceLabels must select either ambient mode or an AKS managed-Istio revision, not both" -}}
{{- end -}}
{{- $awsCaPath := "/etc/temporal/provider-certs/aws-rds-roots.pem" -}}
{{- if eq $policy.platform "aws" -}}
{{- if ne (toString $defaultSql.tls.caFile) $awsCaPath -}}
{{- fail (printf "AWS Temporal PostgreSQL must use the packaged CA at %s" $awsCaPath) -}}
{{- end -}}
{{- if ne (toString $visibilitySql.tls.caFile) $awsCaPath -}}
{{- fail (printf "AWS Temporal visibility PostgreSQL must use the packaged CA at %s" $awsCaPath) -}}
{{- end -}}
{{- else -}}
{{- if or (get $defaultSql.tls "caFile") (get $visibilitySql.tls "caFile") -}}
{{- fail "Azure managed PostgreSQL must use the stock image trust store; remove tls.caFile" -}}
{{- end -}}
{{- end -}}
{{- end }}
