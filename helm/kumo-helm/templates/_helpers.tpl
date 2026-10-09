{{- /*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/ -}}
{{/*
Temporal workflow namespace — defaults to global.tenant
*/}}
{{- define "kumo.temporalNamespace" -}}
{{- .Values.temporal.namespace | default .Values.global.tenant -}}
{{- end }}

{{/* Temporal gRPC endpoint — preserve an explicit port, otherwise use 7233. */}}
{{- define "kumo.temporalAddress" -}}
{{- $address := toString .Values.temporal.address -}}
{{- if regexMatch ":[0-9]+$" $address -}}
{{- $address -}}
{{- else -}}
{{- printf "%s:7233" $address -}}
{{- end -}}
{{- end }}

{{/* Provider CA bundle baked into both Kumo backend images. */}}
{{- define "kumo.postgresCaPath" -}}
{{- if eq .Values.postgres.authMode "aws_iam" -}}
/opt/kumo/certs/aws-rds-roots.pem
{{- else -}}
/etc/ssl/certs/ca-certificates.crt
{{- end -}}
{{- end }}

{{/*
Reject unresolved install placeholders and values that cannot work with the
current images. Cloud identity remains provider-neutral: object-store installs
must configure each identity-bearing role, but the chart does not interpret the
annotation key or value.
*/}}
{{- define "kumo.deployStage" -}}
{{- ternary "prod" "preprod" .Values.auth.enabled -}}
{{- end }}

{{- define "kumo.validateInstallConfig" -}}
{{- $tenant := toString .Values.global.tenant -}}
{{- $tag := toString .Values.image.tag -}}
{{- $bucket := toString .Values.kumoConfig.storage.dataBucket -}}
{{- $gsaKey := .Values.kumoConfig.gsaKey | default dict -}}
{{- $gsaSecret := toString ($gsaKey.existingSecret | default "") -}}
{{- if and (not $gsaSecret) ($gsaKey.projectId | default "") -}}
{{- fail "kumoConfig.gsaKey.projectId requires existingSecret" -}}
{{- end -}}
{{- if hasKey .Values "security" -}}
{{- fail "security is not supported: podSecurityContext and containerSecurityContext moved to global.*, and security.transport was removed with the chart's service-mesh ownership" -}}
{{- end -}}
{{- if hasKey .Values "serviceAccount" -}}
{{- fail "serviceAccount is not supported; configure the current serviceAccounts.control, worker, and controller roles" -}}
{{- end -}}
{{- if hasKey .Values "platform" -}}
{{- fail "platform is not supported; apply a provider profile as a values overlay" -}}
{{- end -}}
{{- if hasKey .Values "routing" -}}
{{- fail "routing is not configurable; the current UI owns the single frontend routing contract" -}}
{{- end -}}
{{- if hasKey .Values.image "uiTag" -}}
{{- fail "image.uiTag is not supported; all current Kumo images use image.tag" -}}
{{- end -}}
{{- if hasKey .Values.workloads.ui "privileged" -}}
{{- fail "workloads.ui.privileged is not supported; the current UI always runs unprivileged" -}}
{{- end -}}
{{- if (get (.Values.global.containerSecurityContext | default dict) "privileged") -}}
{{- fail "global.containerSecurityContext.privileged cannot be enabled; current Kumo images run unprivileged" -}}
{{- end -}}
{{- if or (eq $tenant "my-tenant") (contains "REPLACE_" (upper $tenant)) -}}
{{- fail "global.tenant must replace the chart placeholder" -}}
{{- end -}}
{{- if or (eq $tag "replace-me") (not (regexMatch "^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$" $tag)) -}}
{{- fail "image.tag must be a valid container tag; kumo-stack supplies a default that customers may override" -}}
{{- end -}}
{{- if contains "REPLACE_" (upper $bucket) -}}
{{- fail "kumoConfig.storage.dataBucket must replace the profile placeholder" -}}
{{- end -}}
{{- if contains "REPLACE_" (upper (toString .Values.kumoConfig.diskgraph.storageClass)) -}}
{{- fail "kumoConfig.diskgraph.storageClass must replace the chart placeholder" -}}
{{- end -}}
{{- $pg := .Values.postgres -}}
{{- $dsnSecret := $pg.dsnSecret | default dict -}}
{{- $usesSecret := ne (toString ($dsnSecret.secretName | default "")) "" -}}
{{- /* The app accepts exactly one connection source: a mounted DSN Secret or a
       cloud auth mode. Reject both, and reject neither. */ -}}
{{- if and $usesSecret $pg.authMode -}}
{{- fail "postgres: set either authMode (passwordless cloud identity) or dsnSecret (password auth), not both" -}}
{{- end -}}
{{- if not (or $usesSecret $pg.authMode) -}}
{{- fail "postgres: set authMode (aws_iam|azure_entra) for passwordless cloud identity, or dsnSecret.secretName for password auth" -}}
{{- end -}}
{{- if and (($pg.caSecret | default dict).secretName) (not $usesSecret) -}}
{{- fail "postgres.caSecret applies only to dsnSecret (password auth); with authMode the chart uses the CA baked into the image" -}}
{{- end -}}
{{- if $usesSecret -}}
{{- if not ($dsnSecret.key | default "") -}}
{{- fail "postgres.dsnSecret.key is required: it is both the Secret key and the name the application resolves the connection under" -}}
{{- end -}}
{{- else -}}
{{- if not (has $pg.authMode (list "aws_iam" "azure_entra")) -}}
{{- fail "postgres.authMode must be aws_iam or azure_entra" -}}
{{- end -}}
{{- if or (not $pg.host) (contains "REPLACE_" (upper $pg.host)) -}}
{{- fail "postgres.host must be the managed Postgres DNS name" -}}
{{- end -}}
{{- range $field, $value := dict "database" $pg.database "username" $pg.username -}}
{{- if not $value -}}
{{- fail (printf "postgres.%s is required" $field) -}}
{{- end -}}
{{- end -}}
{{- if and (eq $pg.authMode "aws_iam") (not $pg.region) -}}
{{- fail "postgres.region is required for aws_iam" -}}
{{- end -}}
{{- end -}}
{{- $extra := .Values.kumoConfig.extra | default dict -}}
{{- if hasKey $extra "auth" -}}
{{- fail "kumoConfig.extra.auth cannot override chart-managed authentication wiring" -}}
{{- end -}}
{{- if hasKey $extra "external_host" -}}
{{- fail "kumoConfig.extra.external_host cannot override chart-managed exposure wiring" -}}
{{- end -}}
{{- range $key := list "ambient_s3" "ambient_gcp" "azure_storage" -}}
{{- if hasKey $extra $key -}}
{{- fail (printf "kumoConfig.extra.%s cannot override chart-managed object-store identity wiring" $key) -}}
{{- end -}}
{{- end -}}
{{- if hasKey $extra "orchestration" -}}
{{- $orchestrationExtra := get $extra "orchestration" -}}
{{- if not (kindIs "map" $orchestrationExtra) -}}
{{- fail "kumoConfig.extra.orchestration must be a map" -}}
{{- end -}}
{{- range $key := list "type" "server_address" "namespace" "tls_cert_dir" -}}
{{- if hasKey $orchestrationExtra $key -}}
{{- fail (printf "kumoConfig.extra.orchestration.%s cannot override chart-managed Temporal wiring" $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if hasKey $extra "metadata_storage" -}}
{{- $metadataExtra := get $extra "metadata_storage" -}}
{{- if not (kindIs "map" $metadataExtra) -}}
{{- fail "kumoConfig.extra.metadata_storage must be a map" -}}
{{- end -}}
{{- range $key := list "type" "dsn" "dsn_secret_name" "auth_mode" "username" "host" "port" "database" "region" "sslrootcert" -}}
{{- if hasKey $metadataExtra $key -}}
{{- fail (printf "kumoConfig.extra.metadata_storage.%s cannot override chart-managed StateDB wiring" $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if hasKey $extra "secrets" -}}
{{- $secretsExtra := get $extra "secrets" -}}
{{- if not (kindIs "map" $secretsExtra) -}}
{{- fail "kumoConfig.extra.secrets must be a map" -}}
{{- end -}}
{{- range $key := list "backend" "mount_path" -}}
{{- if hasKey $secretsExtra $key -}}
{{- fail (printf "kumoConfig.extra.secrets.%s cannot override chart-managed Secret wiring" $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if hasKey $extra "storage" -}}
{{- $storageExtra := get $extra "storage" -}}
{{- if not (kindIs "map" $storageExtra) -}}
{{- fail "kumoConfig.extra.storage must be a map" -}}
{{- end -}}
{{- if hasKey $storageExtra "data_plane_storage_root" -}}
{{- fail "kumoConfig.extra.storage.data_plane_storage_root cannot override chart-managed object storage wiring" -}}
{{- end -}}
{{- range $key := list "control_plane_storage_root" "artifact_root" -}}
{{- if hasKey $storageExtra $key -}}
{{- $rootValue := get $storageExtra $key -}}
{{- if not (empty $rootValue) -}}
{{- $root := toString $rootValue -}}
{{- $supported := or (hasPrefix "s3://" $root) (hasPrefix "gs://" $root) (hasPrefix "abfss://" $root) -}}
{{- if not $supported -}}
{{- fail (printf "kumoConfig.extra.storage.%s must use s3://, gs://, or abfss:// when set" $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $gsaSecret -}}
{{- if not (hasPrefix "gs://" $bucket) -}}
{{- fail "kumoConfig.gsaKey.existingSecret applies only to a gs:// kumoConfig.storage.dataBucket" -}}
{{- end -}}
{{- if not ($gsaKey.key | default "") -}}
{{- fail "kumoConfig.gsaKey.key is required when existingSecret is set" -}}
{{- end -}}
{{- $restExtraEnv := .Values.workloads.rest.extraEnv | default dict -}}
{{- if hasKey $restExtraEnv "GOOGLE_APPLICATION_CREDENTIALS" -}}
{{- fail "workloads.rest.extraEnv.GOOGLE_APPLICATION_CREDENTIALS cannot override kumoConfig.gsaKey" -}}
{{- end -}}
{{- if and ($gsaKey.projectId | default "") (hasKey $restExtraEnv "GOOGLE_CLOUD_PROJECT") -}}
{{- fail "workloads.rest.extraEnv.GOOGLE_CLOUD_PROJECT cannot override kumoConfig.gsaKey.projectId" -}}
{{- end -}}
{{- $poolExtraEnv := .Values.workloads.poolcontroller.extraEnv | default dict -}}
{{- range $key := list
      "GOOGLE_APPLICATION_CREDENTIALS"
      "POOL_WORKER_GCP_APPLICATION_CREDENTIALS_SECRET"
      "POOL_WORKER_GCP_APPLICATION_CREDENTIALS_KEY" -}}
{{- if hasKey $poolExtraEnv $key -}}
{{- fail (printf "workloads.poolcontroller.extraEnv.%s cannot override kumoConfig.gsaKey" $key) -}}
{{- end -}}
{{- end -}}
{{- if and ($gsaKey.projectId | default "") (hasKey $poolExtraEnv "GOOGLE_CLOUD_PROJECT") -}}
{{- fail "workloads.poolcontroller.extraEnv.GOOGLE_CLOUD_PROJECT cannot override kumoConfig.gsaKey" -}}
{{- end -}}
{{- $gcpIdentityAnnotationKeys := list
      "iam.gke.io/gcp-service-account"
      "cloud.google.com/workload-identity-provider"
      "cloud.google.com/service-account-email"
      "cloud.google.com/injection-mode"
      "cloud.google.com/audience" -}}
{{- range $role := list "control" "worker" "controller" -}}
{{- $config := include "kumo.serviceAccountConfig" (dict "root" $ "role" $role) | fromJson -}}
{{- $annotations := $config.annotations | default dict -}}
{{- range $key := $gcpIdentityAnnotationKeys -}}
{{- if hasKey $annotations $key -}}
{{- fail (printf "kumoConfig.gsaKey.existingSecret cannot be combined with GCP workload-identity annotations: clear serviceAccounts.%s.annotations[%q]" $role $key) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if hasPrefix "abfs://" $bucket -}}
{{- fail "kumoConfig.storage.dataBucket must use abfss:// for encrypted Azure Data Lake traffic" -}}
{{- end -}}
{{- if hasPrefix "az://" $bucket -}}
{{- fail "kumoConfig.storage.dataBucket does not support az://; use an abfss:// ADLS URI" -}}
{{- end -}}
{{- $objectStore := or (hasPrefix "s3://" $bucket) (hasPrefix "gs://" $bucket) (hasPrefix "abfss://" $bucket) -}}
{{- if not $objectStore -}}
{{- fail "kumoConfig.storage.dataBucket must use s3://, gs://, or abfss://" -}}
{{- end -}}
{{- $oidcIssuer := toString (.Values.auth.oidc.issuerUrl | default "") -}}
{{- if and .Values.auth.enabled (eq .Values.auth.method "oidc") $oidcIssuer (not (hasPrefix "https://" $oidcIssuer)) -}}
{{- fail "auth.oidc.issuerUrl must use https://" -}}
{{- end -}}
{{- if .Values.auth.enabled -}}
{{- range $origin := .Values.auth.allowedReturnOrigins | default list -}}
{{- $originString := toString $origin -}}
{{- if contains "," $originString -}}
{{- fail (printf "auth.allowedReturnOrigins entry %q must contain exactly one origin" $originString) -}}
{{- end -}}
{{- if not (hasPrefix "https://" $originString) -}}
{{- fail (printf "auth.allowedReturnOrigins entry %q must use https://" $originString) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- $slackUrl := toString (.Values.observability.slackUrl | default "") -}}
{{- if and $slackUrl (not (hasPrefix "https://" $slackUrl)) -}}
{{- fail "observability.slackUrl must use https://" -}}
{{- end -}}
{{- $usesStaticGcpCredentials := and (hasPrefix "gs://" $bucket) (ne $gsaSecret "") -}}
{{- if not $usesStaticGcpCredentials -}}
{{- range $role := list "control" "worker" "controller" -}}
{{- $config := include "kumo.serviceAccountConfig" (dict "root" $ "role" $role) | fromJson -}}
{{- $annotations := $config.annotations | default dict -}}
{{- if eq (len (compact (values $annotations))) 0 -}}
{{- fail (printf "object storage requires a nonempty workload-identity annotation on serviceAccounts.%s" $role) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Resolved fixed role ServiceAccount configuration. Cloud federation is bound
to each ServiceAccount subject.
*/}}
{{- define "kumo.serviceAccountConfig" -}}
{{- $root := .root -}}
{{- $role := .role -}}
{{- $accounts := $root.Values.serviceAccounts | default dict -}}
{{- $roleConfig := index $accounts $role | default dict -}}
{{- $name := printf "kumo-%s" $role -}}
{{- $annotations := deepCopy ($roleConfig.annotations | default dict) -}}
{{- $podLabels := deepCopy ($root.Values.global.podLabels | default dict) -}}
{{- $podLabels = mergeOverwrite $podLabels (deepCopy ($roleConfig.podLabels | default dict)) -}}
{{- dict "name" $name "annotations" $annotations "podLabels" $podLabels | toJson -}}
{{- end }}

{{- define "kumo.serviceAccountName" -}}
{{- $config := include "kumo.serviceAccountConfig" . | fromJson -}}
{{- $config.name -}}
{{- end }}

{{- define "kumo.serviceAccountAutomount" -}}
{{- has .role (list "worker" "controller") -}}
{{- end }}

{{- define "kumo.workloadPodLabels" -}}
{{- $config := include "kumo.serviceAccountConfig" . | fromJson -}}
{{- with $config.podLabels -}}{{- toYaml . -}}{{- end -}}
{{- end }}

{{/*
Portable security-context mergers. Callers may supply an `extra` map for the
few workload-specific fields (for example an ephemeral PVC's fsGroup). Null
removes an inherited key so platform overlays can reject fixed identities.
*/}}
{{- define "kumo.podSecurityContext" -}}
{{- $ctx := deepCopy (.root.Values.global.podSecurityContext | default dict) -}}
{{- $ctx = mergeOverwrite $ctx (deepCopy (.extra | default dict)) -}}
{{- range $key, $value := $ctx -}}
{{- if kindIs "invalid" $value -}}{{- $_ := unset $ctx $key -}}{{- end -}}
{{- end -}}
{{- with $ctx -}}{{- toYaml . -}}{{- end -}}
{{- end }}

{{- define "kumo.containerSecurityContext" -}}
{{- $ctx := deepCopy (.root.Values.global.containerSecurityContext | default dict) -}}
{{- $ctx = mergeOverwrite $ctx (deepCopy (.extra | default dict)) -}}
{{- range $key, $value := $ctx -}}
{{- if kindIs "invalid" $value -}}{{- $_ := unset $ctx $key -}}{{- end -}}
{{- end -}}
{{- $_ := set $ctx "privileged" false -}}
{{- with $ctx -}}{{- toYaml . -}}{{- end -}}
{{- end }}

{{/*
Stable deployment data-plane storage root. global.tenant is retained as the
chart's deployment slug, not an application authorization or tenancy boundary.
Keeping it in the path preserves existing data and IAM prefix grants.
*/}}
{{- define "kumo.dataPlaneStorageRoot" -}}
{{- printf "%s/%s" (required "kumoConfig.storage.dataBucket is required" .Values.kumoConfig.storage.dataBucket) .Values.global.tenant -}}
{{- end }}

{{/*
Shared mount path for Secret material.
Baked constant; kept in one place so config paths and volume mounts never drift.
*/}}
{{- define "kumo.secretsMountPath" -}}/var/run/secrets/kumo{{- end }}

{{- define "kumo.imageRepoBase" -}}
{{- $p := .Values.image.repoPath | default "" -}}
{{- if $p -}}{{ .Values.image.registry }}/{{ $p }}{{- else -}}{{ .Values.image.registry }}{{- end -}}
{{- end }}

{{- define "kumo.restImage" -}}
{{ include "kumo.imageRepoBase" . }}/kumo-backend-cpu:{{ .Values.image.tag }}
{{- end }}

{{- define "kumo.dataplaneImage" -}}
{{ include "kumo.imageRepoBase" . }}/kumo-backend-cpu:{{ .Values.image.tag }}
{{- end }}

{{- define "kumo.poolcontrollerImage" -}}
{{ include "kumo.imageRepoBase" . }}/kumo-backend-cpu:{{ .Values.image.tag }}
{{- end }}

{{- define "kumo.uiImage" -}}
{{ include "kumo.imageRepoBase" . }}/kumo-ui:{{ .Values.image.tag }}
{{- end }}

{{/*
DEPLOY_ENV_CONFIG_JSON — the structured deploy config consumed by ALL kumo pods via
service_framework deploy_env(). Emitted into the single kumo-config ConfigMap in the
release namespace; shape mirrors RawDeployConfig (kumo: service_framework/_raw_deploy_config.py).

The DiskGraph-mode constants are BAKED here (the app's own defaults are wrong for this
mode — K8s ml-compute, AWS_SM secrets, CloudWatch logging). The remaining sections are
assembled from the operator-facing value blocks:
  - data_compute     : DISKGRAPH + k8s pool; node types/data_root from kumoConfig.diskgraph
  - storage          : single-tenant root from kumoConfig.storage.dataBucket
  - orchestration    : self-hosted Temporal, optionally with client mTLS
  - metadata_storage : managed Postgres connection and cloud identity auth mode
Finally kumoConfig.extra (snake_case, mirrors RawDeployConfig) is DEEP-MERGED
on top for forward fields and non-security defaults. Chart-managed Temporal
StateDB, authentication, Secret, object-store, and exposure fields cannot be
overridden through that escape hatch.
*/}}
{{- define "kumo.deployEnvConfigJson" -}}
{{- include "kumo.validateInstallConfig" . -}}
{{- include "kumo.validatePodLabels" . -}}
{{- /* tenantSecretKey required-check — production pods need the deployment secret at startup. */ -}}
{{- $ts := .Values.tenantSecretKey | default dict -}}
{{- $stage := include "kumo.deployStage" . -}}
{{- if and $ts.secretName (not $ts.secretKey) -}}
{{-   fail "tenantSecretKey.secretName is set but tenantSecretKey.secretKey is empty — set both." -}}
{{- end -}}
{{- if and (not $ts.secretName) (eq $stage "prod") -}}
{{-   fail "tenantSecretKey is required in prod: the app reads the deployment Fernet key and peppers from kumo-secret-key at startup. Provide tenantSecretKey containing a valid Fernet key, or set auth.enabled=false, which derives the preprod stage." -}}
{{- end -}}
{{- $dg := .Values.kumoConfig.diskgraph -}}
{{- include "kumo.validatePoolExtraTolerations" $dg.poolExtraTolerations -}}
{{- if not $dg.nodeTypes -}}
{{-   fail "kumoConfig.diskgraph.nodeTypes must contain at least one entry" -}}
{{- end -}}
{{- range $name, $nodeType := $dg.nodeTypes -}}
{{-   if not (kindIs "map" $nodeType) -}}
{{-     fail (printf "kumoConfig.diskgraph.nodeTypes[%q] must be a map with a positive ssd_gib" $name) -}}
{{-   else if or (not (hasKey $nodeType "ssd_gib")) (le (int $nodeType.ssd_gib) 0) -}}
{{-     fail (printf "kumoConfig.diskgraph.nodeTypes[%q].ssd_gib must be positive" $name) -}}
{{-   else if and (hasKey $nodeType "gpu_count") (le (int $nodeType.gpu_count) 0) -}}
{{-     fail (printf "kumoConfig.diskgraph.nodeTypes[%q].gpu_count must be positive when set" $name) -}}
{{-   end -}}
{{- end -}}
{{- $dsnSecret := .Values.postgres.dsnSecret | default dict -}}
{{- $metadata := dict "type" "postgres" -}}
{{- if $dsnSecret.secretName -}}
{{- /* Password auth: the app reads the structured connection blob from the
       mounted Secret. Emitting no other connection field keeps the app's
       "exactly one source" validator satisfied. */ -}}
{{- $_ := set $metadata "dsn_secret_name" $dsnSecret.key -}}
{{- else -}}
{{- $_ := set $metadata "auth_mode" .Values.postgres.authMode -}}
{{- $_ := set $metadata "username" .Values.postgres.username -}}
{{- $_ := set $metadata "host" .Values.postgres.host -}}
{{- $_ := set $metadata "port" .Values.postgres.port -}}
{{- $_ := set $metadata "database" .Values.postgres.database -}}
{{- $_ := set $metadata "region" .Values.postgres.region -}}
{{- $_ := set $metadata "sslrootcert" (include "kumo.postgresCaPath" .) -}}
{{- end -}}
{{- $cfg := dict
    "data_compute" (dict
      "type" "DISKGRAPH"
      "data_root" $dg.dataRoot
      "pool_strategy" "k8s"
      "pool_max_ssd_idle_secs" $dg.snapshotCacheTtlSeconds
      "k8s_pool" (dict "pool_diskgraph_node_types" $dg.nodeTypes))
    "secrets" (dict "backend" "k8s_mount" "mount_path" (include "kumo.secretsMountPath" .))
    "logging" (dict "backend" "stdout")
    "storage" (dict "data_plane_storage_root" (include "kumo.dataPlaneStorageRoot" .))
    "orchestration" (dict
      "type" "custom"
      "server_address" (include "kumo.temporalAddress" .)
      "namespace" (include "kumo.temporalNamespace" .)
      "tls_cert_dir" (ternary (include "kumo.temporalCertDir" .) ""
        (ne .Values.temporal.clientMtlsSecret "")))
    "metadata_storage" $metadata -}}
{{- with $dg.poolExtraTolerations -}}
{{- $_ := set $cfg.data_compute.k8s_pool "pool_extra_tolerations" (deepCopy .) -}}
{{- end -}}
{{- with $dg.poolResources -}}
{{- $poolResources := dict -}}
{{- range $valueName, $configName := dict
    "daemon" "daemon"
    "activityDefaults" "activity_defaults"
    "trainingOverrides" "training_overrides"
    "predictionOverrides" "prediction_overrides" -}}
{{- if hasKey $dg.poolResources $valueName -}}
{{- $_ := set $poolResources $configName (deepCopy (index $dg.poolResources $valueName)) -}}
{{- end -}}
{{- end -}}
{{- $_ := set $cfg.data_compute.k8s_pool "pool_resources" $poolResources -}}
{{- end -}}
{{- if .Values.auth.enabled }}
{{-   $dc := dict
        "login_method" .Values.auth.method
        "allowed_return_origins" (.Values.auth.allowedReturnOrigins | default list)
        "admin_emails" (.Values.auth.adminEmails | default list) -}}
{{-   if eq .Values.auth.method "oidc" -}}
{{-     $o := .Values.auth.oidc | default dict -}}
{{-     $_ := required "auth.oidc.clientSecret.secretName is required when auth.method=oidc (the chart delivers it as env; the app crashes at get_secrets without it)" ($o.clientSecret | default dict).secretName -}}
{{-     $_ := required "auth.oidc.clientSecret.secretKey is required when auth.method=oidc (an empty key renders secretKeyRef.key: empty, which is invalid)" ($o.clientSecret | default dict).secretKey -}}
{{-     $_ := set $dc "oidc" (dict
          "issuer" (required "auth.oidc.issuerUrl is required when auth.method=oidc" $o.issuerUrl)
          "client_id" (required "auth.oidc.clientId is required when auth.method=oidc" $o.clientId)
          "client_secret_ref" (dict "secret_name" "KUMO_OIDC_CLIENT_SECRET" "secret_key" "KUMO_OIDC_CLIENT_SECRET")
          "groups_claim" ($o.groupsClaim | default "groups")) -}}
{{-   else if eq .Values.auth.method "password" -}}
{{-     $_ := required "auth.password.temporaryPassword.secretName is required when auth.method=password (the chart delivers it as env; the app crashes at get_secrets without it)" ((.Values.auth.password | default dict).temporaryPassword | default dict).secretName -}}
{{-     $_ := required "auth.password.temporaryPassword.secretKey is required when auth.method=password (an empty key renders secretKeyRef.key: empty, which is invalid)" ((.Values.auth.password | default dict).temporaryPassword | default dict).secretKey -}}
{{-     $_ := set $dc "password" (dict
          "temporary_password_ref" (dict "secret_name" "KUMO_DC_TEMP_PASSWORD" "secret_key" "KUMO_DC_TEMP_PASSWORD")) -}}
{{-   else -}}
{{-     fail (printf "auth.method must be 'oidc' or 'password', got %q" .Values.auth.method) -}}
{{-   end -}}
{{-   $_ := set $cfg "auth" (dict "enabled" true "backend" "direct_connect" "direct_connect" $dc) -}}
{{- end }}
{{- $amb := .Values.kumoConfig.ambientS3 | default dict }}
{{- if $amb.enabled }}
{{-   $arns := $amb.approvedRoleArns | default list }}
{{-   if not $arns }}
{{-     $workerServiceAccount := include "kumo.serviceAccountConfig" (dict "root" . "role" "worker") | fromJson }}
{{-     $arn := index ($workerServiceAccount.annotations | default dict) "eks.amazonaws.com/role-arn" }}
{{-     $arns = list (required "kumoConfig.ambientS3.enabled=true needs a role ARN: set serviceAccounts.worker.annotations[\"eks.amazonaws.com/role-arn\"], or set kumoConfig.ambientS3.approvedRoleArns explicitly" $arn) }}
{{-   end }}
{{-   range $arn := $arns }}
{{-     if not (regexMatch "^arn:(aws|aws-us-gov|aws-cn):iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_-]+$" $arn) }}
{{-       fail (printf "kumoConfig.ambientS3: %q is not a concrete, pathless IAM role ARN (arn:aws:iam::<account>:role/<name>). The app's AmbientS3WorkloadConfig rejects role paths (role/<path>/<name>) and non-role ARNs when parsing DEPLOY_ENV_CONFIG_JSON, which crashes every backend at startup — assumed-role session ARNs normalize back to a pathless role, so pathful roles can never match the allowlist. Use a pathless role." $arn) }}
{{-     end }}
{{-   end }}
{{-   $_ := set $cfg "ambient_s3" (dict "approved_role_arns" $arns) }}
{{- end }}
{{- /* Externally reachable host: app-generated absolute URLs (e.g. the training-job
       tracking_url) use it instead of the SaaS <tenant>.kumoai.cloud convention. */}}
{{- if .Values.exposure.host }}
{{- $_ := set $cfg "external_host" .Values.exposure.host }}
{{- end }}
{{- with .Values.kumoConfig.extra }}
{{- $cfg = mergeOverwrite $cfg (deepCopy .) -}}
{{- end }}
{{- /* Validate after the merge so kumoConfig.extra cannot bypass the chart value schema. */ -}}
{{- if hasKey $cfg.data_compute.k8s_pool "pool_extra_tolerations" -}}
{{- include "kumo.validatePoolExtraTolerations" $cfg.data_compute.k8s_pool.pool_extra_tolerations -}}
{{- end -}}
{{- if hasKey $cfg.data_compute.k8s_pool "pool_resources" -}}
{{- include "kumo.validatePoolResources" $cfg.data_compute.k8s_pool.pool_resources -}}
{{- end -}}
{{- $snapshotCacheTtlSeconds := dig "data_compute" "pool_max_ssd_idle_secs" 0 $cfg -}}
{{- if lt (int $snapshotCacheTtlSeconds) 300 -}}
{{- fail "effective data_compute.pool_max_ssd_idle_secs must be at least 300 seconds" -}}
{{- end -}}
{{- $cfg | toJson -}}
{{- end }}

{{/*
Hook annotations shared by the preflight's auxiliary resources (SA, Roles,
RoleBindings, ConfigMap — weight -20, before the Job at -15). Cleaned up on
success AND failure: they grant Secret reads and must not outlive the hook.
The Job declares its own annotations (no hook-failed) so a failed run's pod
and logs survive for debugging.
*/}}
{{- define "kumo.preflightAuxHookAnnotations" -}}
"helm.sh/hook": pre-install,pre-upgrade
"helm.sh/hook-weight": "-20"
"helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded,hook-failed
{{- end }}

{{/*
Common labels applied to all resources
*/}}
{{- define "kumo.commonLabels" -}}
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/part-of: kumo
tenant: {{ .Values.global.tenant }}
{{- end }}

{{/*
Reject caller labels that collide with chart-owned pod identity. The frontend
NetworkPolicy boundary reserves the instance and component labels.
*/}}
{{- define "kumo.validatePodLabels" -}}
{{- $labelSets := dict "global.podLabels" (.Values.global.podLabels | default dict) -}}
{{- range $role := list "control" "worker" "controller" -}}
{{-   $roleConfig := index ($.Values.serviceAccounts | default dict) $role | default dict -}}
{{-   $_ := set $labelSets (printf "serviceAccounts.%s.podLabels" $role) ($roleConfig.podLabels | default dict) -}}
{{- end -}}
{{- $reserved := list "app" "tenant" "app.kubernetes.io/managed-by" "app.kubernetes.io/part-of" -}}
{{- $reserved = concat $reserved (list "app.kubernetes.io/instance" "app.kubernetes.io/component") -}}
{{- range $path, $labels := $labelSets -}}
{{-   range $key := $reserved -}}
{{-     if hasKey $labels $key -}}
{{-       fail (printf "%s[%q] is reserved by the chart" $path $key) -}}
{{-     end -}}
{{-   end -}}
{{- end -}}
{{- end -}}

{{- define "kumo.frontendPodLabels" -}}
{{- $root := .root -}}
app.kubernetes.io/instance: {{ $root.Release.Name | quote }}
app.kubernetes.io/component: {{ .component | quote }}
{{- end }}

{{/*
imagePullSecrets block — rendered only when image.pullSecretName is set
*/}}
{{- define "kumo.imagePullSecrets" -}}
{{- if .Values.image.pullSecretName }}
imagePullSecrets:
- name: {{ .Values.image.pullSecretName }}
{{- end }}
{{- end }}

{{/*
Auth login-secret env (controlplane kumo-rest only — it hosts /v0/auth/* and seeds the
direct_connect store). Delivers the OIDC client secret / password bootstrap secret to the
Kumo secrets backend via env vars of the same name it looks up (get_secrets checks an env
var of the matching name before the file mount), so no extra volume is needed. Rendered only
when auth.enabled and the corresponding Secret is provided.

INVARIANT (load-bearing): each env var NAME here MUST equal the ref's secret_key set in
deployEnvConfigJson (KUMO_OIDC_CLIENT_SECRET / KUMO_DC_TEMP_PASSWORD). get_secrets wraps a raw
env value as {name: value} (kumo io_utils/secrets.py) and the app looks up [secret_key], so
name != secret_key would resolve to nil and fail at runtime. Keep the two in sync.
(secret_key/secretKey under auth.* values is a DIFFERENT thing — the key in the operator's own
k8s Secret, used only in the secretKeyRef below to source the raw value.)
The `| default dict` guards keep this nil-safe if an operator explicitly nulls a sub-block.
*/}}
{{- define "kumo.authEnv" -}}
{{- if .Values.auth.enabled }}
{{- $oidcSecret := (.Values.auth.oidc | default dict).clientSecret | default dict }}
{{- $pwSecret := (.Values.auth.password | default dict).temporaryPassword | default dict }}
{{- if and (eq .Values.auth.method "oidc") $oidcSecret.secretName }}
- name: KUMO_OIDC_CLIENT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ $oidcSecret.secretName }}
      key: {{ $oidcSecret.secretKey }}
{{- end }}
{{- if and (eq .Values.auth.method "password") $pwSecret.secretName }}
- name: KUMO_DC_TEMP_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ $pwSecret.secretName }}
      key: {{ $pwSecret.secretKey }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Secret key env (controlplane kumo-rest AND kumo-api). The app derives THREE things
from one deployment secret (all default to secret `kumo-secret-key`, field
`credentials_encryption_key`). Production startup fails if it is absent:
  - API-key pepper (rest/driver/api_key_driver.py) — hashing/validating API keys; an explicit
    value remains authoritative in every stage so existing hashes survive upgrades.
  - password-reset token pepper (rest/driver/direct_connect_auth_storage.py) — password auth method.
  - Fernet CredentialsEncryptor key (rest/config/encryption.py) — used under a non-preprod stage
    for the OIDC login-state cookie, stored client secret, and stored connector credentials.
We deliver ONE operator-provided value to all of them via env overrides the app checks before the
secret file (get_secrets reads an env var of the matching name first), so no volume/field-matching
is needed. FERNET_SECRET_NAME==credentials_encryption_key so the Fernet lookup name==field==env name.
Rendered only when tenantSecretKey.secretName is set. Runtime-created pool pods receive the same
values through the POOL_WORKER_TENANT_SECRET_* coordinates passed to the pool controller; current
Kumo renders those coordinates as secretKeyRef-backed env on daemon and activity pods.
*/}}
{{- define "kumo.tenantSecretKeyEnv" -}}
{{- $ts := .Values.tenantSecretKey | default dict -}}
{{- if $ts.secretName }}
- name: FERNET_SECRET_NAME
  value: credentials_encryption_key
{{- range $env := (list "credentials_encryption_key" "API_KEY_PEPPER" "PASSWORD_RESET_TOKEN_PEPPER") }}
- name: {{ $env }}
  valueFrom:
    secretKeyRef:
      name: {{ $ts.secretName }}
      key: {{ $ts.secretKey }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Temporal client-mTLS volume + mount (shared across CP rest/api and DP
workers/poolcontroller). The mount path is chart-owned so values cannot diverge
from the generated deploy config.
*/}}
{{- define "kumo.temporalCertDir" -}}
/etc/temporal/certs
{{- end }}

{{/*
Static GCP ADC from a customer-created Secret. Only the selected key is exposed
and it is remapped to a chart-owned path so GOOGLE_APPLICATION_CREDENTIALS
cannot drift from the volume projection.
*/}}
{{- define "kumo.gcpApplicationCredentialsDir" -}}
/var/run/secrets/kumo/gcp
{{- end }}

{{- define "kumo.gcpApplicationCredentialsPath" -}}
{{- printf "%s/application-credentials.json" (include "kumo.gcpApplicationCredentialsDir" .) -}}
{{- end }}

{{- define "kumo.gcpApplicationCredentialsVolume" -}}
{{- with (.Values.kumoConfig.gsaKey | default dict) }}
{{- if .existingSecret }}
- name: gcp-application-credentials
  secret:
    defaultMode: 292
    secretName: {{ .existingSecret | quote }}
    items:
      - key: {{ .key | quote }}
        path: application-credentials.json
{{- end }}
{{- end }}
{{- end }}

{{- define "kumo.gcpApplicationCredentialsVolumeMount" -}}
{{- if (.Values.kumoConfig.gsaKey | default dict).existingSecret }}
- mountPath: {{ include "kumo.gcpApplicationCredentialsDir" . }}
  name: gcp-application-credentials
  readOnly: true
{{- end }}
{{- end }}

{{- define "kumo.secretsVolume" -}}
{{- if (.Values.postgres.dsnSecret | default dict).secretName }}
- name: kumo-secrets
  secret:
    defaultMode: 420
    secretName: {{ .Values.postgres.dsnSecret.secretName }}
{{- end }}
{{- with (.Values.postgres.caSecret | default dict) }}
{{- if .secretName }}
- name: kumo-statedb-ca
  secret:
    defaultMode: 420
    secretName: {{ .secretName }}
    # Remap to a fixed filename so tls.sslrootcert is a constant the operator
    # copies, not a path they compose from values.
    items:
      - key: {{ .key | default "ca.pem" }}
        path: ca.pem
{{- end }}
{{- end }}
{{- end }}

{{- define "kumo.postgresCaSecretPath" -}}
{{- printf "%s/ca/ca.pem" (include "kumo.secretsMountPath" .) -}}
{{- end }}

{{- define "kumo.secretsVolumeMount" -}}
{{- if (.Values.postgres.dsnSecret | default dict).secretName }}
- mountPath: {{ include "kumo.secretsMountPath" . }}
  name: kumo-secrets
  readOnly: true
{{- end }}
{{- with (.Values.postgres.caSecret | default dict) }}
{{- if .secretName }}
- mountPath: {{ dir (include "kumo.postgresCaSecretPath" $) }}
  name: kumo-statedb-ca
  readOnly: true
{{- end }}
{{- end }}
{{- end }}

{{- define "kumo.temporalVolume" -}}
{{- if .Values.temporal.clientMtlsSecret }}
- name: temporal-mtls
  secret:
    defaultMode: 420
    secretName: {{ .Values.temporal.clientMtlsSecret }}
{{- end }}
{{- end }}

{{- define "kumo.temporalVolumeMount" -}}
{{- if .Values.temporal.clientMtlsSecret }}
- mountPath: {{ include "kumo.temporalCertDir" . }}
  name: temporal-mtls
  readOnly: true
{{- end }}
{{- end }}

{{/*
Global pod labels used by hook pods. Regular workloads call
``kumo.workloadPodLabels``, which also adds role-configured identity labels.
*/}}
{{/*
Raw pod annotations applied to every workload and hook pod. The chart adds none
of its own here; callers emit their own alongside this include.
*/}}
{{- define "kumo.podAnnotations" -}}
{{- with .Values.global.podAnnotations -}}{{- toYaml . -}}{{- end -}}
{{- end }}

{{- define "kumo.podLabels" -}}
{{- with .Values.global.podLabels }}
{{- toYaml . }}
{{- end }}
{{- end }}

{{/*
Downward-API identity env consumed by the kumo app (kumo#29699):
POD_NAMESPACE steers K8s launches; POD_SERVICE_ACCOUNT is inherited by
pool-launched pods.
*/}}
{{- define "kumo.downwardApiEnv" -}}
- name: POD_NAMESPACE
  valueFrom: { fieldRef: { fieldPath: metadata.namespace } }
- name: POD_SERVICE_ACCOUNT
  valueFrom: { fieldRef: { fieldPath: spec.serviceAccountName } }
{{- end }}

{{/*
Node targeting applied uniformly to every workload and install hook.
*/}}
{{- define "kumo.nodeTargeting" -}}
{{- with .Values.nodeTargeting.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.nodeTargeting.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "kumo.noSurgeRollingStrategy" -}}
strategy: { type: RollingUpdate, rollingUpdate: { maxSurge: 0, maxUnavailable: 1 } }
{{- end }}

{{/*
Temporal worker liveness probe (mlworkflow, databackend)
*/}}
{{- define "kumo.temporalWorkerLiveness" -}}
livenessProbe:
  exec:
    command:
    - "/bin/sh"
    - "-c"
    - "if [ -f /opt/app/temporalio_commons/health_check.py ]; then python3 /opt/app/temporalio_commons/health_check.py --worker $WORKERS; else echo 'health check skipped'; fi"
  failureThreshold: {{ .Values.workloads.temporalWorkerLivenessProbe.failureThreshold }}
  initialDelaySeconds: {{ .Values.workloads.temporalWorkerLivenessProbe.initialDelaySeconds }}
  periodSeconds: {{ .Values.workloads.temporalWorkerLivenessProbe.periodSeconds }}
  timeoutSeconds: {{ .Values.workloads.temporalWorkerLivenessProbe.timeoutSeconds }}
{{- end }}
