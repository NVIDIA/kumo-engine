# External-service configuration

The authoritative inputs are [values.yaml](../values.yaml),
[values.schema.json](../values.schema.json), and the
[rendering helpers](../templates/_helpers.tpl). Follow them with the runtime
configuration types in `service_framework` when changing the contract.

## PostgreSQL password authentication

Set `postgres.dsnSecret.secretName` and `postgres.dsnSecret.key`; leave
`postgres.authMode` empty. The selected Secret key contains connection JSON,
not a PostgreSQL URL. Its required fields are `username`, `host`, and
`database`; supply the password and port appropriate to the server.

```json
{
  "username": "kumo",
  "password": "REPLACE_WITH_PASSWORD",
  "host": "postgres.example.com",
  "port": "5432",
  "database": "kumo",
  "tls": {
    "sslmode": "verify-full",
    "sslrootcert": "/var/run/secrets/kumo/ca/ca.pem"
  }
}
```

This is a shape example, not a credential to deploy. When a `tls` block is
present, use `verify-ca` or `verify-full`. The CA path is inside the container.
Set `postgres.caSecret.secretName` and `.key` to mount an operator-supplied CA
at `/var/run/secrets/kumo/ca/ca.pem`. Use the image's existing trust store only
when it actually trusts the selected database certificate.

For workload identity, `postgres.authMode` selects `aws_iam` or `azure_entra`;
configure `postgres.host`, `.database`, `.username`, and the operator-provisioned
identity. For `aws_iam`, also set `postgres.region`. Do not combine `authMode` with `dsnSecret`.

## Temporal

Set `temporal.address`, `.namespace`, and `.registerNamespace` for the supplied
service. `clientMtlsSecret` selects an existing `kubernetes.io/tls` Secret in
the release namespace, containing `tls.crt`, `tls.key`, and `ca.crt`. The chart mounts it read-only and the runtime reads these three
files for client mTLS. Verify the selected server trust and client identity.
This is not an assurance that every Temporal authentication mechanism is supported.

## Application and cloud identity

`global.tenant` identifies the application instance and supplies the default
Temporal namespace. Kubernetes resources install into Helm's release namespace;
the tenant string does not select it.

Use `serviceAccounts` for role-specific workload identity. The control,
worker, and controller roles have different consumers. For the GCS key fallback,
`kumoConfig.gsaKey` references an existing Secret; do not commit key JSON into
values. The chart rejects conflicting GCP identity mechanisms.

`auth` selects login mode and its Secret references. `tenantSecretKey` points
to stable operator-owned key material; keep it across upgrades. The application
chart does not create the authentication Secrets for you.

`kumoConfig.extra` overrides assembled deployment configuration. Review its
runtime consumers and rendered result carefully; it is not a substitute for
adding typed, tested public settings.
