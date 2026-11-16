# Install Kumo on one Azure GPU VM

Deploy Kumo on one Azure T4 VM with ADLS Gen2 as its data plane and Databricks
as its source, using the companion `docker-compose.yml`.

This is a reference deployment. The checked-in Compose file currently pins
organization-specific Kumo images; public image references remain pending.
Use a release-provided Compose file with images you can pull before following
these steps.

## 1. Prepare Azure

Ask your Azure administrator to provide:

- **VM:** Ubuntu 22.04 on `Standard_NC16as_T4_v3`; NVIDIA driver 580+, NVIDIA
  Container Toolkit, Docker Engine, Docker Compose v2, `curl`, and `openssl`.
  Your user must have `sudo` access and run `docker` without `sudo`.
- **Storage:** At least 128 GiB of Premium SSD space for the OS and Docker, plus
  an XFS or ext4 scratch disk at `/mnt/localssd`. Size it to at least three
  times the largest source snapshot, with more space for concurrent snapshots.
- **Network:** Allow inbound TCP 22 and 443 from the operator network. Allow
  outbound DNS and TCP 443 to NGC, Docker Hub or its approved mirror, Microsoft
  Entra ID, ADLS Blob and DFS endpoints, and Databricks. Containers must reach
  Azure IMDS at `169.254.169.254:80`.
- **ADLS:** A writable ADLS Gen2 filesystem and a dedicated user-assigned
  managed identity. Assign the identity to this single-purpose Kumo VM. Grant
  it `Storage Blob Data Contributor` on the filesystem and
  `Storage Blob Delegator` on the storage account.
- **HTTPS:** A DNS name for the VM, its PEM full-chain certificate, and the
  matching unencrypted PEM private key. The VM and operator browser must trust
  its issuing CA.

This deployment uses password login. VM and Docker administrators can read
passwords in `.env` and container environment variables.

On the VM, run:

```bash
(
  set -euo pipefail
  nvidia-smi
  docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'
  docker compose up --help | grep -q -- '--wait-timeout'
  test "$(findmnt -n -o SOURCE -T /mnt/localssd)" != "$(findmnt -n -o SOURCE -T /)"
  findmnt -n -o FSTYPE -T /mnt/localssd | grep -Eq '^(xfs|ext4)$'
)
```

Stop if this fails; the block does not create or format a disk.
If Docker reports permission denied for `/var/run/docker.sock`, run
`sudo usermod -aG docker "$USER"`, disconnect, and reconnect to the VM.

## 2. Configure Kumo

Place both files in one directory, run all remaining commands there, and do not
edit `docker-compose.yml`.

Create `.env`; the two passwords are generated automatically:

```bash
(
  set -euo pipefail
  test ! -e .env || { echo 'ERROR: .env already exists.' >&2; exit 1; }
  read -rp 'Kumo DNS hostname (no scheme/path; for example kumo.example.com): ' KUMO_DNS_NAME
  read -rp 'Initial admin email(s), comma-separated: ' ADMIN_EMAILS
  read -rp 'ADLS storage account: ' ADLS_ACCOUNT
  read -rp 'ADLS Gen2 filesystem/container name: ' ADLS_FILESYSTEM
  read -rp 'User-assigned identity client ID (not object ID): ' AZURE_CLIENT_ID
  umask 077
  cat > .env <<EOF
PUBLIC_URL=https://${KUMO_DNS_NAME}
ADMIN_EMAILS=${ADMIN_EMAILS}
AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
DATAPLANE_ROOT=abfss://${ADLS_FILESYSTEM}@${ADLS_ACCOUNT}.dfs.core.windows.net/kumo

DIRECT_CONNECT_ADMIN_TEMP_PASSWORD=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 32)
EOF
)
```

Copy the TLS files into the fixed locations and validate them:

```bash
(
  set -euo pipefail
  read -rp 'PEM full-chain certificate path: ' TLS_CERT_SOURCE
  read -rp 'Unencrypted PEM private-key path: ' TLS_KEY_SOURCE
  install -d -m 0700 tls
  sudo install -o "$(id -u)" -g "$(id -g)" -m 0644 \
    "$TLS_CERT_SOURCE" tls/server.crt
  sudo install -o "$(id -u)" -g "$(id -g)" -m 0600 \
    "$TLS_KEY_SOURCE" tls/server.key
  PUBLIC_URL="$(sed -n 's/^PUBLIC_URL=//p' .env)"
  PUBLIC_HOST="${PUBLIC_URL#https://}"
  openssl x509 -in tls/server.crt -noout -checkend 86400
  openssl x509 -in tls/server.crt -noout -checkhost "$PUBLIC_HOST"
  cmp <(openssl x509 -in tls/server.crt -pubkey -noout \
    | openssl pkey -pubin -outform DER) \
    <(openssl pkey -in tls/server.key -pubout -outform DER)
)
```

## 3. Start Kumo

With an NGC API key whose account has pull access to
`nvcr.io/<organization>/<team>` (replace this placeholder with the namespace
used by your Compose images), run:

```bash
(
  set -euo pipefail
  read -rsp 'NGC API key: ' NGC_API_KEY; echo
  printf '%s' "$NGC_API_KEY" \
    | docker login nvcr.io -u '$oauthtoken' --password-stdin
  trap 'docker logout nvcr.io >/dev/null 2>&1 || true' EXIT

  docker compose pull
  docker compose up -d --wait --wait-timeout 600
)
```

Verify the deployment:

```bash
docker compose ps --all
docker compose exec -T kumo-pool-worker nvidia-smi

docker compose exec -T kumo-worker-data python3 - <<'PY'
import posixpath
import uuid
from io_utils import fsutil
from service_framework.deploy_env_config import deploy_env

root = deploy_env().data_plane_storage_root
directory = posixpath.join(root, f".kumo-check-{uuid.uuid4().hex}")
probe = posixpath.join(directory, "probe.txt")
fsutil.makedirs(directory)
try:
    fsutil.write_text(probe, "ok")
    assert fsutil.read_text(probe) == "ok"
    assert probe in fsutil.listdir(directory)
    fsutil.generate_directory_read_sas(directory)
finally:
    fsutil.rm(directory, ignore_errors=True)
print("ADLS check passed")
PY

PUBLIC_URL="$(sed -n 's/^PUBLIC_URL=//p' .env)"
PUBLIC_HOST="${PUBLIC_URL#https://}"
curl --fail --noproxy '*' --resolve "${PUBLIC_HOST}:443:127.0.0.1" \
  "$PUBLIC_URL/" -o /dev/null
```

`docker compose ps --all` must show `kumo-bootstrap`, `temporal-schema`, and
`temporal-namespace` as `Exited (0)` and every other service as running.

On startup failure, inspect `docker compose logs --tail=200`. If `.env` must
change before data is loaded, run `docker compose down -v`, edit `.env`, and
repeat step 3. This deletes the local database, secrets, and generated config.

From an operator workstation, open `PUBLIC_URL` without a certificate warning.
Sign in with an `ADMIN_EMAILS` address and the
`DIRECT_CONNECT_ADMIN_TEMP_PASSWORD` from `.env`, then change the password.

## 4. Connect Databricks

Ask your Databricks administrator for the workspace URL, service-principal
client ID, Databricks-generated OAuth M2M secret value, SQL warehouse ID,
catalog, schema, and readable source table. They must add the principal to the
workspace with `Databricks SQL access`, then grant:

- `CAN USE` on the warehouse.
- `USE CATALOG` on the catalog, `USE SCHEMA` on the schema, and `SELECT` on the
  source table.

In Kumo:

1. Create or select a project; note its group.
2. **Admin > Identities > Add identity:** choose Databricks; enter a name, the
   workspace URL without `/api`, OAuth credentials, and project group; click
   **Add**.
3. **Connectors > Add connector:** choose Databricks and the identity; enter a
   name, warehouse ID, catalog, and schema; click **Add**.
4. **Tables > Add table:** enter a Kumo table name, choose the connector and
   source table, confirm the preview loads, and click **Add**.
