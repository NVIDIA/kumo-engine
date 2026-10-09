# Single-Host Kumo AMI Build

## Prerequisites

- AWS SSO session active: `aws sso login`
- Packer installed: `brew install packer`
- Docker images already built and pushed to a registry configured for the AMI.
  The existing build workflows and Packer defaults still require release-owner
  configuration and artifact access; this is not yet a verified public build path.
- **kumo-base AMI** exists in the AWS account (see Step 0 — built rarely)

## Step 0: Build kumo-base AMI (one-time / when driver bumps)

The singlehost AMI sources from `kumo-base-ubuntu22-*` (owners=self) instead
of an AWS-published DLAMI. kumo-base ships only what container hosts actually
need (NVIDIA driver, Docker, nvidia-container-toolkit) – no host-side CUDA
toolkit. install-base.sh strips the DLAMI's ~40 GB host CUDA toolkit,
dropping the baked kumo-base root from ~49 GB used to ~9 GB.

You usually do **not** need to rebuild kumo-base. Rebuild only when:
- Bumping `NVIDIA_DRIVER_VERSION` (default `580-open` — see `install-base.sh`)
- Pulling in OS security patches
- Adding new tools to the base layer

To rebuild:

```bash
cd deploy/ami

# Extract SSO creds for Packer (same logic packer-build.sh uses)
eval "$(python3 -c "
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser('~/.aws/cli/cache/*.json')), key=os.path.getmtime, reverse=True):
    d = json.load(open(f))
    if 'Credentials' in d:
        c = d['Credentials']
        print(f'export AWS_ACCESS_KEY_ID={c[\"AccessKeyId\"]}')
        print(f'export AWS_SECRET_ACCESS_KEY={c[\"SecretAccessKey\"]}')
        print(f'export AWS_SESSION_TOKEN={c[\"SessionToken\"]}')
        break
")"

packer build \
  -var "aws_access_key_id=${AWS_ACCESS_KEY_ID}" \
  -var "aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}" \
  -var "aws_session_token=${AWS_SESSION_TOKEN}" \
  kumo-base.pkr.hcl
```

The smoke test inside `kumo-base.pkr.hcl` runs `nvidia-smi` in a Docker
container with `--gpus all` — fails the AMI build at Packer time if GPU
passthrough fails this smoke test. This gate does not qualify every runtime
workload or target instance type.

## Step 1: Build Docker Images

```bash
# Get the commit SHA to build from
git rev-parse HEAD

# Trigger image builds (CPU backend, GPU backend, UI)
python3 scripts/release/build_kumo_images.py <full-40-char-sha> --app kumo3 --attach
```

Run this command from the repository root with authorization to dispatch the configured GitHub Actions image workflow. Takes ~30 min. The `--attach` flag waits for completion.

**Important**: Use the full 40-character commit SHA when selecting the image-build revision.

## Step 2: Build AMI

```bash
cd deploy/ami
./packer-build.sh <full-40-char-sha>
```

The script handles all AWS credential plumbing automatically — it extracts SSO creds from `~/.aws/cli/cache/`, exports them as env vars for Packer, and passes them as `-var` flags for the build instance's ECR access.

**Prerequisite**: valid AWS SSO session (`aws sso login` + `aws sts get-caller-identity`). SSO tokens expire every few hours.

### EBS Sizing & Performance

The kumo-singlehost AMI is built with a 100 GB root (kumo-base itself is ~75 GB provisioned / ~9 GB used), leaving ~90 GB usable for Docker images + glove + build artifacts. At instance launch time, resize to 300-500GB as needed for production data.

For cold-start: provision the gp3 volume to **4000 IOPS + 1000 MB/s throughput** (the launch example below does this). Default gp3 (3000 IOPS / 125 MB/s) is bandwidth-starved during boot — Python imports across all backend containers walk through the same blocks in lockstep, gated on EBS lazy-load from the snapshot. Bumping throughput cuts cold-start from ~245s to ~135s. Cost premium: ~$0.06/hr (about 3% of a g4dn.8xlarge).

## Step 3: Launch Instance

```bash
aws ec2 run-instances \
  --image-id <ami-id> \
  --instance-type g4dn.8xlarge \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Iops":4000,"Throughput":1000}}]' \
  --metadata-options HttpTokens=optional,HttpEndpoint=enabled,HttpPutResponseHopLimit=2 \
  --user-data file://user-data.txt
```

`HttpPutResponseHopLimit=2` is **required**: Kumo containers run on a docker
bridge network and reach IMDS (`169.254.169.254`) through one extra hop. The
default IMDSv2 hop limit of 1 drops the response, breaking in-container AWS
credential loading and causing graph ingestion to fail with
`Failed to load AWS credentials`. See USER_GUIDE.md §8 for details.

Use the complete [EC2 user-data example](USER_GUIDE.md#7-create-ec2-user-data),
including `public_url`, administrator identities, and the selected login
method. Storage paths alone do not satisfy firstboot validation.

### Auto-Stop on Idle

The instance can automatically stop itself after a period of inactivity (no running or recently completed Temporal workflows). This is disabled by default. To enable:

```bash
echo "KUMO_AUTOSTOP_IDLE_MINUTES=60" > /etc/kumo/autostop.conf
```

Run history is logged to `/var/log/kumo-autostop.log`.

## Local Testing Without an AMI

To test pipeline or diskgraph changes without the 1–3 hour AMI build cycle, use `make compose-local`. It runs the identical service topology from locally-built images with `/tmp` paths instead of NVMe/host paths.

```bash
make compose-local-build
make compose-local
make compose-local-logs
make compose-local-down
```

Custom diskgraph data directory:

```bash
LOCAL_DISKGRAPH_DATA=/fast/ssd make compose-local
```

See `docker/compose/README.md` for full details.

## Images Required

| Image | Source | Notes |
|-------|--------|-------|
| `kumo/kumo-backend-cpu` | `Dockerfile.kumo_backend_cpu` | REST (8000/8001), data + ML temporal workers, pool controller |
| `kumo/kumo-backend-gpu` | `Dockerfile.kumo_backend_gpu` | Pool worker (spawns kumo-auto/kumo-pred GPU subprocesses) |
| `kumo/kumo-ui` | `Dockerfile.ui` | React static build + nginx |
| `temporalio/server:1.31.1`, `temporalio/admin-tools:1.31.1` | Upstream Temporal images | Defaults can be overridden during AMI bake |
| `prom/prometheus:v2.53.4` | Public | Metrics collection |
| `jaegertracing/all-in-one:1.76.0` | Public | Trace collection |
| `nginx:1.29.7` | Public | Reverse proxy / TLS termination |
| `postgres:16-alpine` | Public | Bundled StateDB + Temporal persistence |

## Compose File Structure

The singlehost stack uses three compose files. **All three are required** for bundled-Postgres production GPU deployments. Run from `/opt/kumo/compose`, where firstboot has generated the `.env` that supplies the required `KUMO_IMAGE_TAG` and `POSTGRES_PASSWORD`:

```bash
docker compose \
  -f docker-compose.singlehost.yml \
  -f docker-compose.singlehost.postgres.yml \
  -f docker-compose.singlehost.gpu.yml \
  up -d
```

`docker-compose.singlehost.yml` defines the core services. `docker-compose.singlehost.postgres.yml` adds bundled Postgres and the Postgres health dependencies. `docker-compose.singlehost.gpu.yml` adds the NVIDIA device reservation to the pool worker. Omitting the Postgres overlay is intended for external/BYO Postgres after firstboot has stored the StateDB DSN in mounted secrets. Omitting the GPU overlay starts all services but disables GPU access for train/predict activities.

The boot, restart, and watchdog scripts in `deploy/shared/scripts/` already include all three files.

For BYO Postgres, provide these structured metadata keys at first boot:

```yaml
postgres_mode: external
postgres_host: postgres.example.com
postgres_port: 5432
postgres_user: kumo
postgres_password_file: /opt/kumo/secrets/customer-postgres-password
state_db_name: kumo
temporal_db_name: temporal
temporal_visibility_db_name: temporal_visibility
postgres_sslmode: verify-full
postgres_sslrootcert: /opt/kumo/secrets/postgres-ca.pem
# Optional for private mirrored images:
temporal_server_image: registry.example.com/temporalio/server:1.31.1
temporal_admin_tools_image: registry.example.com/temporalio/admin-tools:1.31.1
```

For air-gapped builds that bake mirrored Temporal images into the AMI, pass the
same image names to Packer:

```bash
packer build \
  -var temporal_server_image=registry.example.com/temporalio/server:1.31.1 \
  -var temporal_admin_tools_image=registry.example.com/temporalio/admin-tools:1.31.1 \
  -var kumo_image_tag=<commit-sha> \
  kumo-singlehost.pkr.hcl
```

The external Postgres deployment must already contain the StateDB, Temporal,
and Temporal visibility databases and grants. In external mode, firstboot
writes structured connection secrets for validation, passes Temporal the stock
Postgres environment variables expected by the upstream Temporal images, and
omits the bundled Postgres compose overlay.

Do not put raw DSNs in the steady-state compose environment. Firstboot validates
the structured bootstrap inputs and writes mounted StateDB secrets. TLS
verification supports `postgres_sslmode: verify-ca|verify-full` and optional
`postgres_sslrootcert`.

Stock Temporal's Postgres driver still requires the server certificate to be
valid for `postgres_host` when a CA file is configured. Use a Postgres endpoint
whose certificate includes that hostname for both `verify-ca` and
`verify-full`.

Firstboot runs a BYO Postgres preflight before writing the final compose env.
The probe connects to the StateDB, Temporal, and Temporal visibility DSNs with
`psql`, creates/inserts/updates/deletes/drops a scratch table, and fails before
service startup if any target database is missing or lacks usable grants.

## Configuration

Deploy config is rendered at first boot from template:
- Template: `/opt/kumo/configs/deploy_env_config.template.json`
- Rendered: `/etc/kumo/deploy_env_config.json`

Deployment values come from EC2 user-data (simple `key: value` pairs). See `deploy/shared/scripts/kumo-firstboot.sh`.

| Key | Default | Description |
|-----|---------|-------------|
| `dataplane_root` | `/data/kumo/dataplane` | Kumo internal state storage. S3 path (`s3://...`) or local EBS path. |
| `diskgraph_data_root` | `/mnt/localssd` | NVMe SSD path for diskgraph index |
| `enable_tls` | `false` | Set to `true` to enable self-signed TLS on port 443. Leave unset or `false` when TLS is handled by your load balancer. |

## Monitoring Build Progress

### Docker image builds
```bash
# Check the consolidated backend/UI image workflow
gh run list --workflow _docker-kumo3-images-custom.yml --limit 1 \
  --json name,status,conclusion \
  --jq '.[0] | "\(.name): \(.status) \(.conclusion)"'
```

### Packer AMI build
```bash
# Check AMI status (after packer reports the AMI ID)
aws ec2 describe-images --image-ids <ami-id> \
  --query 'Images[0].{State:State,Name:Name}' --output table
```

## Identifying AMIs

Each AMI is named `kumo-singlehost-<image-tag>` by default and tagged with metadata:

- **Name**: `kumo-singlehost-<suffix>` — the Docker image tag unless Packer's `ami_name_tag` overrides it; this does not identify the deployment-script revision. Record that revision separately.
- **ImageTag** (AWS tag): the Docker image tag baked into the AMI
- **On a running instance**: `cat /etc/kumo/image_tag` shows the Docker image tag

```bash
# List AMIs with tags
aws ec2 describe-images --owners self \
  --filters "Name=name,Values=kumo-singlehost-*" \
  --query 'Images[*].{AMI:ImageId,Name:Name,Created:CreationDate,ImageTag:Tags[?Key==`ImageTag`].Value|[0]}' \
  --output table
```

## Managing AMIs and Instances

```bash
# List all single-host AMIs
aws ec2 describe-images --owners self \
  --filters "Name=name,Values=kumo-singlehost-*" \
  --query 'Images[*].{AMI:ImageId,Name:Name,Created:CreationDate,State:State}' \
  --output table

# List running single-host instances
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=kumo-singlehost*" "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].{ID:InstanceId,IP:PublicIpAddress,Type:InstanceType,Launch:LaunchTime,Profile:IamInstanceProfile.Arn}' \
  --output table

# Terminate a test instance
aws ec2 terminate-instances --instance-ids <instance-id>

# Deregister an old AMI (to allow rebuilding with same tag)
aws ec2 deregister-image --image-id <ami-id>

# Attach IAM instance profile (for S3 access)
aws ec2 associate-iam-instance-profile \
  --instance-id <instance-id> \
  --iam-instance-profile Name=YOUR_INSTANCE_PROFILE
```

**Remember to terminate test instances when done**; running instances incur AWS charges.

## TLS Certificate

When `enable_tls: true`, a self-signed certificate is generated at first boot (`/etc/kumo/tls/server.crt` and `server.key`). To replace with a customer-provided certificate:

```bash
# Copy your cert and key
sudo cp /path/to/your/cert.pem /etc/kumo/tls/server.crt
sudo cp /path/to/your/key.pem /etc/kumo/tls/server.key
sudo chmod 600 /etc/kumo/tls/server.key

# Restart nginx to pick up the new cert
docker restart kumo-nginx
```

Alternatively, place the cert files at `/etc/kumo/tls/` before first boot (e.g., via user-data or config management) — the firstboot script will skip self-signed generation if both `server.crt` and `server.key` already exist.

## Troubleshooting

### Packer build fails with "No valid credential sources"
Refresh SSO: `aws sso login`

### Docker pull fails during AMI build
For ECR images, verify login to the AMI's configured registry, substituting its region and account: `aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com`

### NVMe not mounted after instance boot
The systemd unit `kumo-mount-nvme.service` mounts NVMe before `docker.service`.
Check:

- `systemctl status kumo-mount-nvme.service`
- `journalctl -b -u kumo-mount-nvme.service --no-pager`
- `/var/log/cloud-init-output.log`
