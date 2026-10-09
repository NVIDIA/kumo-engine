# Monitoring

## Setup

Ensure the following default ports are exposed as inbound rules in the EC2 instance "Security" tab:

| Service | Port | Notes |
|---------|------|-------|
| Grafana (system monitoring) | 3000 | |
| Prometheus (metrics web UI) | 9090 | Only if web UI for system metrics is needed |
| Jaeger (tracing) | 16686 | |

Custom ports may be selected by modifying the port variables (`*_PORT`) in `monitoring.sh`.

Once the proper security rules have been configured, run from this directory:

```bash
# Point to a compatible local monitoring bundle.
export KUMO_GRAFANA_HOME=/path/to/monitoring-bundle
./monitoring.sh
```

If the bundle is not already present, set `KUMO_GRAFANA_REPO_URL` and the
script will clone that repository before starting it.

Use the `-h` flag to see documentation for advanced options.
