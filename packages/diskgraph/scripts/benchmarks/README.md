# Benchmark Ingest (systemd --user)

Run the benchmark ingest loop under a per-user, repo-local `systemd --user` unit. This keeps the long-running ingest resilient (auto-restart, logs, easy status) without requiring system-wide services.

## Quickstart

Set `S3_PREFIX` to the S3 location of your benchmark datasets. The checked-in
configurations use paths relative to this location. For the user service, set
`Environment=S3_PREFIX=s3://example-bucket/benchmarks` in a service override
with `systemctl --user edit benchmark_ingest`.

From the repo root:

```bash
./scripts/benchmarks/benchmark_ingest_systemd.sh install
```

Check status and control the service:

```bash
./scripts/benchmarks/benchmark_ingest_systemd.sh status
./scripts/benchmarks/benchmark_ingest_systemd.sh restart
```

Tail logs (journal + latest benchmark log):

```bash
./scripts/benchmarks/benchmark_ingest_tail.sh
```

Stop the service:

```bash
./scripts/benchmarks/benchmark_ingest_systemd.sh stop
```

## Files

- `scripts/benchmarks/benchmark_ingest.sh` main ingest loop.
- `scripts/benchmarks/benchmark_configs/` dataset configs + `config_list.txt`.
- `scripts/benchmarks/benchmark_ingest.service` systemd unit template.
- `scripts/benchmarks/benchmark_ingest_systemd.sh` install/control helper.
- `scripts/benchmarks/benchmark_ingest_tail.sh` log tail helper.
- `benchmark/logs/` ingest logs (git-ignored).

## Storage configuration

The GitHub workflow reads `DISKGRAPH_BENCHMARK_S3_PREFIX` from repository
variables. Set it before enabling the scheduled benchmark. The local loop reads
`S3_PREFIX` from its environment. Neither path selects an internal bucket.

To prepare a configuration for data already downloaded to a local directory:

```bash
python3 scripts/benchmarks/rewrite_config.py path/to/config.json \
  --data-root /path/to/benchmark-data --output /tmp/benchmark-config.json
```

## Notes

- The ingest loop expects `config_list.txt` entries to be repo-relative paths.
- `benchmark_ingest.sh` uses `sudo` for SSD setup/reset. Ensure sudo is configured for the required commands.
- Service logs go to the user journal. Use `journalctl --user -u benchmark_ingest` to inspect.

## Troubleshooting

- `systemctl --user` fails:
  - Ensure user services are enabled for your environment.
- No logs to tail:
  - Start the service or run the ingest once so logs are created.
- Git fetch/merge failures in logs:
  - The loop continues on failure; inspect `benchmark_ingest.sh` output.
