# Benchmark Ingest (systemd --user)

Run the benchmark ingest loop under a per-user, repo-local `systemd --user` unit. This keeps the long-running ingest resilient (auto-restart, logs, easy status) without requiring system-wide services.

## Quickstart

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
