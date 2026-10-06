# StateDB SQLite devserver setup

Local devserver deployments use SQLite for StateDB metadata. Production
deployments use PostgreSQL; the SQLite backend is restricted to non-production
deployments by [the runtime guard](../../statedb/db/config.py).

## How it works

- The schema inventory and index definitions live in `statedb/db/schema.py`.
- `statedb/db/bootstrap.py` creates and validates the schema at startup using
  explicit SQLite DDL.
- REST and ML processes share one database at `<db-storage-root>/state.sqlite`.
- The database persists across devserver sessions until the devserver state is
  reset.

## SQLite console

Install the `sqlite3` command-line tool if needed. On Ubuntu:

```bash
sudo apt install sqlite3
```

For a coordinated child, use its `SQLite DB` tmux pane, which inherits that
child’s environment. From another shell, use
`make devserver-env DEVSERVER_ID=<child-id>` to obtain its `KUMO_HOME`; pass
`DEVSERVER_KUMO_HOME` if launch used a custom coordinator root. Verify
`<child-home>/db/state.sqlite` exists, then run:

```bash
STATE_DB_PATH=/verified/child/home/db/state.sqlite make sqlite
```

The console script uses `STATE_DB_PATH`, then `DB_STORAGE_ROOT/state.sqlite`,
then the caller’s deployment configuration. Plain `make sqlite` does not
discover the active child, and `make devserver-env` does not export its database
root. The script prints the selected path before opening SQLite in line mode.

For example:

```sql
SELECT COUNT(*) FROM tables;
```

## Resetting local state

See the [devserver reset warning](../devserver.md#resetting-local-state) before
removing state. StateDB recreates its database on startup after the selected
state is removed.

If the console reports an import or configuration error, initialize the
worktree environment using the [setup guide](setup.md) and verify that the devserver
deployment environment variables are present.
