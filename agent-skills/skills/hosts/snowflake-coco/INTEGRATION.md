# Integration — Snowflake CoCo

For the people deploying the skill, not for the agent; the skill never points here.

---

## 10. Still needs the CoCo team

Not answerable from anything on hand. Do not guess.

**Blocking (answer these first):**

- **Is Kumo present in the CoCo runtime — does `import kumo.engine` succeed?** If not, this path is dead (`runtime.md` §0, §1.1).
- **Does the CoCo runtime expose `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_HOST`, and `/snowflake/session/token`?** A Snowpark session alone is explicitly insufficient. If no, generated code falls back to `connections.toml`, then to a key pair from secrets (`runtime.md` §0, §3).
- **Which paths in a CoCo session are writable, and what is backed by fast local storage?** (`runtime.md` §1.2)

**Non-blocking but required before shipping:**

- CoCo skill packaging format and manifest schema.
- Notebook/runtime compute and GPU selection available to a CoCo session, and whether a GPU is visible to the Python process (`runtime.md` §2).
- Lifetime of CoCo session storage across restarts — determines how aggressively results must be flushed to Snowflake (`runtime.md` §3.1).
- Whether CoCo imposes conventions on the prediction output table.
- CoCo-level context window, token budget, latency, and rate limits.
- Long-running training, for the opt-in Run step only: whether a CoCo session stays alive for the whole of a blocking `fit`, which can run for hours. There is no job to poll; if the session cannot outlive training, the Run step is not viable on this host.

**Needed from Kumo (not CoCo):**

- The GRANT set for this path (`runtime.md` §6).
- Validated GPU/memory/storage sizing for this path (`runtime.md` §2).


---
## 11. Integration checklist

- [ ] `import kumo.engine` succeeds in the CoCo runtime; version printed.
- [ ] Generated code begins at `import kumo.engine`; no environment-setup step anywhere in it.
- [ ] Scratch and artifact directories exist and are writable.
- [ ] One authentication option available, in priority order: the SPCS token (`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_HOST`, `/snowflake/session/token`), a `connections.toml` entry, or key-pair secrets attached to the notebook.
- [ ] `kumo.configure(scratch_dir=...)` called once, before any `Graph` or `Trainer`.
- [ ] Connector built by `from_spcs_env()`, else `from_connection_toml()`, else `SnowflakeIdentity` from secrets — with explicit warehouse, database, schema_name.
- [ ] `connector.table_names()` asserted — then separately verify the output write.
- [ ] No call from `kumoai-reference/references/kumo_finetuning_sdk.md` §11.1 appears anywhere.
- [ ] No credential literal anywhere in generated code.
- [ ] Every result the user needs later is written to a durable Snowflake destination, not left in artifacts.
- [ ] Preflight SQL (`adapter.md` §6.1) runs against the user's `source_scope` from a CoCo session.
- [ ] End-to-end test: run the canonical good example from `validate/examples/good_request.md` unmodified in the target environment.
