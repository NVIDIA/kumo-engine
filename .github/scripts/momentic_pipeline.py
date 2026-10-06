#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Momentic local-env setup plus public-API job orchestration.

Four subcommands:

* ``momentic-env <config_file> <base_url>`` — inject a timestamped 'local'
  environment into momentic.config.yaml and sync the epoch suffix across the
  API config(s). (Bare ``<config_file> <base_url>`` is still accepted and maps
  to this subcommand for backwards compatibility.)

* ``trigger-jobs <base_url>`` — end to end against one shared graph:
  resolve the graph (GET /graphs/<id>), for each config create a predictive query
  and launch a training job. Writes the launched job ids to the state file for
  the polling step.

* ``trigger-prediction-jobs <base_url>`` — launch one batch-prediction job per
  bp_fixtures config, reusing the training jobs ``trigger-jobs`` just created
  (matched by config file name) and a fresh snapshot of the shared graph.
  Appends the launched prediction-job ids to the state file.

* ``poll-jobs <base_url>`` — read the launched job ids from the state file and
  poll each to a terminal state, dumping the backend failure reason (and, with
  ``--chase-train-table``, the gen-train-table sub-job) for any that FAILED.
  Exits non-zero if any job failed or timed out.

NOTE: a batch-prediction fixture must share the exact file name of the training
config it predicts against (e.g. job_config_fixtures/multi_class.json <->
job_config_fixtures/bp_fixtures/multi_class.json) — that name is how
trigger-prediction-jobs pairs a BP job with its training job.
"""

import argparse
import json
import os
import posixpath
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Training configs live under job_config_fixtures/, batch-prediction configs
# under bp_fixtures/; both are discovered at runtime (drop in a *.json).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_CONFIG_DIR = "job_config_fixtures"
BP_CONFIG_DIR = "bp_fixtures"

# Where setup hands the pquery ids to trigger. Steps in one GitHub Actions job
# share the filesystem, so a temp file under RUNNER_TEMP survives between steps.
DEFAULT_STATE_FILE = os.path.join(os.environ.get("RUNNER_TEMP", SCRIPT_DIR),
                                  "momentic_api_jobs.json")

# Every epoch occurrence appears as an underscore-prefixed run of >=10 digits
# (e.g. FINANCIAL_1781333580003, TRANS_1781333580003.*).
EPOCH_RE = re.compile(r"(?<=_)\d{10,}")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _api_config_dir():
    return os.path.join(SCRIPT_DIR, API_CONFIG_DIR)


def _config_path(name):
    return name if os.path.isabs(name) else os.path.join(
        _api_config_dir(), name)


def _load_config(name):
    with open(_config_path(name)) as f:
        return json.load(f)


def _api_config_files():
    """Sorted *.json under job_config_fixtures/ (excludes bp_fixtures/)."""
    d = _api_config_dir()
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f.endswith(".json") and os.path.isfile(os.path.join(d, f)))


def _bp_config_dir():
    return os.path.join(_api_config_dir(), BP_CONFIG_DIR)


def _bp_config_files():
    """Sorted *.json under bp_fixtures/."""
    d = _bp_config_dir()
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".json"))


def _load_bp_config(name):
    with open(os.path.join(_bp_config_dir(), name)) as f:
        return json.load(f)


def _request(method, url, api_key=None, body=None):
    """Issue one HTTP request, returning (status_code, response_text). HTTP
    errors are returned (not raised); only transport failures abort.
    """
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except urllib.error.URLError as e:
        raise SystemExit(f"{method} {url} failed: {e.reason}")


def _post(base, path, api_key, body, accept_extra=None):
    """POST to base+path. Accept 2xx (and optionally accept_extra, e.g. an
    idempotent 400); on any other status, print the API error and exit.
    """
    status, text = _request("POST", base + path, api_key, body)
    if not (200 <= status < 300 or status == accept_extra):
        print(f"POST {path} -> HTTP {status}", file=sys.stderr)
        print(text, file=sys.stderr)
        raise SystemExit(1)
    return status, text


def _api_key():
    return os.environ.get("PUBLIC_API_KEY") or None


def _write_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _read_state(path):
    """Load the state file if it exists, else an empty dict."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Subcommand: momentic-env
# --------------------------------------------------------------------------- #
def sync_api_config_epoch(epoch):
    """Rewrite the epoch suffix in every API config (training + bp_fixtures) so
    they reference the same connector/table names the Momentic UI test creates.
    """
    paths = (
        [_config_path(name) for name in _api_config_files()] +
        [os.path.join(_bp_config_dir(), name) for name in _bp_config_files()])
    for path in paths:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found; skipping epoch sync",
                  file=sys.stderr)
            continue

        with open(path) as f:
            text = f.read()

        old = sorted(set(EPOCH_RE.findall(text)))
        new_text = EPOCH_RE.sub(epoch, text)

        with open(path, "w") as f:
            f.write(new_text)

        print(f"Synced {path} epoch -> {epoch} (replaced {old or '<none>'})")


def cmd_momentic_env(args):
    import yaml  # only this subcommand needs it; setup/trigger stay stdlib-only

    config_file, base_url = args.config_file, args.base_url
    epoch = str(int(time.time()))

    sync_api_config_epoch(epoch)

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    local_env = {
        "name": "local",
        "baseUrl": base_url,
        "envVariables": {
            "connector_name": f"FINANCIAL_{epoch}",
            "bg_connector_name": f"BIGQ_CONNECTOR_TEST_{epoch}",
            "snow_connector_name": f"SNOW_CONNECTOR_TEST_{epoch}",
            "DIMENSION_TABLE_1": f"ORDERS_{epoch}",
            "FACT_TABLE_1": f"TRANS_{epoch}",
            "FACT_TABLE_2": f"ACCOUNT_{epoch}",
            "GRAPH_NAME": f"FINANCIAL_GRAPH_{epoch}",
            "QUERY_NAME": f"FINANCIAL_BINARY_{epoch}",
            "BP_TAG": f"FINANCIAL_BINARY_PRED_{epoch}",
            "MULTICLASS_QUERY_TAG":
            f"FINANCIAL_MULTICLASS_CLASSIFICATION_{epoch}",
            "TEMP_REG_QUERY_TAG": f"FINANCIAL_TEMP_REG_{epoch}",
            "MULTILABEL_QUERY_TAG": f"FINANCIAL_MULTILABEL_{epoch}",
            "STATIC_REG_QUERY_TAG": f"FINANCIAL_STATIC_REG_{epoch}",
            "STATIC_LP_QUERY_TAG": f"FINANCIAL_STATIC_LP_{epoch}",
            "MULTICLASS_BP_TAG": f"FINANCIAL_MULTICLASS_PRED_{epoch}",
            "MULTILABEL_BP_TAG": f"FINANCIAL_MULTILABEL_PRED_{epoch}",
            "STATIC_LP_BP_TAG": f"FINANCIAL_STATIC_LP_PRED_{epoch}",
            "STATIC_REG_BP_TAG": f"FINANCIAL_STATIC_REG_PRED_{epoch}",
            "TEMPORAL_LP_BP_TAG": f"FINANCIAL_TEMPORAL_LP_PRED_{epoch}",
            "TEMPORAL_REG_BP_TAG": f"FINANCIAL_TEMPORAL_REG_PRED_{epoch}",
            "SNOWFLAKE_BINARY_CLASSIFICATION_BP_TAG":
            f"SNOWFLAKE_BINARY_CLASSIFICATION_PRED_{epoch}",
            "SNOW_CONNECTOR_PASSWORD": "${SNOW_CONNECTOR_PASSWORD}",
            "SNOW_CONNECTOR_USER": "${SNOW_CONNECTOR_USER}",
            "service_ac_key": "${SERVICE_AC_KEY}",
            "DB_CLIENT_ID": "${DB_CLIENT_ID}",
            "DB_CLIENT_SECRET": "${DB_CLIENT_SECRET}",
        },
    }

    envs = cfg.setdefault("environments", [])
    for i, env in enumerate(envs):
        if env.get("name") == "local":
            envs[i] = local_env
            break
    else:
        envs.append(local_env)

    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)

    graph_name = local_env["envVariables"]["GRAPH_NAME"]
    # Record epoch + graph name for trigger-jobs/polling; merge so a re-run
    # keeps prior keys.
    state = _read_state(DEFAULT_STATE_FILE)
    state.update({"epoch": epoch, "graph_name": graph_name})
    _write_state(DEFAULT_STATE_FILE, state)

    print(f"Set 'local' environment: baseUrl={base_url}, "
          f"connector_name=FINANCIAL_{epoch}, "
          f"graph_name={graph_name}, "
          f"TABLES=ORDERS_{epoch} TRANS_{epoch} ACCOUNT_{epoch}")
    print(f"Wrote graph_name -> {DEFAULT_STATE_FILE}")


# --------------------------------------------------------------------------- #
# Subcommand: trigger-jobs (graph -> pquery -> job, end to end)
# --------------------------------------------------------------------------- #
def _latest_job(base, api_key, resource):
    """Return {config, job_id} for the newest job under GET /<resource> (the one
    the Momentic UI test just launched), or None. The list is newest-first.
    """
    status, text = _request("GET", f"{base}/{resource}", api_key)
    if not (200 <= status < 300):
        print(f"GET /{resource} -> HTTP {status}; no existing job to poll",
              file=sys.stderr)
        if text:
            print(text, file=sys.stderr)
        return None

    items = json.loads(text)
    if not items:
        return None

    item = items[0]
    job_id = item.get("job_id") or item.get("id")
    if not job_id:
        return None
    tags = item.get("tags") or {}
    return {
        "config": tags.get("pquery_name") or "momentic_ui_test",
        "job_id": job_id,
    }


def _ensure_connector(base, api_key, shared_graph):
    """Create the file connector the graph references (name/root_dir from its
    first source table) so source connector_ids resolve. 400 = already exists.
    """
    tables = shared_graph.get("tables", {})
    st = next((t.get("source_table") or {} for t in tables.values()), {})
    connector_id = st.get("connector_id")
    s3_path = st.get("s3_path")
    if not connector_id or not s3_path:
        print("No connector_id/s3_path in shared graph; skipping connector",
              file=sys.stderr)
        return

    root_dir = posixpath.dirname(s3_path)
    body = {
        "config": {
            "name": connector_id,
            "root_dir": root_dir,
            "type": "file"
        },
        "credentials": None,
        "description": "",
    }
    status, _ = _post(base, "/connectors", api_key, body, accept_extra=400)
    existed = " (already exists)" if status == 400 else ""
    print(f"Connector ready: {connector_id} -> {root_dir}{existed}")


def _create_graph_resource(base, api_key, graph_definition, name_alias=None):
    """POST graph_definition to /graphs, returning the resource id.

    On 422 (graph with this definition already exists), falls back to fetching
    the existing resource by name_alias so reruns don't abort.
    """
    path = "/graphs"
    if name_alias:
        path += "?name_alias=" + urllib.parse.quote(name_alias, safe="")
    status, text = _request("POST", base + path, api_key, graph_definition)
    if not (200 <= status < 300):
        print(f"POST {path} -> HTTP {status}", file=sys.stderr)
        if text:
            print(text, file=sys.stderr)
        if status == 422 and name_alias:
            return _resolve_graph_resource_id(base, api_key, name_alias)
        raise SystemExit(1)
    resource_id = json.loads(text)["id"]
    label = f" ({name_alias})" if name_alias else ""
    print(f"Created graph resource: {resource_id}{label}")
    return resource_id


def _resolve_shared_graph(base, api_key, graph_id, configs):
    """The shared graph: the API graph for graph_id, else the first config's
    graph_definition.
    """
    if graph_id:
        status, text = _request("GET", f"{base}/graphs/{graph_id}", api_key)
        if 200 <= status < 300:
            print(f"Using graph from API for all jobs: {graph_id}")
            return json.loads(text)["graph"]
        print(
            f"GET /graphs/{graph_id} -> HTTP {status}; "
            "falling back to per-config graph", file=sys.stderr)
        if text:
            print(text, file=sys.stderr)
    print("Using the first config's graph_definition as the shared graph")
    return _load_config(configs[0])["graph_definition"]


# Jobs are serialized: overlapping jobs run their Temporal activities in one
# worker process and trip a process-global temp_home_dir race that fails them
# nondeterministically. Wait for each to reach a terminal state before the next.
JOB_WAIT_DEADLINE_S = 900
JOB_WAIT_INTERVAL_S = 10


def _wait_for_terminal(base, api_key, job_id, resource="training_jobs"):
    """Poll GET /<resource>/<id> until it reaches DONE/FAILED (or the deadline),
    returning 'DONE'/'FAILED'/'TIMEOUT'. Transient errors count as still running.
    """
    deadline = time.time() + JOB_WAIT_DEADLINE_S
    while True:
        status, text = _request("GET", f"{base}/{resource}/{job_id}", api_key)
        st = "UNKNOWN"
        if 200 <= status < 300:
            try:
                st = json.loads(text)["job_status_report"]["status"]
            except (KeyError, ValueError, TypeError):
                st = "UNKNOWN"
        print(f"Job {job_id} status -> {st}")
        if st in ("DONE", "FAILED"):
            return st
        if time.time() >= deadline:
            return "TIMEOUT"
        time.sleep(JOB_WAIT_INTERVAL_S)


def _poll_latest_momentic_job(base, api_key, resource, label, persist):
    """Find the newest Momentic-UI job under /<resource> and wait for it to
    finish, returning its entry (or None). persist(job) records it before the
    wait (caller owns the state key); waiting avoids overlapping job activities.
    """
    job = _latest_job(base, api_key, resource)
    if not job:
        print(
            f"No existing {label} job found to poll; continuing to create "
            "jobs", file=sys.stderr)
        return None
    persist(job)
    job_id = job["job_id"]
    print(f"Waiting on Momentic UI test {label} job first: {job_id}")
    final_status = _wait_for_terminal(base, api_key, job_id, resource=resource)
    print(f"[momentic] {label} job {job_id} finished: {final_status}")
    return job


def cmd_trigger_jobs(args):
    """Poll the Momentic UI training job, resolve the shared graph + connector,
    then per config create a predictive query and a (serialized) training job.
    """
    base = args.base_url.rstrip("/")
    api_key = _api_key()
    configs = _api_config_files()

    prior_state = _read_state(DEFAULT_STATE_FILE)

    entries = []

    def _persist_momentic(job):
        entries.append(job)
        _write_state(DEFAULT_STATE_FILE, {
            **prior_state, "base_url": base,
            "jobs": entries
        })

    _poll_latest_momentic_job(base, api_key, "training_jobs", "training",
                              _persist_momentic)

    # graph_name was recorded by momentic-env; resolve the graph by it
    # (_resolve_shared_graph falls back to the first config's graph_definition).
    graph_id = prior_state.get("graph_name")
    if graph_id:
        print(f"Resolving shared graph by id: {graph_id}")

    shared_graph = _resolve_shared_graph(base, api_key, graph_id, configs)

    # Create the connector the graph references so its source connector_ids
    # resolve before any predictive query.
    _ensure_connector(base, api_key, shared_graph)

    shared_tables = set(shared_graph.get("tables", {}).keys())

    # Per config: predictive query + training job; use per-config graph only when tables differ from shared.
    job_ids = []
    for name in configs:
        cfg = _load_config(name)
        cfg_tables = set(cfg["graph_definition"].get("tables", {}).keys())
        uses_own_graph = cfg_tables != shared_tables
        graph_for_pq = cfg[
            "graph_definition"] if uses_own_graph else shared_graph

        pq_body = {
            "query_string": cfg["predictive_query"],
            "graph": graph_for_pq
        }
        _, text = _post(base, "/predictive_queries", api_key, pq_body)
        pquery_id = json.loads(text)["id"]
        print(f"[{name}] predictive query: {pquery_id}")

        tj_body = {
            "pquery_id": pquery_id,
            "job_tags": cfg["job_tags"],
            "model_plan": cfg["model_plan"],
        }
        _, text = _post(base, "/training_jobs", api_key, tj_body)
        job_id = json.loads(text)["id"]
        print(f"[{name}] training job: {job_id}")

        entry = {"config": name, "pquery_id": pquery_id, "job_id": job_id}
        if uses_own_graph:
            entry["uses_own_graph"] = True
        entries.append(entry)
        job_ids.append(job_id)

        # Persist after each job so the async poll step sees it even if a later
        # job or the runner dies mid-chain.
        _write_state(DEFAULT_STATE_FILE, {
            **prior_state, "base_url": base,
            "jobs": entries
        })

        # Wait before launching the next (see serialization note above).
        final_status = _wait_for_terminal(base, api_key, job_id)
        print(f"[{name}] training job {job_id} finished: {final_status}")

    # Covers the no-configs case where only the Momentic UI job was recorded.
    _write_state(DEFAULT_STATE_FILE, {
        **prior_state, "base_url": base,
        "jobs": entries
    })
    print(f"Launched {len(job_ids)} training jobs: {' '.join(job_ids)}")
    print(f"Wrote state -> {DEFAULT_STATE_FILE}")


# --------------------------------------------------------------------------- #
# Subcommand: trigger-prediction-jobs (training job -> batch prediction)
# --------------------------------------------------------------------------- #
def _resolve_graph_resource_id(base, api_key, graph_name):
    """Resolve a graph name to its public resource id (graph-<fpid>) via GET
    /graphs; /graphsnapshots needs the id, not a bare name.
    """
    status, text = _request("GET", f"{base}/graphs/{graph_name}", api_key)
    if not (200 <= status < 300):
        raise SystemExit(
            f"GET /graphs/{graph_name} -> HTTP {status}; cannot resolve graph "
            f"resource id for snapshot. {text}")
    resource_id = json.loads(text).get("id")
    if not resource_id:
        raise SystemExit(
            f"GET /graphs/{graph_name} returned no 'id' field: {text}")
    print(f"Resolved graph '{graph_name}' -> resource id {resource_id}")
    return resource_id


def _create_graph_snapshot(base, api_key, graph_id):
    """Mint a fresh snapshot of the run's graph (graph_id must be a resource id);
    BP jobs need one for this run since the bp configs carry a stale snapshot id.
    """
    path = "/graphsnapshots?graph_id=" + urllib.parse.quote(graph_id, safe="")
    _, text = _post(base, path, api_key, {})
    snapshot_id = json.loads(text)["id"]
    print(f"Created graph snapshot for {graph_id}: {snapshot_id}")
    return snapshot_id


def cmd_trigger_prediction_jobs(args):
    """Poll the Momentic UI prediction job, then launch a BP job for each
    training job with a matching bp_fixtures config (no fixture = skipped).
    """
    base = args.base_url.rstrip("/")
    api_key = _api_key()

    state = _read_state(DEFAULT_STATE_FILE)

    pred_jobs = []

    def _persist_momentic(job):
        pred_jobs.append(job)
        state["prediction_jobs"] = pred_jobs
        _write_state(DEFAULT_STATE_FILE, state)

    _poll_latest_momentic_job(base, api_key, "prediction_jobs", "prediction",
                              _persist_momentic)

    # config file name -> training job entry written by trigger-jobs.
    training_by_config = {
        j["config"]: j
        for j in state.get("jobs", []) if j.get("config") and j.get("job_id")
    }
    if not training_by_config:
        raise SystemExit("No training jobs in state file; run trigger-jobs "
                         f"first ({DEFAULT_STATE_FILE})")

    graph_id = state.get("graph_name")
    if not graph_id:
        raise SystemExit("No graph_name in state file; run momentic-env first "
                         f"({DEFAULT_STATE_FILE})")

    bp_configs = set(_bp_config_files())
    if not bp_configs:
        print(
            f"No bp_fixtures configs in {_bp_config_dir()}; nothing to "
            "predict", file=sys.stderr)
        return

    # Snapshots are minted lazily and cached by graph ref so configs that
    # trained on different graphs (e.g. static_lp) get their own snapshot.
    graph_snapshots = {}  # graph_ref -> snapshot_id

    def _get_snapshot(graph_ref):
        if graph_ref in graph_snapshots:
            return graph_snapshots[graph_ref]
        if graph_ref.startswith("graph-"):
            resource_id = graph_ref
        else:
            resource_id = _resolve_graph_resource_id(base, api_key, graph_ref)
        snapshot_id = _create_graph_snapshot(base, api_key, resource_id)
        graph_snapshots[graph_ref] = snapshot_id
        return snapshot_id

    for name, training in training_by_config.items():
        if name not in bp_configs:
            print(f"No prediction fixture for {name}; skipping",
                  file=sys.stderr)
            continue

        # For configs that trained on a distinct graph, register it as a
        # graph resource now (lazily — only when a BP fixture actually exists).
        if training.get("uses_own_graph"):
            graph_def = _load_config(name)["graph_definition"]
            graph_ref = _create_graph_resource(
                base, api_key, graph_def, name_alias=name.removesuffix(".json"))
        else:
            graph_ref = graph_id
        snapshot_id = _get_snapshot(graph_ref)

        cfg = _load_bp_config(name)
        bp_body = {
            "model_training_job_id": training["job_id"],
            "graph_snapshot_id": snapshot_id,
            "predict_options": cfg["predict_options"],
            "explanations": cfg.get("explanations", False),
            "job_tags": cfg.get("job_tags", {}),
        }
        # create_prediction_job returns (job_id, validation_response); a null
        # job id means validation rejected the request.
        _, text = _post(base, "/prediction_jobs", api_key, bp_body)
        result = json.loads(text)
        job_id = result[0] if isinstance(result, list) else result
        if not job_id:
            print(f"[{name}] prediction job not created: {result}",
                  file=sys.stderr)
            raise SystemExit(1)
        print(f"[{name}] prediction job: {job_id} (model={training['job_id']})")
        pred_jobs.append({
            "config": name,
            "model_training_job_id": training["job_id"],
            "job_id": job_id,
        })

        # Persist after each job, then wait before launching the next so only
        # one job's activities run at a time (see serialization note above).
        state["prediction_jobs"] = pred_jobs
        _write_state(DEFAULT_STATE_FILE, state)
        final_status = _wait_for_terminal(base, api_key, job_id,
                                          resource="prediction_jobs")
        print(f"[{name}] prediction job {job_id} finished: {final_status}")

    state["prediction_jobs"] = pred_jobs
    _write_state(DEFAULT_STATE_FILE, state)
    # pred_jobs also holds the polled Momentic UI job (no model_training_job_id);
    # count only the ones this step launched.
    launched = sum(1 for j in pred_jobs if j.get("model_training_job_id"))
    print(f"Launched {launched} prediction jobs")
    print(f"Wrote state -> {DEFAULT_STATE_FILE}")


# --------------------------------------------------------------------------- #
# Subcommand: poll-jobs (await launched jobs, report failures)
# --------------------------------------------------------------------------- #
def _job_status(base, api_key, resource, job_id):
    """Current status of GET /<resource>/<id>, or 'UNKNOWN' on any error."""
    status, text = _request("GET", f"{base}/{resource}/{job_id}", api_key)
    if 200 <= status < 300:
        try:
            return json.loads(text)["job_status_report"]["status"]
        except (KeyError, ValueError, TypeError):
            pass
    return "UNKNOWN"


def _poll_until_terminal(base, api_key, resource, ids, deadline_s):
    """Poll every id under /<resource> until each reaches DONE/FAILED or the
    shared deadline elapses. Returns {id: 'DONE'|'FAILED'|'TIMEOUT'}.
    """
    deadline = time.time() + deadline_s
    final = {}
    while True:
        for job_id in ids:
            if job_id in final:
                continue
            st = _job_status(base, api_key, resource, job_id)
            print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] "
                  f"{job_id} status={st}")
            if st in ("DONE", "FAILED"):
                final[job_id] = st
        if len(final) == len(ids):
            break
        if time.time() >= deadline:
            print(f"Timed out after {deadline_s // 60} minutes")
            break
        time.sleep(JOB_WAIT_INTERVAL_S)
    return final


def _print_failure_detail(report, indent="    "):
    """Print the validation errors + event log carried by a job_status_report."""
    for e in (report.get("validation_response") or {}).get("errors") or []:
        print(f"{indent}title: {e.get('title')} | message: {e.get('message')}")
    for line in report.get("event_log") or []:
        print(f"{indent}log: {line}")


def _chase_train_table(base, api_key, resp):
    """On a failed training job, dump its gen-train-table sub-job (where the root
    cause often lives), resolved from config.train_table_job_id.
    """
    gtt_id = (resp.get("config") or {}).get("train_table_job_id") or ""
    if not gtt_id:
        print("    (no train_table_job_id on training job; gen-train-table "
              "may not have started)")
        return
    print(f"    --- gen-train-table failure detail for {gtt_id} ---")
    _, text = _request("GET", f"{base}/gentraintable_jobs/{gtt_id}", api_key)
    try:
        report = json.loads(text)["job_status_report"]
    except (KeyError, ValueError, TypeError):
        print("    (failed to fetch gen-train-table detail)")
        return
    print(f"    gtt status: {report.get('status')}")
    _print_failure_detail(report)


def cmd_poll_jobs(args):
    """Poll the launched jobs recorded under args.state_key to a terminal state,
    dumping the failure reason for any that failed. Returns an exit code.
    """
    base = args.base_url.rstrip("/")
    api_key = _api_key()
    resource = args.resource

    state = _read_state(DEFAULT_STATE_FILE)
    ids = [
        j["job_id"] for j in state.get(args.state_key, []) if j.get("job_id")
    ]
    if not ids:
        msg = (f"No {args.state_key} to poll "
               f"(no job_id entries in {DEFAULT_STATE_FILE})")
        if args.allow_empty:
            print(msg + "; nothing to poll")
            return 0
        print(msg, file=sys.stderr)
        return 1
    print(f"Polling {len(ids)} {resource}: {' '.join(ids)}")

    final = _poll_until_terminal(base, api_key, resource, ids,
                                 args.deadline_seconds)

    # Jobs whose config name marks them as expected to fail (e.g. failure_job.json).
    expected_failure_ids = {
        j["job_id"]
        for j in state.get(args.state_key, [])
        if j.get("job_id") and "failure" in (j.get("config") or "")
    }

    # Fail the step if any job failed, dumping its backend failure reason.
    rc = 0
    for job_id in ids:
        st = final.get(job_id, "TIMEOUT")
        print(f"  {job_id} -> {st}")
        if st != "FAILED":
            continue
        if job_id in expected_failure_ids:
            print("    (expected failure — not failing step)")
            continue
        rc = 1
        print(f"    --- failure detail for {job_id} ---")
        _, text = _request("GET", f"{base}/{resource}/{job_id}", api_key)
        try:
            resp = json.loads(text)
            _print_failure_detail(resp["job_status_report"])
        except (KeyError, ValueError, TypeError):
            print("    (failed to fetch failure detail)")
            continue
        if args.chase_train_table:
            _chase_train_table(base, api_key, resp)
    return rc


# --------------------------------------------------------------------------- #
# Failure reporting (GitHub issue + Slack)
# --------------------------------------------------------------------------- #
def _humanize_step(step_id):
    """Turn a step id (``run_create_prediction``) into a readable label."""
    s = step_id.replace("_", " ").strip()
    return (s[:1].upper() + s[1:]) if s else step_id


def _set_output(name, value):
    """Append a step output to $GITHUB_OUTPUT (multiline-safe). No-op locally."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        if "\n" in value:
            delim = f"__EOF_{name.upper()}__"
            f.write(f"{name}<<{delim}\n{value}\n{delim}\n")
        else:
            f.write(f"{name}={value}\n")


def cmd_report_failures(args):
    """Read the GitHub Actions ``steps`` context (passed as STEPS_JSON) and, for
    any step whose outcome is 'failure', emit the failure summary as step
    outputs (failed / title / body / slack_message). The workflow opens the
    ticket and sends Slack from those — no GitHub token is handled here. Always
    exits 0; the originally failing step already marks the job red.
    """
    raw = os.environ.get("STEPS_JSON", "")
    try:
        steps = json.loads(raw) if raw else {}
    except ValueError:
        print("STEPS_JSON is not valid JSON; nothing to report.",
              file=sys.stderr)
        return 0

    failed = sorted(
        sid for sid, meta in steps.items()
        if isinstance(meta, dict) and meta.get("outcome") == "failure")
    if not failed:
        print("No failed steps; skipping ticket + Slack notification.")
        _set_output("failed", "false")
        return 0
    print(f"Detected {len(failed)} failed step(s): {', '.join(failed)}")

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_url = f"https://github.com/{repo}/actions/runs/{run_id}"

    bullets = "\n".join(f"- **{_humanize_step(s)}** (`{s}`)" for s in failed)
    title = (f"[Momentic Tests - Nightly] {len(failed)} step(s) failed "
             f"— run #{run_id}")
    body = (f"Automated failure report.\n\n"
            f"## Failed step(s)\n{bullets}\n\n"
            f"Run: {run_url}\n")

    step_lines = "\n".join(f"• {_humanize_step(s)}" for s in failed)
    slack_message = (f":x: *Momentic UI Tests* failed — {len(failed)} step(s)\n"
                     f"*Failed steps:*\n{step_lines}\n"
                     f"<{run_url}|View run>")

    # Emit the failure info as step outputs; the workflow opens the ticket
    # (with the token Actions injects) and forwards the Slack message.
    _set_output("failed", "true")
    _set_output("title", title)
    _set_output("body", body)
    _set_output("slack_message", slack_message)
    print("Wrote report outputs (failed=true).")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser(
        "momentic-env",
        help="Inject 'local' env into momentic.config.yaml + sync epoch")
    m.add_argument("config_file")
    m.add_argument("base_url")
    m.set_defaults(func=cmd_momentic_env)

    t = sub.add_parser("trigger-jobs",
                       help="graph -> pquery -> training job per config, E2E")
    t.add_argument("base_url", help="e.g. http://localhost:10002/v1")
    t.set_defaults(func=cmd_trigger_jobs)

    pp = sub.add_parser(
        "trigger-prediction-jobs",
        help="batch-prediction job per bp config, reusing trigger-jobs' "
        "training jobs")
    pp.add_argument("base_url", help="e.g. http://localhost:10002/v1")
    pp.set_defaults(func=cmd_trigger_prediction_jobs)

    po = sub.add_parser(
        "poll-jobs",
        help="poll launched jobs to a terminal state + report failures")
    po.add_argument("base_url", help="e.g. http://localhost:10002/v1")
    po.add_argument(
        "--resource", default="training_jobs",
        help="REST resource to poll (training_jobs|prediction_jobs)")
    po.add_argument("--state-key", default="jobs",
                    help="state-file key holding the job entries to poll")
    po.add_argument("--deadline-seconds", type=int, default=720,
                    help="give up polling after this many seconds")
    po.add_argument("--allow-empty", action="store_true",
                    help="exit 0 (not 1) when no jobs are recorded to poll")
    po.add_argument("--chase-train-table", action="store_true",
                    help="on failure, also dump the gen-train-table sub-job")
    po.set_defaults(func=cmd_poll_jobs)

    rf = sub.add_parser(
        "report-failures",
        help="open a GitHub issue + Slack ping for any failed step "
        "(reads the steps context from STEPS_JSON)")
    rf.set_defaults(func=cmd_report_failures)

    return p


def main():
    argv = sys.argv[1:]
    known = {
        "momentic-env", "trigger-jobs", "trigger-prediction-jobs", "poll-jobs",
        "report-failures"
    }
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv = ["momentic-env", *argv]
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
