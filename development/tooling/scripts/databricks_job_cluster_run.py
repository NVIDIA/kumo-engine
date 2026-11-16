#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job cluster keep-alive and monitoring script.

This script is deployed to S3 (s3://kumo-databricks-public/databricks-jars/run.py)
and runs as `spark_python_task` on Databricks job clusters. Its purpose is to:

1. **Keep the job cluster alive** — without this script, the Databricks job
   run would complete immediately and the cluster would be terminated.
2. **Monitor Spark activity** — queries the Spark UI via the Databricks
   driver-proxy-api to detect jobs from Spark Connect clients.
3. **Update `last_used_time`** — writes activity timestamps to the cluster
   info cache file so the session manager can detect idle clusters for
   pool eviction.
4. **Self-terminate on idle** — exits after `JOB_CLUSTER_IDLE_TIMEOUT_SECONDS`
   of no Spark activity, which causes the Databricks run to complete and the
   cluster to be terminated.
5. **Enforce max lifetime** — exits after `JOB_CLUSTER_MAX_LIFETIME_SECONDS`
   regardless of activity.

Uses the Databricks driver-proxy-api to query the Spark UI REST API, which
provides visibility into all Spark jobs on the cluster (including those from
Spark Connect clients).

When Spark activity is detected, updates `last_used_time` in the cluster
info JSON file. The session manager reads this to determine if a pooled
cluster is idle and should be terminated.

The cache file is on a Unity Catalog volume (dbfs:/Volumes/catalog/schema/volume/).

No CLI parameters needed - everything is auto-detected:
    - cluster_id: from DATABRICKS_CLUSTER_ID env or Spark config
    - Cache path: {KUMO_CACHE_DIR}/databricks_job_clusters/{cluster_id}
    - Credentials: from DATABRICKS_HOST, DATABRICKS_TOKEN or
      DATABRICKS_CLIENT_ID/SECRET env vars
    - Max lifetime: from JOB_CLUSTER_MAX_LIFETIME_SECONDS env var (default 86400s / 24h)
    - Idle timeout: from JOB_CLUSTER_IDLE_TIMEOUT_SECONDS env var (default 1800s / 30min)
    - Poll interval: 60s
"""

import json
import os
import time
from datetime import datetime, timezone


def get_cluster_id() -> str:
    """Get the current cluster's ID from env or Spark config."""
    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID")
    if cluster_id:
        return cluster_id

    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
    except Exception:
        pass

    raise RuntimeError("Could not determine cluster ID")


def get_proxy_config() -> tuple:
    """Get workspace URL and Spark UI port for the driver-proxy-api.

    Returns:
        (workspace_url, spark_ui_port) tuple
    """
    workspace_url = os.environ.get("DATABRICKS_HOST", "").rstrip("/") or None
    spark_ui_port = "40001"

    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        if not workspace_url:
            url = spark.conf.get("spark.databricks.workspaceUrl")
            if url:
                if not url.startswith("https://"):
                    url = f"https://{url}"
                workspace_url = url.rstrip("/")

        port = spark.conf.get("spark.ui.port")
        if port:
            spark_ui_port = str(port)
    except Exception:
        pass

    return workspace_url, spark_ui_port


def get_auth_headers() -> dict:
    """Build auth headers from environment variables.

    Credentials must be passed via spark_env_vars when creating the job
    cluster (auto-detect does not work on USER_ISOLATION clusters).

    Supports two modes:
        1. PAT token: DATABRICKS_TOKEN env var
        2. OAuth M2M: DATABRICKS_HOST + DATABRICKS_CLIENT_ID +
           DATABRICKS_CLIENT_SECRET env vars
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    host = os.environ.get("DATABRICKS_HOST")
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")

    if token:
        return {"Authorization": f"Bearer {token}"}

    if host and client_id and client_secret:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(host=host, client_id=client_id,
                            client_secret=client_secret)
        return w.config.authenticate()

    print("[run.py] ERROR: No credentials found. Set DATABRICKS_TOKEN or "
          "DATABRICKS_HOST + DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET "
          "via spark_env_vars.")
    return {}


def query_spark_ui(workspace_url: str, cluster_id: str, auth_headers: dict,
                   spark_ui_port: str) -> dict:
    """Query the Spark UI REST API via the driver-proxy-api.

    Returns a dict with total_jobs, active_jobs, latest_completion_time,
    and errors.
    """
    import requests as http_requests

    base_url = (f"{workspace_url}/driver-proxy-api/o/0/"
                f"{cluster_id}/{spark_ui_port}")
    result = {"total_jobs": 0, "active_jobs": 0, "errors": []}

    # 1. Get the Spark application ID
    try:
        resp = http_requests.get(f"{base_url}/api/v1/applications",
                                 headers=auth_headers, timeout=15)
        resp.raise_for_status()
        apps = resp.json()
        if not apps:
            result["errors"].append("No Spark applications found")
            return result
        app_id = apps[0].get("id")
    except Exception as e:
        result["errors"].append(f"apps: {type(e).__name__}: {e}")
        return result

    # 2. Get job counts and latest completion time
    try:
        resp = http_requests.get(
            f"{base_url}/api/v1/applications/{app_id}/jobs",
            headers=auth_headers, timeout=15)
        resp.raise_for_status()
        jobs = resp.json()

        result["total_jobs"] = len(jobs)
        result["active_jobs"] = sum(1 for j in jobs
                                    if j.get("status") == "RUNNING")

        completion_times = [
            j["completionTime"] for j in jobs if j.get("completionTime")
        ]
        if completion_times:
            result["latest_completion_time"] = max(completion_times)
    except Exception as e:
        result["errors"].append(f"jobs: {type(e).__name__}: {e}")

    return result


def compute_last_used_time(activity: dict) -> str | None:
    """Derive last_used_time from the Spark UI activity snapshot.

    Logic:
        1. active_jobs > 0  -> now (cluster is actively working)
        2. Jobs exist, none active -> latest completionTime from job history
        3. No jobs -> None
    """
    total_jobs = activity.get("total_jobs") or 0
    active_jobs = activity.get("active_jobs") or 0

    if total_jobs == 0:
        return None

    if active_jobs > 0:
        return datetime.now(timezone.utc).isoformat()

    # Parse latest completionTime from Spark UI
    # Format: "2026-02-09T19:48:29.171GMT"
    latest_ct = activity.get("latest_completion_time")
    if latest_ct:
        try:
            ct_str = latest_ct.replace("GMT", "+00:00")
            return datetime.fromisoformat(ct_str).isoformat()
        except (ValueError, TypeError):
            pass

    return None


def resolve_cache_path(cache_dir: str, cluster_id: str) -> str:
    """Resolve the cluster info file path from KUMO_CACHE_DIR.

    KUMO_CACHE_DIR is a UC Volume URL (dbfs:/Volumes/catalog/schema/volume/).
    On the Databricks cluster driver, UC Volumes are accessible as local
    filesystem paths (/Volumes/...) via FUSE mount.

    Returns the local filesystem path to the cluster info JSON file.
    """
    full_path = f"{cache_dir.rstrip('/')}/databricks_job_clusters/{cluster_id}"

    # UC Volume paths: strip dbfs: prefix to get local filesystem path
    if full_path.startswith("dbfs:"):
        return full_path[len("dbfs:"):]

    return full_path


def read_cluster_info(path: str) -> dict:
    """Read and parse cluster info JSON from the cache file."""
    try:
        with open(path, 'r') as f:
            return json.loads(f.read())
    except Exception as e:
        print(f"[run.py] WARNING: Failed to read cluster info from "
              f"{path}: {e}")
        return {}


def write_cluster_info(path: str, info: dict) -> bool:
    """Write cluster info JSON to the cache file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(json.dumps(info, indent=2))
        return True
    except Exception as e:
        print(f"[run.py] WARNING: Failed to write cluster info to "
              f"{path}: {e}")
        return False


def main():
    max_lifetime_seconds = int(
        os.environ.get("JOB_CLUSTER_MAX_LIFETIME_SECONDS", "86400"))
    idle_timeout_seconds = int(
        os.environ.get("JOB_CLUSTER_IDLE_TIMEOUT_SECONDS", "7200"))
    poll_interval = 60  # seconds

    print(
        f"[run.py] Starting monitoring. max_lifetime_seconds={max_lifetime_seconds}s, "
        f"idle_timeout_seconds={idle_timeout_seconds}s, poll_interval={poll_interval}s"
    )

    cluster_id = get_cluster_id()
    print(f"[run.py] Cluster ID: {cluster_id}")

    cache_dir = os.environ.get("KUMO_CACHE_DIR")
    if not cache_dir:
        print("[run.py] WARNING: KUMO_CACHE_DIR not set, "
              "falling back to sleep mode")
        time.sleep(max_lifetime_seconds)
        return

    cache_path = resolve_cache_path(cache_dir, cluster_id)
    print(f"[run.py] Cache path: {cache_path}")

    workspace_url, spark_ui_port = get_proxy_config()
    print(f"[run.py] Proxy: {workspace_url}, port={spark_ui_port}")

    auth_headers = get_auth_headers()

    start_time = time.time()
    poll_count = 0
    prev_total_jobs = 0
    # Track the latest last_used_time seen (ISO timestamp string or None)
    current_last_used_time = None

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_lifetime_seconds:
            print("[run.py] Max lifetime reached. Exiting.")
            break

        # Check idle timeout based on actual last_used_time timestamp.
        # - If we've seen activity: compare now vs last_used_time
        # - If no activity ever: compare now vs start_time
        now = datetime.now(timezone.utc)
        if current_last_used_time:
            try:
                last_used_dt = datetime.fromisoformat(current_last_used_time)
                idle_seconds = (now - last_used_dt).total_seconds()
            except (ValueError, TypeError):
                idle_seconds = elapsed
        else:
            idle_seconds = elapsed

        if idle_seconds >= idle_timeout_seconds:
            print(f"[run.py] Idle timeout reached: {round(idle_seconds)}s "
                  f"since last activity (timeout={idle_timeout_seconds}s). "
                  f"last_used_time={current_last_used_time}. Exiting.")
            break

        poll_count += 1
        activity_snapshot = {}
        last_used_time = None

        if workspace_url and auth_headers:
            try:
                # Re-authenticate periodically to handle token refresh
                if poll_count % 5 == 0:  # every ~5 min
                    auth_headers = get_auth_headers()

                activity_snapshot = query_spark_ui(workspace_url, cluster_id,
                                                   auth_headers, spark_ui_port)

                last_used_time = compute_last_used_time(activity_snapshot)

                total_jobs = activity_snapshot.get("total_jobs") or 0
                active_jobs = activity_snapshot.get("active_jobs") or 0

                # Update cache if there's Spark activity:
                # - Active jobs running now
                # - New jobs completed since last poll
                has_activity = (active_jobs > 0 or total_jobs > prev_total_jobs)
                prev_total_jobs = total_jobs

                if has_activity and last_used_time:
                    info = read_cluster_info(cache_path)
                    info["last_used_time"] = last_used_time
                    write_cluster_info(cache_path, info)

                # Update our tracked last_used_time
                if last_used_time:
                    current_last_used_time = last_used_time

            except Exception as e:
                print(f"[run.py] ERROR poll {poll_count}: "
                      f"{type(e).__name__}: {e}")

        print(f"[run.py] Poll {poll_count}: "
              f"total_jobs={activity_snapshot.get('total_jobs')}, "
              f"active_jobs={activity_snapshot.get('active_jobs')}, "
              f"last_used_time={last_used_time}, "
              f"idle={round(idle_seconds)}s, "
              f"elapsed={round(elapsed, 1)}s")

        time.sleep(poll_interval)

    print(f"[run.py] Done. polls={poll_count}")


if __name__ == "__main__":
    main()
    # Note: Do NOT use sys.exit(0) here. Databricks wraps spark_python_task
    # in IPython, which intercepts SystemExit and reports it as a failure.
    # Just let the script end naturally.
