# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-cluster install preflight, run by the chart's pre-install/pre-upgrade hook.

The in-cluster counterpart of check.sh: verifies optional Temporal mTLS
material, the PostgreSQL CA bundle, and TCP reachability from the workloads'
own vantage.

Stdlib only (the kumo image has no kubectl); all inputs arrive as env vars
rendered by templates/preflight/job-preflight.yaml, so this file needs no Helm
templating and ships into the hook's ConfigMap via .Files.Get.

Limitation: TEMPORAL_ACCESS is a TCP connect, not a gRPC/mTLS handshake — a
wrong client cert still passes here (the pre-install register-temporal-ns
hook exercises the real client and catches that).
"""
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request

API = "https://kubernetes.default.svc"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

failures = []


def report(status, contract_id, msg):
    print("[%s] %s: %s" % (status, contract_id, msg))
    if status == "FAIL" and contract_id not in failures:
        failures.append(contract_id)


def api_opener():
    # ProxyHandler({}) pins the apiserver call in-cluster even when the pod
    # inherits HTTPS_PROXY from an admission webhook or base image.
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(
            cafile=CA_PATH)), urllib.request.ProxyHandler(dict()))


def get_secret(opener, namespace, name):
    """Return (secret_body, None) or (None, error_string)."""
    url = "%s/api/v1/namespaces/%s/secrets/%s" % (API, namespace, name)
    with open(TOKEN_PATH) as token_file:
        token = token_file.read().strip()
    req = urllib.request.Request(
        url,
        headers=dict(Authorization="Bearer " + token))
    last_err = None
    for attempt in range(3):
        if attempt:
            time.sleep(3)
        try:
            with opener.open(req, timeout=10) as resp:
                return json.load(resp), None
        except urllib.error.HTTPError as err:
            return None, "HTTP %d" % err.code
        except (urllib.error.URLError, OSError) as err:
            last_err = err  # transient apiserver problem — retry
    return None, "apiserver unreachable after 3 attempts (%s)" % last_err


def check_secret(opener, contract_id, namespace, name, required_keys):
    """Verify the Secret exists with the required keys; return its data map."""
    body, err = get_secret(opener, namespace, name)
    if err == "HTTP 404":
        report("FAIL", contract_id,
               "required Secret not found in the release namespace")
        return None
    if err is not None:
        report(
            "FAIL", contract_id,
            "cannot read required Secret — namespace, RBAC, or "
            "apiserver problem")
        return None
    data = body.get("data") or dict()
    missing = [k for k in required_keys if k not in data]
    if missing:
        report("FAIL", contract_id,
               "required Secret is missing one or more required keys")
        return None
    report("PASS", contract_id,
           "required Secret is present with all required keys")
    return data


def check_dial(contract_id, host, port):
    last_err = None
    for attempt in range(3):
        if attempt:
            time.sleep(2)
        try:
            socket.create_connection((host, int(port)), timeout=5).close()
            report("PASS", contract_id, "%s:%s reachable" % (host, port))
            return
        except socket.gaierror as err:
            last_err = err
            break  # name does not resolve — retrying won't help
        except OSError as err:
            last_err = err
    report("FAIL", contract_id,
           "cannot reach %s:%s (%s)" % (host, port, last_err))


def split_host_port(address, default_port):
    """Return (host, port), accepting DNS/IPv4 and bracketed IPv6."""
    host = address
    port = default_port
    if address.startswith("["):
        end = address.find("]")
        if end < 0 or (len(address) > end + 1 and address[end + 1] != ":"):
            return None
        host = address[1:end]
        if len(address) > end + 1:
            port = address[end + 2:]
    elif address.count(":") == 1:
        host, port = address.rsplit(":", 1)
    if not host:
        return None
    try:
        port_number = int(port)
    except (TypeError, ValueError):
        return None
    if not 1 <= port_number <= 65535:
        return None
    return host, str(port_number)


def main():
    temporal_secret = os.environ.get("TEMPORAL_CLIENT_MTLS_SECRET", "")
    temporal_address = split_host_port(os.environ["TEMPORAL_HOST"], "7233")

    if temporal_address is None:
        report("FAIL", "TEMPORAL_ACCESS",
               "temporal.address must be a valid host with an optional port")
        temporal_host = None
    else:
        temporal_host, temporal_port = temporal_address

    if temporal_secret:
        check_secret(api_opener(), "TEMPORAL_MTLS", os.environ["POD_NAMESPACE"],
                     temporal_secret, ["tls.crt", "tls.key", "ca.crt"])
    else:
        print("[SKIP] TEMPORAL_MTLS: temporal.clientMtlsSecret is not set")

    ca_path = os.environ["PG_CA_PATH"]
    try:
        with open(ca_path, "rb") as ca_bundle:
            ca_bundle.read(1)
    except OSError:
        report("FAIL", "POSTGRES_CA",
               "baked Postgres CA bundle is not readable")
    else:
        report("PASS", "POSTGRES_CA", "Postgres CA bundle is readable")

    if temporal_host is not None:
        check_dial("TEMPORAL_ACCESS", temporal_host, temporal_port)

    # With password auth the host lives in the mounted Secret, not in values.
    pg_secret_path = os.environ.get("PG_DSN_SECRET_PATH")
    if pg_secret_path:
        try:
            with open(pg_secret_path, "rb") as handle:
                conn = json.loads(handle.read())
            pg_host = str(conn["host"])
            pg_port = str(conn.get("port") or "5432")
            pg_tls_ca = (conn.get("tls") or {}).get("sslrootcert")
        except (OSError, ValueError, KeyError, TypeError):
            report("FAIL", "PG_ACCESS",
                   "postgres.dsnSecret does not contain readable JSON with a "
                   "host field")
            pg_host = pg_port = pg_tls_ca = None
    else:
        pg_host, pg_port = os.environ["PG_HOST"], os.environ["PG_PORT"]
        pg_tls_ca = None

    if pg_host is not None:
        check_dial("PG_ACCESS", pg_host, pg_port)

    # Helm never reads the DSN Secret, so a wrong sslrootcert path only surfaces
    # at runtime. Check it here to fail the install instead.
    if pg_tls_ca:
        try:
            with open(pg_tls_ca, "rb") as ca_file:
                ca_file.read(1)
        except OSError:
            report(
                "FAIL", "POSTGRES_TLS_CA",
                "tls.sslrootcert is not a readable file in the pod; set "
                "postgres.caSecret and point sslrootcert at "
                "/var/run/secrets/kumo/ca/ca.pem")
        else:
            report("PASS", "POSTGRES_TLS_CA",
                   "tls.sslrootcert is readable")

    if failures:
        print()
        print("Preflight FAILED — prerequisite(s) not satisfied: %s" %
              ", ".join(failures))
        print("These are customer-provided prerequisites; the chart does not "
              "create them.")
        print("Fix: use the matching kumo-stack/docs/install-{eks,aks,aro}.md "
              "recipe, or run conformance/check.sh prereq for the full probe.")
        print("To install anyway (workloads will crash-loop until the "
              "prerequisites exist):")
        print("  --set preflight.enabled=false")
        sys.exit(1)
    print()
    print("Preflight passed.")


if __name__ == "__main__":
    main()
