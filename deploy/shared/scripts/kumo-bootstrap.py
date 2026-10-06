#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Kumo single-host derived state (secrets, TLS, configs).

This is the one shared, idempotent implementation of the single-host
generation logic. Two wrappers drive it against different roots:

  * ``kumo-firstboot.sh`` (AMI cloud-init) runs it as root on the host,
    writing the host paths the AMI compose binds expect (/opt/kumo/secrets,
    /etc/kumo/deploy_env_config.json, /opt/kumo/nginx/nginx.conf, ...).
  * the ``kumo-bootstrap`` one-shot compose service runs it as root inside
    the backend image, writing the named volumes every other service mounts
    read-only.

Artifacts are generated idempotently: operator-owned config inputs are left
untouched, while bootstrap-owned derived state may be refreshed from current
inputs. Secret material is written under a restrictive umask with explicit
perms set before any content is flushed.

Inputs arrive as environment variables (same names firstboot already
parses from instance metadata). Output locations are parameterized through
KUMO_BOOTSTRAP_* variables so each wrapper can point the same code at host
paths or container mount targets.
"""

import base64
import json
import os
import secrets
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path

# Container/AMI-user identity the backend services run as. Secret files must
# be owned by this uid so the read-only mounts are readable at runtime.
OWNER_UID = int(os.environ.get("KUMO_BOOTSTRAP_OWNER_UID", "999"))
OWNER_GID = int(os.environ.get("KUMO_BOOTSTRAP_OWNER_GID", "999"))

SCRIPT_DIR = Path(__file__).resolve().parent
# Source assets (deploy config template, nginx variants, prometheus.yml) live
# as siblings of the scripts dir under deploy/shared. SRC_ROOT is overridable
# for wrappers that stage the tree at a different mount point.
SRC_ROOT = Path(
    os.environ.get("KUMO_BOOTSTRAP_SRC_ROOT", str(SCRIPT_DIR.parent)))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _log(message: str) -> None:
    print(f"[kumo-bootstrap] {message}", flush=True)


def _chown_best_effort(path: str) -> bool:
    """chown to the runtime uid/gid; returns whether ownership was set.

    Left best-effort so the module can also be exercised unprivileged in
    tests — real runs (AMI host root, container user "0") always succeed and
    the O_CREAT modes below already fixed perms before any content landed.
    """
    try:
        os.chown(path, OWNER_UID, OWNER_GID)
        return True
    except OSError:
        _log(f"warning: could not chown {path} to {OWNER_UID}:{OWNER_GID} "
             "(not running as root)")
        return False


def _write_file(path: str, content: str, mode: int, *, chown: bool) -> None:
    # Create with the final mode from the start (umask-guarded) so secret
    # bytes are never briefly world-readable between open and chmod.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)
    os.chmod(path, mode)
    if chown:
        _chown_best_effort(path)


def _write_secret_file(path: str, content: str, mode: int) -> None:
    """Atomically replace a secret file (temp file in the same dir + rename).

    Used for the customer secret whose contents may be merged across runs: the
    reader (services) never sees a half-written file, and an interrupted run
    cannot corrupt an existing valid secret. The temp file is created with the
    final restrictive mode so the bytes are never briefly world-readable, and
    ownership is set before the rename so the published inode is already owned
    by the runtime uid.
    """
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, prefix=".kumo-secret-")
    try:
        try:
            os.fchmod(fd, mode)
            os.write(fd, content.encode())
        finally:
            os.close(fd)
        _chown_best_effort(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ensure_secrets_dir(secrets_dir: str) -> None:
    # Enforce perms/ownership every run (matches firstboot): packer pre-creates
    # /opt/kumo/secrets root-owned 0755, so a skip-if-exists would leave the
    # dir world-listable instead of 0750 owned by the runtime uid. When
    # ownership cannot be set (rootless bind mount), fall back to a
    # world-traversable 0755 so uid-999 services can still reach the
    # individually-perm'd secret files — matching compose-local preflight.
    os.makedirs(secrets_dir, exist_ok=True)
    if _chown_best_effort(secrets_dir):
        os.chmod(secrets_dir, 0o750)
    else:
        os.chmod(secrets_dir, 0o755)


def generate_customer_secret(secrets_dir: str) -> None:
    """Secret-key JSON: credential encryption key + API-key pepper.

    Merge semantics (matches the pre-module firstboot block): an existing file
    is never overwritten key-for-key. Instead, only keys absent for the current
    mode are added — the always-required encryption key + API-key pepper, plus
    the Direct Connect secrets (OIDC client secret / local admin temp password)
    when they are supplied. This lets a file first written with auth disabled
    gain its Direct Connect secrets on a later auth-enabled run without rotating
    the encryption material. Non-secret auth settings live in the deploy config
    instead and reference this file by name+key.
    """
    secret_file = os.path.join(secrets_dir, "kumo-secret-key")

    if os.path.exists(secret_file):
        with open(secret_file, encoding="utf-8") as f:
            secret = json.load(f)
        original = dict(secret)
    else:
        secret = {}
        original = None

    # Always-required material: minted once and never rotated. setdefault adds
    # it only when the file predates these keys (older firstboot output) or is
    # brand new; an existing value is left exactly as-is.
    secret.setdefault(
        "credentials_encryption_key",
        base64.urlsafe_b64encode(os.urandom(32)).decode(),
    )
    secret.setdefault("api_key_pepper", secrets.token_urlsafe(32))

    # Direct Connect secrets for the active mode: add the supplied values only
    # when missing, never overwriting anything already on disk.
    updates = {
        "oidc_client_secret": _env("OIDC_CLIENT_SECRET"),
        "local_admin_temp_password": _env("DIRECT_CONNECT_ADMIN_TEMP_PASSWORD"),
    }
    for key, value in updates.items():
        if value:
            secret.setdefault(key, value)

    if original is not None and secret == original:
        # Existing file already complete for this mode: leave the bytes
        # untouched (idempotent) and only reassert perms/ownership.
        os.chmod(secret_file, 0o600)
        _chown_best_effort(secret_file)
        return

    _write_secret_file(secret_file, json.dumps(secret) + "\n", 0o600)
    _log(f"wrote customer secret {secret_file}")


def _write_postgres_dsn_secrets(secrets_dir: str, extra_env: dict) -> None:
    state_file = os.path.join(secrets_dir, "STATE_DB_DSN")
    temporal_file = os.path.join(secrets_dir, "TEMPORAL_DB_DSN")
    visibility_file = os.path.join(secrets_dir, "TEMPORAL_VISIBILITY_DB_DSN")
    if all(
            os.path.exists(p)
            for p in (state_file, temporal_file, visibility_file)):
        return

    child_env = dict(os.environ)
    child_env.update(extra_env)
    child_env.update({
        "STATE_DB_DSN_FILE": state_file,
        "TEMPORAL_DB_DSN_FILE": temporal_file,
        "TEMPORAL_VISIBILITY_DB_DSN_FILE": visibility_file,
    })
    subprocess.run(
        [sys.executable,
         str(SCRIPT_DIR / "write-postgres-secrets.py")],
        check=True,
        env=child_env,
    )
    for path in (state_file, temporal_file, visibility_file):
        os.chmod(path, 0o640)
        _chown_best_effort(path)
    _log("generated structured Postgres DSN secrets")


def generate_postgres_secrets(secrets_dir: str) -> None:
    """Bundled mode: persist a random Postgres password + structured DSNs.

    External mode: the caller supplies connection settings; only the DSN
    secrets are written (no bundled password is invented).
    """
    mode = _env("KUMO_POSTGRES_MODE", "bundled").lower()
    if mode == "external":
        _write_postgres_dsn_secrets(secrets_dir, extra_env={})
        return

    password_file = os.path.join(secrets_dir, "kumo-postgres-password")
    dsn_files = [
        os.path.join(secrets_dir, name)
        for name in ("STATE_DB_DSN", "TEMPORAL_DB_DSN",
                     "TEMPORAL_VISIBILITY_DB_DSN")
    ]
    dsns_exist = all(os.path.exists(p) for p in dsn_files)
    password_exists = os.path.exists(password_file)

    if dsns_exist and password_exists:
        # Fully provisioned (AMI re-run, compose-local preflight output, or a
        # prior bootstrap). Leave the authored password + DSNs untouched —
        # inventing a fresh bundled password here would disagree with the
        # password already baked into those DSNs (and the compose
        # POSTGRES_PASSWORD). Only reassert perms/ownership.
        for path in [password_file, *dsn_files]:
            os.chmod(path, 0o640)
            _chown_best_effort(path)
        return

    if dsns_exist and not password_exists:
        # The DSNs already baked a specific password, but the password file
        # firstboot reads back into the compose .env is gone. Rebuild it from
        # the supplied POSTGRES_PASSWORD (which must match the one embedded in
        # the DSNs); never invent a random password that would contradict the
        # existing DSNs.
        supplied = _env("POSTGRES_PASSWORD")
        if not supplied:
            raise SystemExit(
                "ERROR: bundled Postgres DSN secrets exist but the password "
                f"file {password_file} is missing and POSTGRES_PASSWORD is not "
                "set. Set POSTGRES_PASSWORD to the password baked into the "
                "existing DSN secrets, or delete the DSN secrets (STATE_DB_DSN, "
                "TEMPORAL_DB_DSN, TEMPORAL_VISIBILITY_DB_DSN) to regenerate the "
                "full set.")
        _write_file(password_file, supplied, 0o640, chown=True)
        _log(f"restored bundled Postgres password file {password_file}")
        return

    # DSNs are absent (fresh boot or partial state): settle on one bundled
    # password so the password file and all three DSNs agree. Prefer an existing
    # password file (prior run), then the supplied POSTGRES_PASSWORD (so a pure
    # `docker compose up` with the bundled overlay uses the same password
    # Postgres itself starts with), and only invent a random one when neither
    # exists (AMI fresh boot, where firstboot reads the file back into the
    # compose .env).
    if password_exists:
        os.chmod(password_file, 0o640)
        _chown_best_effort(password_file)
        password = Path(password_file).read_text(encoding="utf-8").strip()
    else:
        password = _env("POSTGRES_PASSWORD") or secrets.token_urlsafe(32)
        _write_file(password_file, password, 0o640, chown=True)
        _log(f"generated bundled Postgres password {password_file}")
    _write_postgres_dsn_secrets(
        secrets_dir,
        extra_env={
            "POSTGRES_USER": _env("POSTGRES_USER", "kumo"),
            "POSTGRES_HOST": _env("POSTGRES_HOST", "postgres"),
            "POSTGRES_PORT": _env("POSTGRES_PORT", "5432"),
            "POSTGRES_PASSWORD": password,
            # Bundled Postgres never uses client TLS; make sure a stray sslmode in
            # the environment does not leak into the generated DSNs.
            "POSTGRES_SSLMODE": "",
            "POSTGRES_SSLROOTCERT": "",
            "POSTGRES_PASSWORD_FILE": "",
        })


class _EmptyDefault(dict):
    """Mapping that renders unset ${VAR} as empty, matching envsubst."""
    def __missing__(self, _key: str) -> str:  # noqa: D401
        return ""


def generate_deploy_config(config_file: str) -> None:
    """Render deploy config, refreshing the bootstrap-owned generated file."""
    config_exists = os.path.exists(config_file)
    if config_exists and _env(
            "KUMO_BOOTSTRAP_DEPLOY_CONFIG_SOURCE") != "kumo_config":
        return

    candidate_file = config_file
    if config_exists:
        fd, candidate_file = tempfile.mkstemp(
            dir=os.path.dirname(config_file) or ".",
            prefix=".kumo-deploy-config-",
        )
        os.close(fd)

    try:
        template_path = SRC_ROOT / "configs" / "deploy_env_config.template.json"
        template_text = template_path.read_text(encoding="utf-8")

        # Defaults for the render match firstboot's template-only variables.
        render_env = _EmptyDefault(os.environ)
        render_env.setdefault("TEMPORAL_HOST", "temporal")
        render_env.setdefault("TEMPORAL_PORT", "7233")
        render_env.setdefault("TEMPORAL_NAMESPACE", "prod")
        render_env.setdefault("METADATA_STORAGE_TYPE", "postgres")
        render_env.setdefault("STATE_DB_DSN_SECRET_NAME", "STATE_DB_DSN")
        render_env.setdefault("DISKGRAPH_DATA_ROOT", "/mnt/localssd")
        render_env.setdefault("LIFECYCLE_READY", "false")

        rendered = string.Template(template_text).substitute(render_env)

        os.makedirs(os.path.dirname(candidate_file), exist_ok=True)
        with open(candidate_file, "w", encoding="utf-8") as f:
            f.write(rendered)

        if _env("DISABLE_AUTH", "false").lower() != "true":
            _inject_direct_connect_auth(candidate_file)
        # Non-secret config (references secrets by name); world-readable so the
        # uid-999 services can read the read-only mount.
        os.chmod(candidate_file, 0o644)

        if config_exists:
            expected = json.loads(
                Path(candidate_file).read_text(encoding="utf-8"))
            try:
                actual = json.loads(
                    Path(config_file).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                actual = object()
            if actual == expected:
                return
            os.replace(candidate_file, config_file)
            candidate_file = config_file
            _log(f"updated generated deploy config {config_file}")
        else:
            _log(f"rendered deploy config {config_file}")
    finally:
        if config_exists and candidate_file != config_file:
            try:
                os.unlink(candidate_file)
            except OSError:
                pass


def _inject_direct_connect_auth(config_file: str) -> None:
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    login_method = (_env("DIRECT_CONNECT_LOGIN_METHOD") or "oidc").lower()
    admin_emails = [
        email.strip() for email in _env("ADMIN_EMAILS").split(",")
        if email.strip()
    ]
    public_url = _env("PUBLIC_URL").rstrip("/")
    secret_name = "kumo-secret-key"

    direct_connect = {
        "login_method": login_method,
        "allowed_return_origins": [public_url] if public_url else [],
        "admin_emails": admin_emails,
    }
    if login_method == "password":
        direct_connect["password"] = {
            "temporary_password_ref": {
                "secret_name": secret_name,
                "secret_key": "local_admin_temp_password",
            },
        }
    else:
        direct_connect["oidc"] = {
            "issuer": _env("OIDC_ISSUER").rstrip("/"),
            "client_id": _env("OIDC_CLIENT_ID"),
            "groups_claim": _env("OIDC_GROUPS_CLAIM", "groups") or "groups",
            "client_secret_ref": {
                "secret_name": secret_name,
                "secret_key": "oidc_client_secret",
            },
        }

    config["auth"] = {
        "enabled": True,
        "backend": "direct_connect",
        "direct_connect": direct_connect,
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def generate_tls(tls_dir: str) -> None:
    """Self-signed cert for TLS-enabled installs (skipped when TLS is off)."""
    if _env("ENABLE_TLS", "false").lower() != "true":
        return
    if _env("KUMO_BOOTSTRAP_GENERATE_TLS", "true").lower() != "true":
        return
    crt = os.path.join(tls_dir, "server.crt")
    key = os.path.join(tls_dir, "server.key")
    # Regenerate when either half is missing: a lone crt (or lone key) is
    # unusable, and openssl rewrites the pair atomically, so a half-present
    # state clobbers nothing usable. A complete pair is left untouched.
    if os.path.exists(crt) and os.path.exists(key):
        return
    os.makedirs(tls_dir, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:4096", "-keyout", key,
            "-out", crt, "-days", "365", "-nodes", "-subj",
            "/CN=kumo-singlehost"
        ],
        check=True,
    )
    os.chmod(key, 0o600)
    os.chmod(crt, 0o644)
    _log(f"generated self-signed TLS certificate in {tls_dir}")


def generate_nginx_config(nginx_file: str) -> None:
    """Place the TLS or HTTP nginx variant (selection, not content authoring).

    Generate-if-absent: this module now runs on every compose `up`, so an
    existing nginx.conf is left in place rather than rewritten — that would
    clobber an operator-customized config every boot. When the existing content
    does not match the variant selected by the current TLS mode, one note is
    logged so a stale-variant state (for example TLS toggled after first boot)
    is visible without overwriting the file.
    """
    variant = "nginx.conf" if _env("ENABLE_TLS", "false").lower() == "true" \
        else "nginx.http.conf"
    source = SRC_ROOT / "nginx" / variant
    expected = source.read_text(encoding="utf-8")
    if os.path.exists(nginx_file):
        if Path(nginx_file).read_text(encoding="utf-8") != expected:
            _log(
                f"note: existing nginx config at {nginx_file} differs from the "
                f"{variant} variant expected for the current TLS mode; leaving "
                "the existing file in place (delete it to regenerate)")
        return
    os.makedirs(os.path.dirname(nginx_file), exist_ok=True)
    Path(nginx_file).write_text(expected, encoding="utf-8")
    os.chmod(nginx_file, 0o644)
    _log(f"placed nginx config ({variant}) at {nginx_file}")


def place_config(source: Path, destination: str, mode: int) -> None:
    """Copy a packaged config into its destination once."""
    if not destination or os.path.exists(destination):
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(source, destination)
    os.chmod(destination, mode)
    _log(f"placed config at {destination}")


def main() -> int:
    secrets_dir = _env("KUMO_BOOTSTRAP_SECRETS_DIR", "/opt/kumo/secrets")
    deploy_config_file = _env("KUMO_BOOTSTRAP_DEPLOY_CONFIG_FILE",
                              "/etc/kumo/config/deploy_env_config.json")
    nginx_file = _env("KUMO_BOOTSTRAP_NGINX_FILE", "/etc/kumo/nginx/nginx.conf")
    prometheus_file = _env("KUMO_BOOTSTRAP_PROMETHEUS_FILE",
                           "/etc/kumo/prometheus/prometheus.yml")
    tls_dir = _env("KUMO_BOOTSTRAP_TLS_DIR", "/etc/kumo/tls")

    # Secret material first. _write_file creates secrets with the final
    # restrictive mode via O_CREAT (perms before content), so there is no
    # window where they are world-readable regardless of the ambient umask.
    _ensure_secrets_dir(secrets_dir)
    generate_customer_secret(secrets_dir)
    generate_postgres_secrets(secrets_dir)

    generate_deploy_config(deploy_config_file)
    generate_tls(tls_dir)
    generate_nginx_config(nginx_file)
    place_config(SRC_ROOT / "prometheus/prometheus.yml", prometheus_file, 0o644)
    place_config(
        SRC_ROOT / "runtime-config/postgres-initdb/01-create-kumo-databases.sh",
        _env("KUMO_BOOTSTRAP_POSTGRES_INIT_FILE"),
        0o755,
    )
    place_config(
        SRC_ROOT / "runtime-config/temporal/dynamicconfig/docker.yaml",
        _env("KUMO_BOOTSTRAP_TEMPORAL_DYNAMIC_CONFIG_FILE"),
        0o644,
    )
    _log("bootstrap complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
