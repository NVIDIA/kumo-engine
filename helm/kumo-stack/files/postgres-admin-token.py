# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage PostgreSQL credentials for the Temporal bootstrap Job.

The psql image lacks cloud SDKs, so an init container mints credentials with the
pod's workload identity and shares them through a pod-private tmpfs.
"""

import os

AWS_RDS_TRUST_ANCHOR = "/opt/kumo/certs/aws-rds-roots.pem"
SYSTEM_TRUST_ANCHOR = "/etc/ssl/certs/ca-certificates.crt"
ENTRA_POSTGRES_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"


def aws_rds_token() -> str:
    import boto3

    return boto3.client(
        "rds", region_name=os.environ["AWS_REGION"]).generate_db_auth_token(
            DBHostname=os.environ["PGHOST"],
            Port=int(os.environ["PGPORT"]),
            DBUsername=os.environ["PGUSER"],
        )


def azure_entra_token() -> str:
    from azure.identity import WorkloadIdentityCredential

    credential = WorkloadIdentityCredential()
    try:
        return credential.get_token(ENTRA_POSTGRES_SCOPE).token
    finally:
        credential.close()


def stage(directory: str, name: str, content: bytes) -> None:
    """Write one file that only this pod's own containers can read."""
    descriptor = os.open(
        os.path.join(directory, name),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o640,
    )
    try:
        os.write(descriptor, content)
        # psql runs under a different UID than this container. fsGroup owns the
        # volume, so hand the file to that group instead of widening its mode.
        os.fchown(descriptor, -1, os.stat(directory).st_gid)
        os.fchmod(descriptor, 0o640)
    finally:
        os.close(descriptor)


def main() -> None:
    directory = os.environ["KUMO_CREDENTIAL_DIR"]
    mode = os.environ["POSTGRES_ADMIN_MODE"]
    if mode == "aws_iam":
        token = aws_rds_token()
        trust_anchor = AWS_RDS_TRUST_ANCHOR
    elif mode == "azure_entra":
        token = azure_entra_token()
        trust_anchor = SYSTEM_TRUST_ANCHOR
    else:
        raise SystemExit(f"unsupported PostgreSQL bootstrap auth mode: {mode}")

    with open(trust_anchor, "rb") as source:
        stage(directory, "ca.pem", source.read())
    stage(directory, "admin-password", token.encode("utf-8"))


if __name__ == "__main__":
    main()
