# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An isolated, loopback S3 endpoint for exercising both Python and native clients."""

import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch


@contextmanager
def local_s3():
    import boto3

    # Use a separate process because native index construction may hold the GIL.
    with tempfile.TemporaryDirectory() as tmp:
        ready = Path(tmp) / "endpoint"
        log = Path(tmp) / "server.log"
        with log.open("w") as output:
            server = subprocess.Popen(
                [sys.executable, "-m", "tests.e2e.s3_endpoint", str(ready)],
                stdout=output,
                stderr=output,
            )
            try:
                deadline = monotonic() + 30
                while not ready.exists():
                    if server.poll() is not None or monotonic() >= deadline:
                        raise RuntimeError(f"S3 test server failed to start:\n{log.read_text()}")
                    sleep(0.025)
                endpoint = ready.read_text()
                env = {
                    "S3_ENDPOINT": endpoint,
                    "S3_ALLOW_HTTP": "true",
                    "S3_REGION": "us-east-1",
                    "S3_VIRTUAL_HOSTED": "false",
                    "S3_ACCESS_KEY": "testing",
                    "S3_SECRET_KEY": "testing",
                    "S3_SESSION_TOKEN": "testing",
                    "AWS_ACCESS_KEY_ID": "testing",
                    "AWS_SECRET_ACCESS_KEY": "testing",
                    "AWS_SESSION_TOKEN": "testing",
                    "AWS_DEFAULT_REGION": "us-east-1",
                    "AWS_EC2_METADATA_DISABLED": "true",
                    "NO_PROXY": "localhost,127.0.0.1",
                    "no_proxy": "localhost,127.0.0.1",
                }
                with patch.dict(os.environ, env):
                    for name in ("S3_PROFILE", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
                        os.environ.pop(name, None)
                    client = boto3.client(
                        "s3",
                        endpoint_url=endpoint,
                        region_name="us-east-1",
                        aws_access_key_id="testing",
                        aws_secret_access_key="testing",
                    )
                    try:
                        client.create_bucket(Bucket="omnisage-fixture")
                        yield client
                    finally:
                        client.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    from moto.server import DomainDispatcherApplication, create_backend_app
    from werkzeug.serving import make_server

    with make_server(
        "127.0.0.1", 0, DomainDispatcherApplication(create_backend_app), threaded=True
    ) as service:
        path = Path(sys.argv[1])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(f"http://127.0.0.1:{service.server_port}")
        tmp.replace(path)
        service.serve_forever()
