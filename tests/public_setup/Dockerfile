# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.14.7-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS dependencies
ENV PIP_CONFIG_FILE=/dev/null UV_NO_CONFIG=1 UV_LINK_MODE=copy
RUN python -m pip install --no-cache-dir uv==0.12.23
WORKDIR /src
# Frozen exports need metadata, not application source or tests.
COPY pyproject.toml uv.lock ./
COPY packages/kumo-sdk/pyproject.toml packages/kumo-sdk/
RUN uv export --frozen --no-dev --no-emit-workspace --no-hashes \
    --package kumoai-studio --extra test --output-file /tmp/requirements.txt \
    && uv pip install --system --no-deps -r /tmp/requirements.txt
ENV HOME=/tmp/public-setup AWS_EC2_METADATA_DISABLED=true PYTHONDONTWRITEBYTECODE=1

FROM dependencies AS sdk
COPY --chown=65534:65534 . .
RUN uv pip install --system --no-deps ./packages/kumo-api ./packages/kumo-sdk \
    && uv pip check
USER 65534:65534

FROM dependencies AS full
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev && rm -rf /var/lib/apt/lists/*
# Only test source from this checkout. This is not a release artifact.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
COPY --from=wheel / /wheels/
RUN uv export --frozen --no-dev --no-emit-workspace --no-hashes \
    --package kumo --package kumo-engine --group public-tests --extra full --extra test \
    --output-file /tmp/requirements.txt \
    && uv pip install --system --no-deps --torch-backend cpu \
    -r /tmp/requirements.txt /wheels/diskgraph-*.whl
COPY --chown=65534:65534 . .
RUN uv pip install --system --no-deps ./packages/kumo-api ./packages/kumo-sdk \
    ./packages/kumo-pql ./packages/kumo-core ./packages/kumo-engine \
    ./packages/kumo-s3 ./packages/kumo-gcs ./packages/kumo-adls ./packages/kumo-snowflake \
    ./packages/kumo-databricks ./packages/kumo-bigquery '.[full,test]' \
    && uv pip check
ENV PYTHONPATH=/src DEPLOY_STAGE=test KUMO_TESTING=True \
    KUMO_HOME=/tmp/public-setup/kumo \
    MAX_DATA_BACKEND_WORKER_PROCESSES=1 AWS_REQUEST_CHECKSUM_CALCULATION=when_required
RUN --network=none python -I - <<'PY'
from importlib.metadata import distributions, version
import kumo.engine
import torch

version('kumo-engine')
assert torch.version.cuda is None
faiss_packages = {d.metadata['Name'] for d in distributions()} & {'faiss-cpu', 'faiss-gpu'}
assert faiss_packages == {'faiss-gpu'}, faiss_packages
PY
USER 65534:65534
