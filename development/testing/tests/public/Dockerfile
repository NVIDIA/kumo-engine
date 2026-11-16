# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.14.7-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2

ENV PIP_CONFIG_FILE=/dev/null PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /src

# Install dependencies before adding source; no package setup hooks or internal
# installer are needed for this source-test profile.
COPY --chown=65534:65534 development/testing/tests/public/requirements*.txt development/testing/tests/public/
RUN python -I -m pip --isolated install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.0+cpu'
RUN python -I -m pip --isolated install --no-cache-dir \
    --index-url https://pypi.org/simple -r development/testing/tests/public/requirements-all.txt \
    && python -I -m pip check

# Dockerfile.dockerignore limits the build context to the selected source/tests.
COPY --chown=65534:65534 packages/ packages/
COPY --chown=65534:65534 apps/studio/io_utils/ apps/studio/io_utils/
COPY --chown=65534:65534 apps/studio/async_utils/ apps/studio/async_utils/
COPY --chown=65534:65534 apps/studio/monitoring/ apps/studio/monitoring/
COPY --chown=65534:65534 apps/studio/service_framework/ apps/studio/service_framework/
COPY --chown=65534:65534 development/testing/test/ development/testing/test/
COPY --chown=65534:65534 development/tooling/scripts/test_public.py development/tooling/scripts/test_public_connectors.py development/tooling/scripts/
COPY --chown=65534:65534 development/testing/tests/public/ development/testing/tests/public/

USER 65534:65534
ENV HOME=/tmp/public-connectors
CMD ["python", "-I", "development/tooling/scripts/test_public_connectors.py", "--suite", "all"]
