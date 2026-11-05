# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# docker build -f examples/mlops/sagemaker/Dockerfile -t kumo-engine-sagemaker <wheel dir>
# <wheel dir> holds the kumo-engine, kumo-core, kumo-api, kumo-pql, kumo-s3 and
# diskgraph wheels; --build-arg TORCH=cpu for CPU.
FROM python:3.12-slim
ARG TORCH=cu126
COPY *.whl /wheels/
# by path: PyPI's 1.0.0 predates the MLOps API
RUN pip install --no-cache-dir --find-links /wheels \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH}" \
      "$(ls /wheels/kumo_engine-*.whl)[mlflow,sagemaker,text]" \
      "torch==2.12.0+${TORCH}" \
 && python -c "import kumo.engine, torch; print(kumo.engine.__version__, torch.__version__)" \
 && rm -rf /wheels
ENV PYTHONUNBUFFERED=1 GIT_PYTHON_REFRESH=quiet MLFLOW_DISABLE_AGENT_HINT=1
