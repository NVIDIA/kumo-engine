# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image for the pipeline's Fit and Predict steps; see README.md.
# Build context: a directory holding the kumo-engine wheel and its kumo-core,
# kumo-api, kumo-pql, kumo-s3 and diskgraph wheels.
#   GPU: docker build -f examples/mlops/sagemaker/Dockerfile -t kumo-engine-sagemaker <wheel dir>
#   CPU: add --build-arg TORCH=cpu
FROM python:3.12-slim
ARG TORCH=cu126
COPY *.whl /wheels/
# The engine wheel is installed by path: PyPI's unrelated "kumo-engine" would
# win over a dev build. Once kumo-engine is on the index, install
# "kumo-engine[mlflow,sagemaker,text]" instead and drop the wheels.
RUN pip install --no-cache-dir --find-links /wheels \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH}" \
      "$(ls /wheels/kumo_engine-*.whl)[mlflow,sagemaker,text]" \
      "torch==2.12.0+${TORCH}" \
 && python -c "import kumo.engine, torch; print(kumo.engine.__version__, torch.__version__)" \
 && rm -rf /wheels
ENV PYTHONUNBUFFERED=1 GIT_PYTHON_REFRESH=quiet MLFLOW_DISABLE_AGENT_HINT=1
