#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

DOCKER_COMPOSE_VERSION=1.25.0
TEMPORAL_LOCAL_VERSION=v1.19.0
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

if DEPLOY_STAGE=dev TEMPORAL_SERVER=localhost python temporalio_commons/wait_server_ready.py -t 1; then
    echo "temporal server is already running"
else
    if [ ! -f /usr/bin/docker-compose ]; then
        sudo curl -L https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-${UNAME_S}-${UNAME_M} -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    if [ ! -d ~/code/temporalio/docker-compose ]; then
        git clone https://github.com/temporalio/docker-compose.git ~/code/temporalio/docker-compose
    fi
    cd ~/code/temporalio/docker-compose
    git checkout main && git fetch --all --tags && git checkout tags/${TEMPORAL_LOCAL_VERSION}
    tmux kill-session -t local_temporal || true
    tmux new-session -d -s local_temporal
    tmux send-keys -t local_temporal 'docker-compose up' ENTER
    echo "Local Temporal Cluster is up and running in 'local_temporal' tmux session"
fi
