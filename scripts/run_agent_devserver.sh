#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the Agent development server in docker.
#
# Usage:
#   bash ./scripts/run_agent_devserver.sh <ui_port_number> <api_port_number> <label>

# This script needs to be run from the root of the Kumo repository.

set -xe

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/run_agent_devserver.sh <port_number>"
    false
fi

PORT_NUMBER=$1

# Diagnostic logging for SSH
echo "Initial SSH_AUTH_SOCK: $SSH_AUTH_SOCK"
echo "SSH Agent Status:"
ssh-add -l || true
echo "SSH Agent Process:"
ps aux | grep ssh-agent || true
echo "SSH Socket Symlink:"
ls -l $SSH_AUTH_SOCK || true
echo "Actual Socket File:"
ls -l $(readlink $SSH_AUTH_SOCK) || true

# Store original SSH_AUTH_SOCK
ORIGINAL_SSH_AUTH_SOCK=$(readlink -f $SSH_AUTH_SOCK)
echo "Using resolved socket path: $ORIGINAL_SSH_AUTH_SOCK"

# Make sure we don't point docker-cli to minikube's internal docker engine
eval $(minikube docker-env -u)

echo "SSH_AUTH_SOCK after minikube env: $SSH_AUTH_SOCK"

# Restore SSH_AUTH_SOCK if unset by minikube
export SSH_AUTH_SOCK=$ORIGINAL_SSH_AUTH_SOCK
echo "Restored SSH_AUTH_SOCK: $SSH_AUTH_SOCK"

# Build the image with SSH forwarding
DOCKER_BUILDKIT=1 docker build \
  --ssh default=$SSH_AUTH_SOCK \
  -t kumo-agent ./agent

# Run container with mounted code for development
docker run \
  -v `pwd`/agent:/app/agent \
  -w /app \
  -ti --rm \
  -p $PORT_NUMBER:$PORT_NUMBER \
  kumo-agent bash -c "\
    cd /app && \
    python3 -m agent.backend.serve --port $PORT_NUMBER"
