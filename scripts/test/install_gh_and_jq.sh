#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "DEPRECATED: scripts/test/install_gh_and_jq.sh is being replaced by mise."
echo "Use 'mise install' for local development."

if ! command -v jq &> /dev/null; then
    # Install jq using curl
    echo "jq is not found. Installing jq..."

    # Download jq binary
    curl -LO https://github.com/stedolan/jq/releases/latest/download/jq-linux64

    # Make the downloaded file executable
    chmod +x jq-linux64

    # Move jq binary to a directory in the PATH
    mv jq-linux64 /usr/local/bin/jq
fi

if ! command -v gh &> /dev/null; then
    if [[ $OSTYPE == "darwin"* ]]; then
        echo "Installing github cli"
        brew install gh
    else
        type -p yum-config-manager >/dev/null -q -y || sudo yum install yum-utils -q -y
        sudo yum-config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo -y -q
        sudo yum install gh -y -q --nogpgcheck
    fi
fi
