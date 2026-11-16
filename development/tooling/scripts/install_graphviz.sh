#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xe

pushd $(mktemp -d )
  if [[ $OSTYPE == "darwin"* ]]; then
    brew install graphviz
  elif [[ $OSTYPE == "linux-gnu"* ]]; then
    sudo apt install graphviz
  else
    echo "Unsupported OS: $OSTYPE"
  fi
popd
