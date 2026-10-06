# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_sources(
    name="pool",
    # ``activities.py``, ``training_progress.py``, ``diskgraph_args.py``,
    # ``wf_coordinator.py``, and ``worker_main.py`` are intentionally NOT
    # Pants-managed: they import
    # ``kumo.*`` (diskgraph engine and training/prediction config),
    # which would pull torch + connectors into Pants targets. Per
    # KUMO_BUILD.md, anything importing ``kumo.*`` is pip-installed; only
    # the controller / RPC layer / strategy / task-queue / dataclass files
    # are Pants-onboarded.
    sources=[
        "*.py",
        "!activities.py",
        "!training_progress.py",
        "!diskgraph_args.py",
        "!wf_coordinator.py",
        "!worker_main.py",
    ],
    dependencies=[
        "//io_utils",                                # kumo_dataclass
        "//io_utils:logging",
        "//io_utils:serde",                          # save_json, load_json
        "//io_utils/fsutil",                         # fsutil.exists / read_text
        "//monitoring",                              # tracing setup
        "//pool/worker_manager",                     # configured k8s strategy
        "//service_framework:core",                  # EnvVar
        "//service_framework:deploy_env_config",      # DiskgraphConfig
        "//service_framework:deploy_env_config_v2",  # DeployEnvConfig
        "//temporalio_commons",                      # auto_rpc, auto_rpc_config
        "//3rdparty:prometheus_client",              # pool/metrics.py
        "//3rdparty:pydantic",
        "//3rdparty:temporalio",
    ],
)
