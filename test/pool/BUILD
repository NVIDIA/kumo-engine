# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

python_tests(
    name="test_pool",
    # Exclude __init__.py — pytest returns exit 5 ("no tests collected")
    # when pointed at it, which Pants treats as a test failure.
    # Tests for kumo-importing modules (test_activities.py,
    # test_worker_main.py) are excluded for the same reason ``pool/BUILD``
    # excludes their sources — they need the pip-installed kumo stack.
    # test_activities.py is an explicit matrix entry in testing.yml.
    sources=[
        "test_*.py",
        "!test_activities.py",
        "!test_worker_main.py",
        "!test_provision_terminal_nonretry.py",  # imports pool.wf_coordinator / pool.activities (not Pants-managed)
        "!test_wf_coordinator.py",  # imports pool.wf_coordinator / pool.activities (not Pants-managed)
    ],
    dependencies=[
        "//pool",
        "//io_utils:serde",             # save_json, load_json
        "//io_utils/fsutil:impl_local",  # tests write to tmp_path
        "//3rdparty:pytest_asyncio",    # pytest itself comes via resolve set
    ],
)
