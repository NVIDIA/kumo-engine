# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_OSS_READINESS = REPO_ROOT / "scripts" / "check_oss_readiness.sh"
CHILD_CHECKS = (
    ("check_public_references.py", "KUMO_TEST_PUBLIC_STATUS"),
    ("check_dead_code.sh", "KUMO_TEST_DEAD_STATUS"),
)


def _test_scripts(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    wrapper = scripts / "check_oss_readiness.sh"
    shutil.copy2(CHECK_OSS_READINESS, wrapper)

    for name, status_variable in CHILD_CHECKS:
        child = scripts / name
        child.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s|%s\\n' \"${0##*/}\" \"$*\" >> \"$KUMO_OSS_TEST_LOG\"\n"
            f'exit "${{{status_variable}:-0}}"\n',
            encoding="utf-8",
        )
        child.chmod(0o755)

    log = tmp_path / "calls.txt"
    environment = os.environ.copy()
    environment["KUMO_OSS_TEST_LOG"] = str(log)
    return wrapper, log, environment


def test_diff_mode_runs_every_check_after_a_failure(tmp_path: Path) -> None:
    wrapper, log, environment = _test_scripts(tmp_path)
    environment["KUMO_TEST_PUBLIC_STATUS"] = "1"

    result = subprocess.run(
        [str(wrapper), "--diff", "base", "head"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert log.read_text(encoding="utf-8").splitlines() == [
        "check_public_references.py|--diff base head",
        "check_dead_code.sh|",
    ]
