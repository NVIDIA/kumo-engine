# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

FORBIDDEN_SYMBOL = "DBFileStorageDAO"
TARGET_MODULES = (
    Path("test/rest/driver/test_table_driver.py"),
)


def test_driver_tests_do_not_depend_on_db_file_storage_dao() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    offenders: list[str] = []

    for relative_path in TARGET_MODULES:
        file_path = repo_root / relative_path
        contents = file_path.read_text(encoding="utf-8")
        if FORBIDDEN_SYMBOL in contents:
            offenders.append(str(relative_path))

    assert not offenders, (
        f"Forbidden dependency '{FORBIDDEN_SYMBOL}' found in: {', '.join(offenders)}"
    )
