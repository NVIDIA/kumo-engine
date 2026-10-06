# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any, Dict, Optional

from io_utils.secrets import get_secrets

_DEFAULT_FIRSTBOOT_SECRET_DIR = "/opt/kumo/secrets"


def _candidate_secret_paths(secret_name: str) -> list[str]:
    explicit_path = os.getenv("KUMO_FIRSTBOOT_SECRET_PATH")
    if explicit_path:
        return [explicit_path]
    secret_dir = os.getenv("KUMO_FIRSTBOOT_SECRET_DIR",
                           _DEFAULT_FIRSTBOOT_SECRET_DIR)
    return [os.path.join(secret_dir, secret_name)]


def load_firstboot_secret(secret_name: str) -> Optional[Dict[str, Any]]:
    """Read a firstboot JSON secret file if it is mounted locally."""
    for secret_path in _candidate_secret_paths(secret_name):
        if not os.path.exists(secret_path):
            continue
        with open(secret_path, encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Firstboot secret file must contain a JSON object: {secret_path}"
            )
        return payload
    return None


def get_secret_prefer_firstboot(secret_name: str) -> Dict[str, Any]:
    """Load local firstboot secret JSON before falling back to configured secrets."""
    local_secret = load_firstboot_secret(secret_name)
    if local_secret is not None:
        return local_secret
    return get_secrets(secret_name)
