# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Look up a visible project without changing the active SDK project.

Set ``KUMO_API_URL``, ``KUMO_API_KEY``, and ``KUMO_PROJECT``. The project may
be a display name or immutable Kumo project ID.
"""
import os

import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]

kumo.init(url=API_URL, api_key=API_KEY)
default_project = kumo.current_project()

project = kumo.get_project(os.environ["KUMO_PROJECT"])
assert kumo.current_project() == default_project

# Stable IDs are suitable for automation and explicit project-scoped objects.
same_project = kumo.get_project(project.id)
assert same_project == project
