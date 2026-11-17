# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

from tests import api_reference


def test_api_reference_is_current() -> None:
    stale = api_reference.stale_files(api_reference.generate())
    assert not stale, (
        f"docs/api/ is out of date with the docstrings: {stale}. "
        "Regenerate it with: python -m tests.api_reference --write"
    )


def test_every_public_symbol_is_documented() -> None:
    api_reference.generate()
    undocumented = []
    for name, obj in api_reference.public_objects():
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if api_reference.own_doc(obj) is None:
            undocumented.append(name)
        if inspect.isclass(obj):
            for member, raw, _ in api_reference.members(obj):
                if api_reference.member_doc(raw) is None:
                    undocumented.append(f"{name}.{member}")
    assert not undocumented, f"public symbols without a docstring: {undocumented}"


def test_anchor_matches_github() -> None:
    assert api_reference.anchor("`DataSchema`") == "dataschema"
    assert api_reference.anchor("`Artifact.load`") == "artifactload"


def test_the_diskgraph_limitations_list_is_current() -> None:
    from tests import limitations_reference

    text = limitations_reference.LIMITATIONS.read_text()
    assert limitations_reference.generate(text) == text, (
        "docs/limitations.md is out of date with DiskGraphEngine.unsupported. "
        "Regenerate it with: python -m tests.limitations_reference --write"
    )
