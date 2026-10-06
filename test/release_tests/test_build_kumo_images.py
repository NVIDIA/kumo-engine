# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which image lineup gets built, and from which ref.

kumo2 (v1.x/v2.x) and kumo3 (v3.0+) publish different image sets. The Python
builder remains shared by active Kumo 3 release scripts and private operational
callers, so its ref and explicit-application resolution stay covered here.
"""
import pytest

from scripts.release.build_kumo_images import (
    DEFAULT_APPLICATION,
    Application,
    Platform,
    WorkflowFile,
    gather_flags,
    images,
    resolve_application,
)

# A ref that carries no version: nothing to derive from.
SHA = "0123456789abcdef0123456789abcdef01234567"

SAAS_LINEUP = {
    Application.KUMO: [
        WorkflowFile.WORKER,
        WorkflowFile.REST,
        WorkflowFile.TEMPORAL_WORKER,
        WorkflowFile.UI,
    ],
    # cpu, gpu and ui are built as one set by a single orchestrator workflow,
    # so a deployable ref always has all three.
    Application.KUMO3: [
        WorkflowFile.KUMO3_IMAGES,
    ],
}


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # Patch tags supported by the builder's application inference.
        ("v1.9.4", Application.KUMO),
        ("v2.21.35", Application.KUMO),
        ("v3.0.0", Application.KUMO3),
        ("v10.0.0", Application.KUMO3),
        # Floating minor tags / release branch names, built alongside the patch tag.
        ("v2.21", Application.KUMO),
        ("v3.0", Application.KUMO3),
        # Refs that carry no version fall back to the default rather than guessing.
        (SHA, DEFAULT_APPLICATION),
        ("master", DEFAULT_APPLICATION),
        # Nonstandard tags must not be read as Kumo release refs.
        ("other-v1.0.0", DEFAULT_APPLICATION),
        # Not release-shaped, even though it starts with a v.
        ("v3-experiment", DEFAULT_APPLICATION),
    ])
def test_ref_selects_the_product_line_and_its_lineup(ref, expected):
    app = resolve_application(None, ref)
    assert app == expected
    # The mapping only matters because it selects a different set of workflows.
    # Close that loop: ref in, image lineup out.
    assert images[app][Platform.SAAS] == SAAS_LINEUP[app]


@pytest.mark.parametrize(
    "ref", ["v2.21.35", "v3.0.0", "v10.0.0", SHA, "master", "other-v1.0.0"])
@pytest.mark.parametrize("explicit", [Application.KUMO, Application.KUMO3])
def test_explicit_app_always_wins(ref, explicit):
    # Release callers pass `--app` explicitly. Derivation must never override
    # an explicit application selection.
    assert resolve_application(explicit, ref) == explicit


def test_kumo3_release_bakes_complete_default_glove_bundle():
    assert dict(
        gather_flags(
            Application.KUMO3,
            Platform.SAAS,
            WorkflowFile.KUMO3_IMAGES,
            object(),
        ))["bake_glove_level"] == "basic"


def test_acr_publish_flag_is_passed_to_the_kumo3_builder():
    class Args:
        publish_to_acr = True

    flags = dict(
        gather_flags(
            Application.KUMO3,
            Platform.SAAS,
            WorkflowFile.KUMO3_IMAGES,
            Args(),
        ))
    assert flags["publish_to_acr"] == "true"
