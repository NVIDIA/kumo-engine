# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency identity in standalone installs and the merged engine wheel."""
import importlib.metadata
import json


def dependency_provenance(name):
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        engine = importlib.metadata.distribution("kumo-engine")
        manifest = engine.locate_file("kumo_bundle_manifest.json")
        components = json.loads(manifest.read_text())["components"]
        for component in components:
            if component["name"] == name:
                return {"version": component["version"],
                        "source": {"bundle": engine.version,
                                   "origin": component["origin"]}}
        raise importlib.metadata.PackageNotFoundError(name) from None
    return {"version": distribution.version,
            "source": json.loads(distribution.read_text("direct_url.json") or "null")}
