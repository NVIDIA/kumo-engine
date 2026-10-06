# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "ui/scripts/generate-third-party-notice.mjs"


def test_notice_covers_installed_production_packages_and_fails_closed(
        tmp_path: Path) -> None:
    packages = {
        "": {
            "name": "fixture",
            "version": "1.0.0",
        },
        "node_modules/alpha": {
            "version": "1.0.0",
            "license": "MIT",
        },
        "node_modules/beta": {
            "version": "2.0.0",
            "license": "MIT",
        },
        "node_modules/dev-only": {
            "version": "3.0.0",
            "dev": True,
        },
        "node_modules/css-only": {
            "version": "1.0.0",
            "dev": True,
        },
        "node_modules/missing-platform": {
            "version": "4.0.0",
            "optional": True,
        },
        "node_modules/local": {
            "link": True,
            "resolved": "../local",
        },
    }
    (tmp_path / "package-lock.json").write_text(
        json.dumps({
            "lockfileVersion": 3,
            "packages": packages
        }),
        encoding="utf-8",
    )
    legal_text = "MIT License\n\nCopyright Example"
    for name, version in (
        ("alpha", "1.0.0"),
        ("beta", "2.0.0"),
        ("dev-only", "3.0.0"),
        ("css-only", "1.0.0"),
    ):
        directory = tmp_path / "node_modules" / name
        directory.mkdir(parents=True)
        (directory / "package.json").write_text(
            json.dumps({
                "license": "MIT",
                "name": name,
                "version": version,
            }),
            encoding="utf-8",
        )
    (tmp_path / "node_modules/alpha/LICENSE").write_text(
        legal_text, encoding="utf-8")
    (tmp_path / "node_modules/beta/NOTICE").write_text(
        "Beta attribution", encoding="utf-8")
    (tmp_path / "node_modules/dev-only/LICENSE").write_text(
        legal_text, encoding="utf-8")
    (tmp_path / "node_modules/css-only/LICENSE").write_text(
        legal_text, encoding="utf-8")
    override_license = tmp_path / "third_party/licenses/shared.LICENSE"
    override_license.parent.mkdir(parents=True)
    override_license.write_text(legal_text, encoding="utf-8")
    overrides = {
        "packages": {
            "beta@2.0.0": "shared",
        },
        "sources": {
            "shared": {
                "license": "MIT",
                "path": "third_party/licenses/shared.LICENSE",
                "sha256": hashlib.sha256(legal_text.encode()).hexdigest(),
            }
        },
    }
    override_inventory = tmp_path / "third_party/npm-license-overrides.json"
    override_inventory.write_text(json.dumps(overrides), encoding="utf-8")

    output = tmp_path / "THIRD-PARTY-NPM.txt"
    subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--output",
         str(output)],
        check=True,
    )
    notice = output.read_text(encoding="utf-8")
    assert "alpha 1.0.0" in notice
    assert "beta 2.0.0" in notice
    assert "dev-only" not in notice
    assert "missing-platform" not in notice
    assert "local" not in notice
    assert notice.count("Copyright Example") == 1
    assert "Beta attribution" in notice

    source_maps = tmp_path / "build/source-maps"
    source_maps.mkdir(parents=True)
    (source_maps / "app.js.map").write_text(
        json.dumps({"sources": ["../../node_modules/alpha/index.js"]}),
        encoding="utf-8",
    )
    css_root = tmp_path / "src"
    css_root.mkdir()
    (css_root / "app.ts").write_text(
        'import "css-only/styles.css";\n', encoding="utf-8")
    override_inventory.write_text(
        json.dumps({"packages": {}, "sources": {}}), encoding="utf-8")
    subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--source-map",
         str(source_maps), "--css-root",
         str(css_root), "--output",
         str(output)],
        check=True,
    )
    notice = output.read_text(encoding="utf-8")
    assert "alpha 1.0.0" in notice
    assert "beta 2.0.0" not in notice
    assert "css-only 1.0.0" in notice

    override_inventory.write_text(json.dumps(overrides), encoding="utf-8")
    subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--source-map",
         str(source_maps), "--include-packages",
         "beta", "--output",
         str(output)],
        check=True,
    )
    notice = output.read_text(encoding="utf-8")
    assert "alpha 1.0.0" in notice
    assert "beta 2.0.0" in notice

    subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--include-packages",
         "dev-only", "--output",
         str(output)],
        check=True,
    )
    notice = output.read_text(encoding="utf-8")
    assert "alpha 1.0.0" in notice
    assert "beta 2.0.0" in notice
    assert "dev-only 3.0.0" in notice

    alpha_license = tmp_path / "node_modules/alpha/LICENSE"
    alpha_license.unlink()
    (tmp_path / "node_modules/alpha/license.js").write_text(
        "export const license = 'MIT';\n", encoding="utf-8")
    failed = subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--output",
         str(output)],
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 1
    assert "alpha@1.0.0 has no legal text or reviewed override" in failed.stderr
    (tmp_path / "node_modules/alpha/license.js").unlink()
    alpha_license.write_text(legal_text, encoding="utf-8")

    del overrides["packages"]["beta@2.0.0"]
    override_inventory.write_text(json.dumps(overrides), encoding="utf-8")
    failed = subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--output",
         str(output)],
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 1
    assert "beta@2.0.0 has no legal text or reviewed override" in failed.stderr

    (tmp_path / "node_modules/beta/package.json").write_text(
        json.dumps({
            "name": "beta",
            "version": "2.0.0"
        }),
        encoding="utf-8",
    )
    failed = subprocess.run(
        ["node",
         str(SCRIPT), "--root",
         str(tmp_path), "--output",
         str(output)],
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 1
    assert "beta@2.0.0 has no declared license" in failed.stderr


def test_ui_image_generates_a_notice_for_catalog_only_dependencies() -> None:
    ui_lock = json.loads((REPO_ROOT / "ui/package-lock.json").read_text())
    catalog_lock = json.loads(
        (REPO_ROOT / "packages/kumo-ui-catalog/package-lock.json").read_text())
    assert "node_modules/@xyflow/react" not in ui_lock["packages"]
    assert "node_modules/@xyflow/react" in catalog_lock["packages"]
    for name in (
        "tailwindcss",
        "tailwindcss-animate",
        "vite-plugin-css-injected-by-js",
    ):
        assert f"node_modules/{name}" in catalog_lock["packages"]

    dockerfile = (REPO_ROOT / "docker/Dockerfile.ui").read_text()
    assert "--root /opt/packages/kumo-ui-catalog" in dockerfile
    assert "--source-map /opt/app/stacktracer/source_map" in dockerfile
    assert "--source-map /opt/packages/kumo-ui-catalog/dist/index.es.js.map" in dockerfile
    assert "--root /opt/app" in dockerfile
    assert "--css-root /opt/app/src" in dockerfile
    assert "--css-root /opt/packages/kumo-ui-catalog/src" in dockerfile
    assert "--include-packages" not in dockerfile
