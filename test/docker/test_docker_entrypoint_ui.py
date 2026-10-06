# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRYPOINT_SCRIPT = REPO_ROOT / "docker" / "docker-entrypoint-ui.sh"
REAL_ENV_JS = REPO_ROOT / "ui" / "env.js"

ENV_JS_TEMPLATE = """\
// This will be replaced by the update-env script
const runningLocally = __UI_LOCAL__;

const port = runningLocally ? 3000 : 8000;

window._env_ = {
  REACT_APP_BASE_URL_DEV: runningLocally ? "/" : `http://127.0.0.1:${port}/`,
  REACT_APP_BASE_URL_PROD: runningLocally ? "/" : `http://127.0.0.1:${port}/`,
};
"""

ENV_JS_LEGACY_LITERAL = """\
const runningLocally = __UI_LOCAL__;

window._env_ = {
  REACT_APP_BASE_URL_DEV: "http://127.0.0.1:8000/",
  REACT_APP_BASE_URL_PROD: "http://127.0.0.1:8000/",
};
"""


def _run_entrypoint(tmp_path: pathlib.Path, env_js: str,
                    extra_env: dict[str, str]) -> str:
    assert ENTRYPOINT_SCRIPT.exists()

    opt_app = tmp_path / "opt" / "app"
    html_dir = tmp_path / "usr" / "share" / "nginx" / "html"
    opt_app.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    env_js_path = opt_app / "env.js"
    env_js_path.write_text(env_js)

    # Rewrite hardcoded container paths to tmpdir paths for hermetic testing.
    script_text = ENTRYPOINT_SCRIPT.read_text()
    script_text = script_text.replace("/opt/app/env.js", env_js_path.as_posix())
    script_text = script_text.replace("/usr/share/nginx/html",
                                      html_dir.as_posix())

    patched_script = tmp_path / "docker-entrypoint-ui.sh"
    patched_script.write_text(script_text)
    patched_script.chmod(0o755)

    env = os.environ.copy()
    env.pop("API_BASE_URL", None)
    env.pop("UI_LOCAL", None)
    env.update(extra_env)

    subprocess.run(
        ["sh", str(patched_script), "true"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    out_path = html_dir / "env-config.js"
    assert out_path.exists()
    return out_path.read_text()


def test_entrypoint_preserves_dynamic_port_when_api_base_url_unset(
        tmp_path: pathlib.Path) -> None:
    rendered = _run_entrypoint(
        tmp_path,
        ENV_JS_TEMPLATE,
        {"UI_LOCAL": "false"},
    )

    assert "__UI_LOCAL__" not in rendered
    assert "const runningLocally = false;" in rendered
    assert ('REACT_APP_BASE_URL_DEV: runningLocally ? "/" : '
            '`http://127.0.0.1:${port}/`') in rendered


def test_entrypoint_uses_same_origin_api_base_in_local_mode(
        tmp_path: pathlib.Path) -> None:
    rendered = _run_entrypoint(
        tmp_path,
        REAL_ENV_JS.read_text(),
        {"UI_LOCAL": "true"},
    )

    assert "const runningLocally = true;" in rendered
    assert 'REACT_APP_BASE_URL_DEV: runningLocally ? "/" :' in rendered
    assert 'REACT_APP_BASE_URL_PROD: runningLocally ? "/" :' in rendered


def test_entrypoint_rewrites_template_url_when_api_base_url_set(
        tmp_path: pathlib.Path) -> None:
    rendered = _run_entrypoint(
        tmp_path,
        ENV_JS_TEMPLATE,
        {
            "UI_LOCAL": "false",
            "API_BASE_URL": "/api/"
        },
    )

    assert "__UI_LOCAL__" not in rendered
    assert "const runningLocally = false;" in rendered
    assert 'REACT_APP_BASE_URL_DEV: runningLocally ? "/" : "/api/"' in rendered
    assert 'REACT_APP_BASE_URL_PROD: runningLocally ? "/" : "/api/"' in rendered


def test_entrypoint_rewrites_legacy_literal_url_when_api_base_url_set(
        tmp_path: pathlib.Path) -> None:
    rendered = _run_entrypoint(
        tmp_path,
        ENV_JS_LEGACY_LITERAL,
        {
            "UI_LOCAL": "false",
            "API_BASE_URL": "/api/"
        },
    )

    assert "__UI_LOCAL__" not in rendered
    assert "const runningLocally = false;" in rendered
    assert 'REACT_APP_BASE_URL_DEV: "/api/"' in rendered
    assert 'REACT_APP_BASE_URL_PROD: "/api/"' in rendered


@pytest.mark.parametrize(
    "api_base_url",
    [
        "https://api.example.com/v1?a=1&b=2",
        "https://api.example.com/p|q",
    ],
)
def test_entrypoint_handles_sed_metacharacters_in_api_base_url(
    tmp_path: pathlib.Path,
    api_base_url: str,
) -> None:
    rendered = _run_entrypoint(
        tmp_path,
        ENV_JS_TEMPLATE,
        {
            "UI_LOCAL": "false",
            "API_BASE_URL": api_base_url
        },
    )

    assert (f'REACT_APP_BASE_URL_DEV: runningLocally ? "/" : '
            f'"{api_base_url}"') in rendered
    assert (f'REACT_APP_BASE_URL_PROD: runningLocally ? "/" : '
            f'"{api_base_url}"') in rendered


def test_entrypoint_smoke_real_env_js_with_api_base_url(
        tmp_path: pathlib.Path) -> None:
    real_content = REAL_ENV_JS.read_text()
    # Guard: substitution must be relevant to the real file, not a no-op
    assert "`http://127.0.0.1:${port}/`" in real_content

    rendered = _run_entrypoint(tmp_path, real_content, {
        "UI_LOCAL": "false",
        "API_BASE_URL": "/api/"
    })

    assert 'REACT_APP_BASE_URL_DEV: runningLocally ? "/" : "/api/"' in rendered
    assert 'REACT_APP_BASE_URL_PROD: runningLocally ? "/" : "/api/"' in rendered


def test_entrypoint_smoke_real_env_js_without_api_base_url(
        tmp_path: pathlib.Path) -> None:
    real_content = REAL_ENV_JS.read_text()
    # Guard: template literal must exist in the real file so it can be preserved
    assert "`http://127.0.0.1:${port}/`" in real_content

    rendered = _run_entrypoint(tmp_path, real_content, {"UI_LOCAL": "false"})

    assert ('REACT_APP_BASE_URL_DEV: runningLocally ? "/" : '
            '`http://127.0.0.1:${port}/`') in rendered


def test_entrypoint_renders_singlehost_same_origin_from_baked_env_js(
        tmp_path: pathlib.Path) -> None:
    # Single-host drops the env.singlehost.js bind and instead sets API_BASE_URL
    # so the baked env.js renders a same-origin API base (nginx proxies REST on
    # the published port) with auth honoring the AUTH_DISABLED env.
    rendered = _run_entrypoint(
        tmp_path,
        REAL_ENV_JS.read_text(),
        {
            "API_BASE_URL": "/",
            "AUTH_DISABLED": "false"
        },
    )

    assert 'REACT_APP_BASE_URL_PROD: runningLocally ? "/" : "/"' in rendered
    assert 'AUTH_DISABLED: false' in rendered
