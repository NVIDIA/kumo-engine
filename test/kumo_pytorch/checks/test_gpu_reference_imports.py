"""The vendored reference must remain test-local and safe to import without CUDA."""

import ast
import subprocess
import sys
from pathlib import Path


def test_reference_import_is_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from test.kumo_pytorch.reference_engine import gpu; "
                "assert callable(gpu.prepare); "
                "assert not ({'cupy', 'cudf', 'rmm', 'pylibcugraph'} & sys.modules.keys())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reference_imports_resolve_inside_test_suite():
    root = Path(__file__).parents[1] / "reference_engine/gpu"
    prefix = "test.kumo_pytorch.reference_engine.gpu"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                assert not name.startswith(
                    ("kumo_pytorch_gpu", "kumo_specs", "profiling")
                ), (path, name)
                if name.startswith(prefix):
                    target = root.joinpath(
                        *name.removeprefix(prefix).strip(".").split(".")
                    )
                    assert target.is_dir() or target.with_suffix(".py").is_file(), (
                        path,
                        name,
                    )
