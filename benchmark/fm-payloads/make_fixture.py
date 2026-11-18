"""Create the frozen smoke case used by the benchmark and SDM notebook."""

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[2] / "packages/experimental/kumo-fm/tests"
sys.path.insert(0, str(TESTS))

from test_integration import fixture  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_fixture.py /path/to/new/case")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    _, workload = fixture(root)
    (root / "supported").rename(root / "bundle")
    (root / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    print(f"bundle: {root / 'bundle'}")
    print(f"workload: {root / 'workload.json'}")


if __name__ == "__main__":
    main()
