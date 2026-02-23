import pytest
import os

@pytest.fixture(scope="session")
def ifood_graph():
    # Use the verified ifood dataset we've been using for benchmarks
    path = "/mnt/data/data/index/ifood_bp-job-8dd9d07e93bf4dbf8fe2845687bd96ef_index"
    if not os.path.exists(path):
        pytest.skip(f"ifood dataset not found at {path}")
    return path
