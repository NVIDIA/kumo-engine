"""Create a dataset-specific directory below an explicit local SSD root."""
from pathlib import Path


def dataset_scratch(root: Path, dataset: str) -> Path:
    if not dataset or Path(dataset).name != dataset or dataset in {'.', '..'}:
        raise ValueError('Use a single dataset directory name.')
    path = root.expanduser().resolve() / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path
