"""Dataset/model-independent benchmark interfaces, owned by the test suite."""

from .interface import BatchCounts, Workload
from .runner import run

__all__ = ["BatchCounts", "Workload", "run"]
