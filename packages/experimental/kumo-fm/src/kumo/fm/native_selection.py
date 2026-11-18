"""Inclusive whole-second graph cutoff for native anchor tables."""
from datetime import timedelta

from .workload import EPOCH


def inclusive_second(anchor):
    return (anchor - EPOCH) // timedelta(seconds=1)
