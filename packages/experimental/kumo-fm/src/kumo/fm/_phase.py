"""Small phase boundary for shared assembly code."""
from contextlib import nullcontext


def phase(*_args, **_kwargs):
    return nullcontext()
