"""Dataset, index, request, and batch lifecycle."""
from __future__ import annotations

import json
from pathlib import Path

from ._pytorch import KumoPytorchBackend
from .batch import PayloadBatch
from .bundle import Bundle
from .workload import Workload


def _select(available, selected, split):
    if selected is None:
        return list(available)
    if not isinstance(selected, (list, tuple)):
        raise TypeError(f"{split} must be an ordered sequence of instance IDs")
    if any(type(value) is not int for value in selected):
        raise ValueError(f"{split} instance IDs must be int64 values")
    by_id = {seed["instance_id"]: seed for seed in available}
    if len(set(selected)) != len(selected) or not set(selected) <= by_id.keys():
        raise ValueError(f"{split} has duplicate or unknown instance IDs")
    return [by_id[value] for value in selected]


class PreparedRequest:
    """Resolved anchors; every load call samples exactly once per included split."""

    def __init__(self, loader, splits):
        self._loader = loader
        self._splits = splits
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._closed = True

    def load(self):
        if self._closed or self._loader._closed:
            raise RuntimeError("Prepared request or loader is closed")
        workload = self._loader.workload
        ready = {
            split: (seeds, self._loader._backend.extract(workload, seeds))
            for split, seeds in self._splits.items()
        }
        return PayloadBatch(workload, ready)


class FMLoader:
    """Open a validated bundle and its compatible reusable DiskGraph index."""

    def __init__(self, dataset, spec, *, index, threads=2, verify_files=True):
        bundle = Bundle(dataset.root if isinstance(dataset, Bundle) else Path(dataset),
                        verify_files=verify_files)
        if isinstance(spec, Workload):
            if spec.bundle.root != bundle.root:
                raise ValueError("Workload is bound to another bundle")
            workload = spec if spec.bundle is bundle else Workload(bundle, spec.spec)
        else:
            document = json.loads(Path(spec).read_text()) if isinstance(spec, (str, Path)) else spec
            workload = Workload(bundle, document)
        if verify_files:
            bundle.validate(threads=threads)
        self.workload = workload
        self._backend = KumoPytorchBackend(workload, index, threads=threads)
        self._closed = False

    @property
    def preparation_mode(self):
        return self._backend.preparation_mode

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            self._backend.close()

    def materialize(self, *, context=None, predict=None):
        if self._closed:
            raise RuntimeError("Loader is closed")
        available = self.workload.instances
        splits = {}
        for split, selected in (("context", context), ("predict", predict)):
            seeds = _select(available[split], selected, split)
            if seeds:
                if len(seeds) > self.workload.budgets["max_instances"]:
                    raise ValueError("Instance budget exceeded")
                self._backend.prepare_batch(seeds)
                splits[split] = seeds
        if not splits:
            raise ValueError("Request needs at least one split")
        return PreparedRequest(self, splits)


def prepare(dataset, spec, *, index, threads=2, verify_files=True):
    return FMLoader(dataset, spec, index=index, threads=threads, verify_files=verify_files)
