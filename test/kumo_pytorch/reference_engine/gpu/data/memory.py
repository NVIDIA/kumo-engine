"""GPU memory and device setup shared by everything in `test.kumo_pytorch.reference_engine.gpu`.

`ensure_pool` makes libcudf's (and so pylibcudf's) and CuPy's allocations come
from one RMM resource that pools memory. Without a pool, each allocation is a
``cudaMalloc`` and each free makes the whole GPU wait: the old GPU backend
measured small CuPy calls at 247 ms with other GPU work queued, and 0.1 ms with
a pool.

The pool is CUDA's stream-ordered allocator (``cudaMallocAsync``), which keeps
no unused memory past a synchronization (`RELEASE_THRESHOLD_BYTES`). PyTorch
allocates batches and models with its own caching allocator beside it, so memory
a pool keeps is memory PyTorch cannot have. An RMM pool that only grew held
80 GiB after a split dataset's fitting while 38 GiB of it was in use, and a
padded batch of 4.5 GiB then failed. The other way round, when one of the
pool's allocations fails while PyTorch caches unused blocks, those are released
and the allocation is tried again (`_release_torch_cache`).

One GPU per process: cuDF faults on a second GPU in the same process, so the
first GPU `test.kumo_pytorch.reference_engine.gpu` works on is kept, and work with another one current is
refused (`device`).

Nothing here is imported by ``import test.kumo_pytorch.reference_engine.gpu``: RMM and CuPy load on first use.
"""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["RELEASE_THRESHOLD_BYTES", "device", "ensure_pool", "free_bytes", "release_unused"]

#: Unused memory the pool keeps past a synchronization: none, so that PyTorch can allocate what the data freed.
#: (Measured on AML HI-small: fitting and delivery took as long as with a pool that keeps everything.)
RELEASE_THRESHOLD_BYTES = 0

#: The starting size of the RMM pool put over a caller's own non-pooling resource (a limit, say); it grows on
#: demand and does not give memory back.
INITIAL_POOL_BYTES = 1 << 30

_DEVICE: int | None = None


def device() -> int:
    """The GPU this process works on: the current one at the first call; another one is refused."""

    import cupy

    global _DEVICE
    current = cupy.cuda.Device().id
    if _DEVICE is None:
        _DEVICE = current
    elif current != _DEVICE:
        raise RuntimeError(
            f"test.kumo_pytorch.reference_engine.gpu already works on GPU {_DEVICE} in this process, and GPU {current} is now current: "
            "cuDF fails on a second GPU in one process, so use one process per GPU"
        )
    return current


def ensure_pool() -> None:
    """Make the current GPU's allocations come from one pooling RMM resource, shared with CuPy. Idempotent.

    With RMM's default resource (plain ``cudaMalloc``), that is a ``cudaMallocAsync`` pool that releases its
    unused memory at each synchronization, behind an adaptor that releases PyTorch's cached blocks when one of
    its allocations fails. A caller's own resource is kept when it already hands out memory without waiting (a
    pool, an arena or CUDA's asynchronous allocator, also behind adaptors such as a limit or statistics). Any
    other resource (a limit over plain memory, say) becomes the upstream of an RMM pool, so its limit still
    holds. CuPy is pointed at RMM unless the caller chose its own allocator.
    """

    import cupy
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    device()
    current = rmm.mr.get_current_device_resource()
    if not _pools(current):
        if type(current) is rmm.mr.CudaMemoryResource:
            pool = rmm.mr.FailureCallbackResourceAdaptor(
                rmm.mr.CudaAsyncMemoryResource(release_threshold=RELEASE_THRESHOLD_BYTES), _release_torch_cache)
        else:
            try:
                pool = rmm.mr.PoolMemoryResource(current, initial_pool_size=INITIAL_POOL_BYTES)
            except MemoryError:  # a limit under the starting size: start small, grow on demand
                pool = rmm.mr.PoolMemoryResource(current, initial_pool_size=1 << 20)
        rmm.mr.set_current_device_resource(pool)
    if cupy.cuda.get_allocator() == cupy.get_default_memory_pool().malloc:
        cupy.cuda.set_allocator(rmm_cupy_allocator)


def release_unused() -> None:
    """Give the pool's unused memory back to CUDA now, for PyTorch to allocate.

    The asynchronous pool releases it at a synchronization, which PyTorch's own
    allocations do not make: called where much data was just freed (a streamed
    partition, a group's build, a fitting) before batches go to the consumer.
    Nothing to do before the pool is set up.
    """

    import cupy

    if _DEVICE is not None:
        cupy.cuda.runtime.deviceSynchronize()


def free_bytes() -> int:
    """The GPU's free memory, counting memory a pool or PyTorch holds unused as free.

    CUDA does not count as free what a pool keeps, though the pool can hand it
    out: an RMM pool's whole size is counted (call this before allocating), and
    the asynchronous pool keeps nothing unused past the synchronization done
    here. PyTorch's cached, unused blocks are counted too: the pool's
    allocations release them when they need them (`ensure_pool`).
    """

    import cupy
    import rmm

    cupy.cuda.runtime.deviceSynchronize()  # the asynchronous pool gives its unused memory back here
    free = int(cupy.cuda.runtime.memGetInfo()[0]) + _torch_cached()
    resource = rmm.mr.get_current_device_resource()
    while resource is not None:
        if isinstance(resource, rmm.mr.PoolMemoryResource):
            return free + int(resource.pool_size())
        resource = _upstream(resource)
    return free


def _release_torch_cache(nbytes: int) -> bool:
    """RMM's callback when an allocation of `nbytes` fails: release PyTorch's cached, unused GPU blocks, and
    try again if that gave any memory back (each retry follows a release, so retries end)."""

    if not _torch_cached():
        return False
    torch = sys.modules["torch"]
    before = torch.cuda.memory_reserved(_DEVICE)
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved(_DEVICE) < before


def _torch_cached() -> int:
    """Bytes PyTorch's caching allocator holds unused on this process's GPU; 0 when PyTorch has not used CUDA
    (it is not imported here for that)."""

    torch = sys.modules.get("torch")
    if torch is None or _DEVICE is None or not torch.cuda.is_initialized():
        return 0
    return max(0, int(torch.cuda.memory_reserved(_DEVICE)) - int(torch.cuda.memory_allocated(_DEVICE)))


def _pools(resource: Any) -> bool:
    """Whether `resource`, or one it wraps, hands out memory without waiting."""

    import rmm

    kinds = (rmm.mr.PoolMemoryResource, rmm.mr.ArenaMemoryResource, rmm.mr.CudaAsyncMemoryResource,
             rmm.mr.CudaAsyncViewMemoryResource)
    while resource is not None:
        if isinstance(resource, kinds):
            return True
        resource = _upstream(resource)
    return False


def _upstream(resource: Any) -> Any:
    getter = getattr(resource, "get_upstream", None)
    return getter() if getter is not None else None
