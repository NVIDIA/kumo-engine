# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable subprocess driver for the DiskGraph ingest binary.

Replaces the in-process pybind call from ``run_diskgraph_ingest`` with a
``subprocess.Popen`` of the standalone ``ingest`` binary so we can:

  * Enforce a per-process RAM cap via ``RLIMIT_AS`` (the binary gets killed
    cleanly when it tries to exceed the cap; in-process pybind cannot do
    this because rlimit is process-wide).
  * Halve thread count of the failing phase on OOM-class failures and
    retry with ``--resume-from-phase`` so completed phases (context,
    encode, ...) are not redone after a pk_map OOM, and phases that are
    not memory-pressure-bound keep their full parallelism.
  * Classify failure by subprocess exit code: with the diskgraph workspace's
    ``panic = "abort"`` profile, every OOM path (allocator failure, panic,
    RLIMIT_AS hit) exits with SIGABRT (134) and the kernel OOM-killer sends
    SIGKILL (137). Anything else is an application error and should not be
    retried.

The retry loop is intentionally inside this wrapper so every caller gets the
same phase-aware recovery behavior. Progress reporting is an injected callback;
this module has no workflow-runtime dependency.

Per-phase contract: the binary takes one ``--<phase>-threads`` flag per
phase (context, encode, pk_map, edges_map, index) and has no general
threads flag. The wrapper tracks one count per phase in ``phase_threads``
and passes all five on every spawn; on OOM only the failing phase's count
is halved while the others are preserved.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Optional, Sequence
from urllib.parse import urlparse

from ._progress import log_progress

if sys.platform != "win32":
    import resource

logger = logging.getLogger(__name__)

# Thread-control order. ``context`` controls concurrent source readers but is
# not a durable ingest phase: DG writes no completion marker for it and does
# not accept it in ``--resume-from-phase``.
PHASE_ORDER = ("context", "encode", "pk_map", "edges_map", "index")
# Phases that accept their own --<phase>-threads override.
PHASE_NAMES = PHASE_ORDER
# Durable phases accepted by DG's ``--resume-from-phase`` CLI. Keep this
# separate from thread controls so an absent (or stale) context marker can
# never produce the unsupported ``--resume-from-phase context``.
RESUMABLE_PHASE_ORDER = PHASE_ORDER[1:]
PHASE_TO_CLI = {
    phase: phase.replace("_", "-") for phase in RESUMABLE_PHASE_ORDER
}

_NON_RETRYABLE_PUBLIC_EXIT_CODE = 78
_OUTPUT_TAIL_MAX_BYTES = 20_000
_PUBLIC_DIAGNOSTIC_CODES = frozenset({"diskgraph_ingest_validation"})
_DIAGNOSTIC_FILENAME = ".diagnostic.json"
_DIAGNOSTIC_MAX_BYTES = 32 * 1024
_DIAGNOSTIC_VISIBILITIES = frozenset({"ordinary", "restricted"})
_DIAGNOSTIC_CATEGORIES = frozenset({
    "internal",
    "validation",
    "provider",
    "io",
    "resource",
    "process",
})
_DIAGNOSTIC_PHASES = frozenset(
    {"supervisor", "worker", "context", *PHASE_ORDER})
_DIAGNOSTIC_CHAIN_KINDS = frozenset(
    {"error", "restricted_boundary", "process"})
_STABLE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_DIAGNOSTIC_ID = re.compile(r"^dg-[0-9a-f]{24}$")
_DIAGNOSTIC_FINGERPRINT = re.compile(r"^dg-v1-[0-9a-f]{64}$")
# Output may echo row values; forward progress lines only.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_LINE = re.compile(r"^\S+\s+INFO\s+\S+: ((?:table started|"
                            r"table producers complete|Ingestion stage "
                            r"progress|\w+ stage complete)\b.*)$")

# Defaults — overridable by env or call sites.
_DEFAULT_MAX_INNER_ATTEMPTS = 5
_DEFAULT_HEARTBEAT_INTERVAL_S = 10.0
# RLIMIT_AS off by default. A static virtual-address cap is misleading: jemalloc
# retains large per-arena address ranges across phases (pk_map's peak vsize lingers
# into edges_map's mmap of pk_hash+sort_key files), which made a 450 GiB cap fail
# legitimate 24-thread resume runs even though RSS stayed well under physical RAM.
# We rely on the kernel OOM-killer for physical-RAM exhaustion (rc=137 — classified
# as oom) and on `panic = "abort"` for in-process allocator failures (rc=134).
# Set to a positive int (e.g. 480 GiB) to opt into a vsize cap when you have
# specific guardrail needs.
_DEFAULT_MEM_LIMIT_BYTES: Optional[int] = None
# Halving stops at this thread count. Below 4 the marginal benefit of
# further halving is small; if 4 threads still OOM, the failure is
# per-shard memory pressure (a topology problem) not parallelism.
_THREAD_FLOOR = 4

# prctl(2) op for setting the parent-death signal.
PR_SET_PDEATHSIG = 1

# Bind libc.prctl at module import (pre-fork). The dynamic linker can hold
# locks at fork() time that the child inherits but cannot release without
# deadlocking against the (now-vanished) parent thread that took them. Doing
# the dlopen + symbol lookup once, while we're still single-threaded and
# pre-fork, ensures the post-fork preexec_fn only makes a bound indirect
# call — no allocator, no dlopen, no Python state.
if sys.platform == "linux":
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
    _PRCTL = _LIBC.prctl
    _PRCTL.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    _PRCTL.restype = ctypes.c_int
else:
    _PRCTL = None


def _is_oom_rc(rc: int) -> bool:
    """rc values we treat as memory-pressure-class failures, eligible for
    the halve-and-resume retry loop.

    Direct ``Popen`` of the binary surfaces signal-deaths as negative
    integers (``-6``, ``-9``); the positive forms (``134``, ``137``) are
    only produced if a shell/wrapper converts ``$?`` to ``128+signum`` and
    are kept for defense-in-depth.

    The bucket is:

    * ``-9`` / ``137`` — SIGKILL. The kernel OOM-killer's signal when the
      cgroup or host runs out of physical memory and reclaim cannot keep
      up. This is the canonical OOM signal in production. Note: it is
      also produced by external ``kill -9`` and k8s liveness-probe kills,
      so we cap retries to bound exposure to that ambiguity.

    * ``-6`` / ``134`` — SIGABRT. With the kumo-diskgraph release profile
      set to ``panic = "abort"``, several memory-pressure paths land here
      that are NOT covered by ``-9``:

        - ``Vec::try_reserve`` returns ``Err`` and code unwraps it →
          panic → abort. Rust's idiomatic "graceful OOM" path.
        - ``set_alloc_error_hook``'s default handler aborts when the
          global allocator returns null (rare on Linux with default
          overcommit, but possible with ``RLIMIT_AS`` or strict
          accounting).
        - ``mmap``/``mremap`` returning ``ENOMEM`` then surfacing as a
          panic via ``unwrap`` on a fallible allocation.

      All of these are halve-and-resume-recoverable: the next attempt's
      reduced thread count lowers concurrent allocation, often enough
      to clear the bottleneck.

      The trade-off: ``-6`` is also the exit code for ordinary panics
      from logic bugs (failed invariants, OOB indices, bad ``unwrap``s).
      Treating those as OOM-class will burn the halving budget retrying
      a deterministic bug. The cost is bounded — the halving floor +
      Temporal's max attempts cap the wasted cycles — and the eventual
      surfaced error is ``DiskgraphRetryExhaustedError``, which Temporal
      treats as non-retryable so it does not recur indefinitely. We
      accept this trade-off because the production failure mode we're
      targeting (large-graph ingest at scale) more often manifests as
      ``-6`` via Rust's fallible-allocation path than as ``-9`` from the
      kernel OOM-killer.

      If logic-bug panics start showing up here in the wild, the next
      step is to add a panic-hook sentinel in the binary so we can
      distinguish "panic" from "allocation-failure abort" before
      classifying.

    SIGSEGV / ``-11`` / ``139`` is deliberately NOT in this bucket.
    Segfaults are by definition memory corruption, unsafe-code bugs, or
    hardware faults; none of them are responsive to halving threads, and
    retrying them masks the cause. They surface as fatal.
    """
    return rc in (134, 137, -6, -9)


def _is_cancel_rc(rc: int) -> bool:
    return rc in (143, -15)  # SIGTERM


def _normalize_private_output_tail(output_tail: str) -> tuple[str, Optional[str]]:
    """Bound and fingerprint an untrusted subprocess tail consistently."""
    tail_bytes = output_tail.encode("utf-8", errors="replace")
    normalized = tail_bytes[-_OUTPUT_TAIL_MAX_BYTES:].decode(
        "utf-8", errors="replace")
    while len(normalized.encode("utf-8")) > _OUTPUT_TAIL_MAX_BYTES:
        normalized = normalized[1:]
    normalized_bytes = normalized.encode("utf-8")
    fingerprint = (hashlib.sha256(normalized_bytes).hexdigest()
                   if normalized_bytes else None)
    return normalized, fingerprint


@dataclass(frozen=True)
class DiagnosticFields:
    attempt: int
    phase: str
    exit_code: Optional[int]
    signal: Optional[int]


@dataclass(frozen=True)
class DiagnosticCause:
    kind: str
    summary: str


@dataclass(frozen=True)
class DiagnosticFrame:
    module: str
    file: str
    line: int


@dataclass(frozen=True)
class PublicDiagnostic:
    code: str
    message: str
    non_retryable: bool


@dataclass(frozen=True)
class Diagnostic:
    diagnostic_id: str
    visibility: str
    operation: str
    category: str
    code: str
    retryable: bool
    summary: str
    public: Optional[PublicDiagnostic]
    fields: DiagnosticFields
    chain: tuple[DiagnosticCause, ...]
    frames: tuple[DiagnosticFrame, ...]
    fingerprint: str
    chain_truncated: bool
    frames_truncated: bool
    summary_truncated: bool


class DiskgraphIngestError(RuntimeError):
    """Base class for portable, user-readable ingest failures."""

    non_retryable = True

    def __init__(
        self,
        message: str = "",
        *,
        private_output_tail: str = "",
        diagnostic: Optional[Diagnostic] = None,
    ) -> None:
        self.message = message
        self.private_output_tail, self.output_fingerprint = (
            _normalize_private_output_tail(private_output_tail))
        self.diagnostic = diagnostic
        super().__init__(message)


class NonRetryableDiskgraphError(DiskgraphIngestError):
    """Wrapper-determined deterministic ingest failure.

    Raised when the wrapper has classified the failure as one that
    retrying cannot recover (non-OOM exit codes, OOM at the thread
    floor, OOM with all phase markers present). Each raise site
    supplies a user-facing message via the ``message`` kwarg.

    Platform adapters may use the stable ``non_retryable`` attribute when
    translating this exception to their workflow runtime.
    """
    template = "{message}"


class DiskgraphDataplaneNoSpaceError(NonRetryableDiskgraphError):
    """Ingest filled the local dataplane mirror volume.

    This must stay distinct from :class:`DiskgraphNoSpaceError`: the latter
    activates adaptive re-placement onto a larger DiskGraph node, while the
    dataplane volume is configured separately and does not grow with that node.
    """
    non_retryable = True


class DiskgraphRetryExhaustedError(DiskgraphIngestError):
    """Inner-retry budget exhausted by the OOM-recovery wrapper.

    Raised when the wrapper has halved thread counts the configured
    number of times (default 5) and the failure persists. Re-running
    the wrapper from scratch on a different worker would just repeat
    the same losing strategy.

    Repeating the same local recovery plan is not expected to succeed.
    """
    template = "{message}"


class DiskgraphCancelledError(DiskgraphIngestError):
    """The native ingest process was stopped by an external cancellation.

    The current activity must stop immediately, but Temporal may retry it on
    a fresh worker and resume from DiskGraph's durable phase markers.
    """

    non_retryable = False
    template = "{message}"


class DiskgraphNoSpaceError(DiskgraphIngestError):
    """Ingest ran out of local SSD space (ENOSPC).

    Unlike ``NonRetryableDiskgraphError`` this is left **retryable**
    (``non_retryable = False``): re-placing the snapshot with a larger disk
    reservation (a bigger node) can succeed. The stable class name is also
    the ``ApplicationError.type`` the ingest workflow matches on to trigger
    adaptive re-placement, so keep it stable.
    """
    non_retryable = False
    template = "{message}"


# Fraction of total SSD free below which a non-OOM ingest failure is treated
# as out-of-space rather than a data/schema error. ``shutil.disk_usage`` is
# racy (cleanup can free bytes after the failing process exits), so this is a
# heuristic floor, not an exact check — false negatives fall back to the
# ``NonRetryableDiskgraphError`` path.
_NO_SPACE_FREE_FRACTION = 0.02


def persisted_ingest_config_path(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    return output_dir.parent / f".{output_dir.name}.ingest_config.json"


def _valid_diagnostic_text(value: object, max_bytes: int) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return (len(encoded) <= max_bytes
            and all(ord(character) >= 0x20 and ord(character) != 0x7F
                    for character in value))


def _read_diagnostic(output_dir: Path) -> Optional[Diagnostic]:
    """Read only the allowlisted v1 diagnostic protocol from the output dir."""
    path = output_dir / _DIAGNOSTIC_FILENAME
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > _DIAGNOSTIC_MAX_BYTES):
            logger.warning(
                "ingest_runner: rejected non-regular or oversized diagnostic "
                "envelope")
            return None
        with os.fdopen(descriptor, "rb") as error_file:
            descriptor = None
            payload = error_file.read(_DIAGNOSTIC_MAX_BYTES + 1)
    except OSError:
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if len(payload) > _DIAGNOSTIC_MAX_BYTES:
        logger.warning("ingest_runner: rejected oversized diagnostic envelope")
        return None
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.warning("ingest_runner: rejected malformed diagnostic envelope")
        return None
    expected_keys = {
        "version",
        "event",
        "diagnostic_id",
        "visibility",
        "component",
        "operation",
        "category",
        "code",
        "retryable",
        "summary",
        "public",
        "fields",
        "chain",
        "frames",
        "fingerprint",
        "truncated",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        logger.warning("ingest_runner: rejected diagnostic envelope schema")
        return None
    if (type(value["version"]) is not int or value["version"] != 1
            or value["event"] != "diskgraph.ingest_failed"
            or value["component"] != "diskgraph-ingest"):
        logger.warning("ingest_runner: rejected unsupported diagnostic envelope")
        return None

    diagnostic_id = value["diagnostic_id"]
    visibility = value["visibility"]
    operation = value["operation"]
    category = value["category"]
    code = value["code"]
    retryable = value["retryable"]
    summary = value["summary"]
    fingerprint = value["fingerprint"]
    if (not isinstance(diagnostic_id, str)
            or _DIAGNOSTIC_ID.fullmatch(diagnostic_id) is None
            or not isinstance(visibility, str)
            or visibility not in _DIAGNOSTIC_VISIBILITIES
            or not isinstance(operation, str)
            or _STABLE_IDENTIFIER.fullmatch(operation) is None
            or not isinstance(category, str)
            or category not in _DIAGNOSTIC_CATEGORIES
            or not isinstance(code, str)
            or _STABLE_IDENTIFIER.fullmatch(code) is None
            or type(retryable) is not bool
            or not _valid_diagnostic_text(summary, 2 * 1024)
            or not isinstance(fingerprint, str)
            or _DIAGNOSTIC_FINGERPRINT.fullmatch(fingerprint) is None):
        logger.warning("ingest_runner: rejected invalid diagnostic fields")
        return None

    public_value = value["public"]
    public: Optional[PublicDiagnostic] = None
    if public_value is not None:
        if (not isinstance(public_value, dict)
                or set(public_value) != {
                    "code", "message", "non_retryable"
                }):
            return None
        public_code = public_value["code"]
        public_message = public_value["message"]
        if (not isinstance(public_code, str)
                or public_code not in _PUBLIC_DIAGNOSTIC_CODES
                or not _valid_diagnostic_text(public_message, 2 * 1024)
                or public_value["non_retryable"] is not True
                or visibility != "ordinary"
                or operation != "validate_input"
                or category != "validation"
                or code != public_code
                or retryable is not False
                or summary != public_message):
            return None
        public = PublicDiagnostic(
            code=public_code,
            message=public_message,
            non_retryable=True,
        )

    fields_value = value["fields"]
    if (not isinstance(fields_value, dict)
            or set(fields_value) != {
                "attempt", "phase", "exit_code", "signal"
            }):
        return None
    attempt = fields_value["attempt"]
    phase = fields_value["phase"]
    exit_code = fields_value["exit_code"]
    exit_signal = fields_value["signal"]
    if (type(attempt) is not int or not 0 <= attempt <= 1000
            or not isinstance(phase, str)
            or phase not in _DIAGNOSTIC_PHASES
            or (exit_code is not None
                and (type(exit_code) is not int
                     or not -255 <= exit_code <= 255))
            or (exit_signal is not None
                and (type(exit_signal) is not int
                     or not 1 <= exit_signal <= 255))):
        return None

    chain_value = value["chain"]
    if not isinstance(chain_value, list) or not 1 <= len(chain_value) <= 8:
        return None
    chain: list[DiagnosticCause] = []
    for cause in chain_value:
        if (not isinstance(cause, dict)
                or set(cause) != {"kind", "summary"}
                or not isinstance(cause["kind"], str)
                or cause["kind"] not in _DIAGNOSTIC_CHAIN_KINDS
                or not _valid_diagnostic_text(cause["summary"], 2 * 1024)):
            return None
        chain.append(
            DiagnosticCause(
                kind=cause["kind"],
                summary=cause["summary"],
            ))
    if (visibility == "restricted"
            and any(cause.kind == "error" for cause in chain)):
        return None

    frames_value = value["frames"]
    if not isinstance(frames_value, list) or len(frames_value) > 20:
        return None
    frames: list[DiagnosticFrame] = []
    for frame in frames_value:
        if (not isinstance(frame, dict)
                or set(frame) != {"module", "file", "line"}):
            return None
        module = frame["module"]
        file = frame["file"]
        line = frame["line"]
        if (not _valid_diagnostic_text(module, 256)
                or not _valid_diagnostic_text(file, 512)
                or "\\" in file or file.startswith("/")
                or any(part in {"", ".", ".."} for part in file.split("/"))
                or type(line) is not int or not 1 <= line <= 10_000_000):
            return None
        frames.append(
            DiagnosticFrame(module=module, file=file, line=line))

    truncated_value = value["truncated"]
    if (not isinstance(truncated_value, dict)
            or set(truncated_value) != {"chain", "frames", "summary"}
            or any(type(truncated_value[key]) is not bool
                   for key in ("chain", "frames", "summary"))):
        return None

    return Diagnostic(
        diagnostic_id=diagnostic_id,
        visibility=visibility,
        operation=operation,
        category=category,
        code=code,
        retryable=retryable,
        summary=summary,
        public=public,
        fields=DiagnosticFields(
            attempt=attempt,
            phase=phase,
            exit_code=exit_code,
            signal=exit_signal,
        ),
        chain=tuple(chain),
        frames=tuple(frames),
        fingerprint=fingerprint,
        chain_truncated=truncated_value["chain"],
        frames_truncated=truncated_value["frames"],
        summary_truncated=truncated_value["summary"],
    )


def _free_disk_fraction(path: Path) -> Optional[float]:
    """Best-effort free-space fraction for ``path``'s filesystem (``None`` if
    it can't be read).

    ENOSPC can prevent creation of the requested snapshot leaf. Walk to the
    closest existing ancestor only when ``stat`` proves a component is missing,
    so access errors cannot be mistaken for a different ancestor filesystem.
    """
    probe_path = path
    while True:
        try:
            probe_path.stat()
        except FileNotFoundError:
            if probe_path == probe_path.parent:
                return None
            probe_path = probe_path.parent
        except OSError:
            return None
        else:
            break
    try:
        usage = shutil.disk_usage(probe_path)
    except OSError:
        return None
    if usage.total <= 0:
        return None
    return usage.free / usage.total


def _local_filesystem_path(path: Optional[str]) -> Optional[Path]:
    """Return ``path`` as a local filesystem path, excluding remote URIs."""
    if not path or path.startswith('@') or urlparse(path).scheme:
        return None
    return Path(path)


# The encode phase holds one temp file open per (table, shard) — ~800+ for a
# 75-shard graph — so Docker's default 1024 soft nofile limit is a real
# failure mode (EMFILE in a writer thread, surfaced as a generic exit 1).
_LOW_NOFILE_THRESHOLD = 4096


def _soft_nofile_limit() -> Optional[int]:
    """Soft RLIMIT_NOFILE inherited by the ingest subprocess (``None`` if
    unreadable)."""
    if sys.platform == "win32":
        return None
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    return soft if soft != resource.RLIM_INFINITY else None


@dataclass
class IngestRunResult:
    stats: dict[str, Any]
    final_thread_count: int
    attempts: int
    phase_threads: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestExecutionConfig:
    """Portable execution policy for the standalone ingest process."""

    col_stats: bool = True
    cat_count_strategy: str = "hybrid"
    output_dataplane_dir: Optional[str] = None
    success_marker_content: Optional[str] = None
    initial_threads: Optional[int] = None
    mem_limit_bytes: Optional[int] = _DEFAULT_MEM_LIMIT_BYTES
    max_attempts: int = _DEFAULT_MAX_INNER_ATTEMPTS
    poll_interval_seconds: float = _DEFAULT_HEARTBEAT_INTERVAL_S
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.initial_threads is not None and self.initial_threads < 1:
            raise ValueError("initial_threads must be positive")
        if self.mem_limit_bytes is not None and self.mem_limit_bytes < 1:
            raise ValueError("mem_limit_bytes must be positive when set")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")


def _default_worker_threads_from_cpu_count() -> int:
    """Fallback mirror of Rust's default using ``os.cpu_count``.

    ``os.cpu_count`` does not honor a cgroup CPU quota, so this must remain a
    fallback for an unavailable or incompatible ingest binary rather than the
    normal source of an explicit per-phase thread count.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise RuntimeError("os.cpu_count() returned None on this platform; "
                           "pass initial_threads explicitly to run_ingest()")
    return (cpu_count * 105 + 99) // 100


def _default_worker_threads() -> int:
    """Return the ingest binary's cgroup-aware resolved thread default.

    Kumo passes an explicit thread count for every ingest phase. Querying the
    binary keeps that explicit value consistent with its cgroup-aware default
    instead of accidentally overriding it with the host-wide CPU count. Keep
    the previous calculation as a resilient fallback for old/missing binaries.
    """
    ingest_bin = _resolve_ingest_bin()
    try:
        result = subprocess.run(
            [ingest_bin, "--print-resolved-threads"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            threads = int(result.stdout.strip())
            if threads >= 1:
                return threads
        logger.warning(
            "ingest_runner: --print-resolved-threads returned rc=%d "
            "stdout=%r; falling back to os.cpu_count()",
            result.returncode,
            result.stdout,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning(
            "ingest_runner: querying %s --print-resolved-threads failed "
            "(%s); falling back to os.cpu_count()",
            ingest_bin,
            exc,
        )
    return _default_worker_threads_from_cpu_count()


def _resolve_ingest_bin() -> str:
    """Locate the ``ingest`` binary. Override via DISKGRAPH_INGEST_BIN."""
    override = os.environ.get("DISKGRAPH_INGEST_BIN")
    if override:
        return override
    binary_name = "ingest.exe" if sys.platform == "win32" else "ingest"
    # Bundled-with-wheel layout: ``<diskgraph_pkg>/bin/<binary_name>``.
    try:
        import diskgraph
        candidate = Path(diskgraph.__file__).parent / "bin" / binary_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    except Exception as e:
        logger.debug(
            "diskgraph package import or bundled-binary lookup failed; "
            "falling back to PATH: %s",
            e,
        )
    # Fall back to PATH lookup.
    return binary_name


def _has_phase_markers(output_dir: Path) -> bool:
    tmp = output_dir / "_tmp"
    if not tmp.exists():
        return False
    return any(tmp.glob("_phase_complete_*"))


def _next_phase_to_run(output_dir: Path) -> Optional[str]:
    """Lowest phase in PHASE_ORDER whose marker is missing."""
    tmp = output_dir / "_tmp"
    if not tmp.exists():
        return PHASE_ORDER[0]
    for name in PHASE_ORDER:
        if not (tmp / f"_phase_complete_{name}").exists():
            return name
    return None  # everything done


def _resume_phase_for_attempt(output_dir: Path) -> Optional[str]:
    """Return the lowest missing phase supported by DG's resume CLI."""
    tmp = output_dir / "_tmp"
    for name in RESUMABLE_PHASE_ORDER:
        if not (tmp / f"_phase_complete_{name}").is_file():
            return name
    # Index may have completed before final publication. Re-entering Index is
    # the safe DG contract for rebuilding final state without wiping its
    # durable predecessors as a fresh ingest would.
    return RESUMABLE_PHASE_ORDER[-1]


def _classify_exit(rc: int) -> str:
    """Map a subprocess exit code to a retry decision class.

    The kumo-diskgraph binary is built with `panic = "abort"` so any
    OOM-related failure (allocator failure, panic, RLIMIT_AS hit) exits
    with SIGABRT (134). Kernel OOM-killer sends SIGKILL (137). All other
    non-zero exit codes are treated as application errors that retrying
    can't fix.
    """
    if rc == 0:
        return "ok"
    if rc == _NON_RETRYABLE_PUBLIC_EXIT_CODE:
        return "public_fatal"
    if _is_cancel_rc(rc):
        return "cancelled"
    if _is_oom_rc(rc):
        return "oom"
    return "fatal"


def _set_rlimit(mem_limit_bytes: int) -> None:
    """preexec_fn: cap virtual address space of the child."""
    if sys.platform == "win32":
        raise NotImplementedError("Address-space limits require POSIX resource")
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))


def _set_pdeathsig(sig: int, expected_parent_pid: int) -> None:
    """preexec_fn helper: deliver `sig` to the child if the parent dies.

    Runs between fork and exec — must not touch the Python allocator,
    GIL-held state, or logging. Symbol binding (libc + prctl) was done at
    module import (pre-fork), so this is just a bound indirect call.
    """
    if _PRCTL is None:
        return
    _PRCTL(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    # Race window: the parent may have exited between fork() and our prctl
    # call. PR_SET_PDEATHSIG only fires for FUTURE parent deaths, so a kill
    # that happened before the prctl will not deliver our signal — we'd be
    # an orphan. Detect that by checking whether our parent is still the
    # process that called Popen; PID 1 can be a legitimate parent inside a
    # container. If not, suicide explicitly to mirror what PDEATHSIG would
    # have done. No allocator/Python state touched.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), sig)


_PROGRESS_READ_CAP = 1 << 20  # Bound bytes read/held per poll.


def _forward_progress(
    output: BinaryIO,
    offset: int,
    skipping: bool = False,
) -> tuple[int, bool]:
    """Forward new progress lines since `offset`.

    Reads at most `_PROGRESS_READ_CAP` bytes per call from a separate reader
    (child's file offset is untouched), so one unbounded line can't make a poll read
    or re-read ever-growing data. A line that exceeds the cap before a
    newline is dropped; `skipping` stays True until the next newline is
    seen. Progress lines are short, so only non-progress output is at risk.
    """
    fd = output.fileno()
    remaining = os.fstat(fd).st_size - offset
    if remaining <= 0:
        return offset, skipping
    to_read = min(remaining, _PROGRESS_READ_CAP)
    if sys.platform == "win32":
        output.seek(offset)
        data = output.read(to_read)
    else:
        data = os.pread(fd, to_read, offset)
    end = data.rfind(b"\n") + 1
    if end == 0:
        if to_read == _PROGRESS_READ_CAP:
            return offset + to_read, True  # No newline in a full chunk.
        return offset, skipping  # Partial line, wait for more data.

    lines = data[:end]
    if skipping:
        lines = lines[lines.find(b"\n") + 1:]  # Drop the overlong line.
    for line in lines.decode("utf-8", errors="replace").splitlines():
        match = _PROGRESS_LINE.match(_ANSI_ESCAPE.sub("", line))
        if match:
            log_progress(logger, "ingest: %s", match.group(1))
    return offset + end, False


def _run_with_heartbeat(
    cmd: list[str],
    *,
    mem_limit_bytes: Optional[int],
    heartbeat_interval_s: float,
    heartbeat: Optional[Callable[[], None]] = None,
    subprocess_environment: Optional[Mapping[str, Optional[str]]] = None,
) -> tuple[int, str]:
    """Spawn the binary and return its code plus a bounded output tail."""
    if sys.platform == "win32" and mem_limit_bytes is not None:
        raise NotImplementedError("Address-space limits require POSIX resource")
    preexec = None
    if sys.platform == "linux":
        expected_parent_pid = os.getpid()

        def _pre():
            _set_pdeathsig(signal.SIGKILL, expected_parent_pid)
            if mem_limit_bytes is not None:
                _set_rlimit(mem_limit_bytes)

        preexec = _pre

    logger.info("ingest_runner: spawning %s", cmd)
    child_env = os.environ.copy()
    child_env["DISKGRAPH_PYTHON"] = sys.executable
    if subprocess_environment:
        for name, value in subprocess_environment.items():
            if value is None:
                child_env.pop(name, None)
            else:
                child_env[name] = value

    # Native subprocess descriptors require local files. Separate opens keep
    # Windows progress reads from moving the child's write offset (no pread).
    with (tempfile.TemporaryDirectory(prefix="diskgraph-output-") as output_dir,
          open(os.path.join(output_dir, "output.log"), "w+b") as output,
          open(os.path.join(output_dir, "output.log"), "rb", buffering=0)
          as progress):
        proc = subprocess.Popen(
            cmd,
            stdout=output,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec,
            env=child_env,
        )
        forwarded = 0
        skipping = False
        healthy = True

        def forward(drain: bool) -> None:
            # Progress is best-effort: never let it fail the ingest.
            nonlocal forwarded, skipping, healthy
            while healthy:
                try:
                    end, skipping = _forward_progress(progress, forwarded,
                                                      skipping)
                except OSError as e:
                    healthy = False
                    logger.debug("ingest progress forwarding stopped: %s", e)
                    return
                advanced = end != forwarded
                forwarded = end
                if not (drain and advanced):
                    return

        try:
            while True:
                try:
                    rc = proc.wait(timeout=heartbeat_interval_s)
                    break
                except subprocess.TimeoutExpired:
                    forward(drain=False)
                    if heartbeat is not None:
                        try:
                            heartbeat()
                        except Exception as e:
                            logger.debug(
                                "ingest progress callback raised: %s",
                                e,
                            )
        finally:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("ingest binary did not exit after SIGKILL")
        output.flush()
        forward(drain=True)
        output.seek(0, os.SEEK_END)
        output_size = output.tell()
        output.seek(max(0, output_size - _OUTPUT_TAIL_MAX_BYTES))
        return rc, output.read().decode("utf-8", errors="replace")


def _remove_files_for_completed_phases(
    output_dir: Path,
    cleanup_paths_after_phase: Mapping[str, Sequence[str]],
) -> None:
    for phase, paths in cleanup_paths_after_phase.items():
        marker = output_dir / "_tmp" / f"_phase_complete_{phase}"
        if marker.is_file():
            for path in paths:
                Path(path).unlink(missing_ok=True)


def _build_ingest_cmd(
    ingest_bin: str,
    config_path: Path,
    output_dir: Path,
    phase_threads: dict[str, int],
    *,
    col_stats: bool,
    cat_count_strategy: str,
    resume_from_phase: Optional[str],
    output_dataplane_dir: Optional[str],
    success_marker_content: Optional[str],
    extra_args: Optional[list[str]],
) -> list[str]:
    """Assemble the argv for one spawn of the ingest binary.

    All five ``--<phase>-threads`` flags are passed on every spawn so the
    binary always sees the wrapper's per-phase counts (even when the
    failing phase is later in the pipeline than ``--resume-from-phase``).

    ``output_dataplane_dir`` / ``success_marker_content`` mirror the
    pyo3 ``ingest()`` kwargs; they are forwarded as CLI flags to the
    binary so the dataplane snapshot mirror writes ``_MANIFEST.json``
    and ``_SUCCESS`` (containing the marker payload) on both backends.
    Required: the binary must be from a kumo-diskgraph build that
    exposes those flags (added in PR adding
    ``vedaant/ingest-cli-dataplane-flags``); older binaries reject
    them at clap parse time.
    """
    cmd: list[str] = [
        ingest_bin,
        "--source",
        str(config_path),
        "--output",
        str(output_dir),
    ]
    for p in PHASE_NAMES:
        cmd += [f"--{p.replace('_', '-')}-threads", str(phase_threads[p])]
    if col_stats:
        cmd += ["--col-stats"]
    cmd += ["--cat-count-strategy", cat_count_strategy]
    if resume_from_phase is not None:
        cmd += ["--resume-from-phase", PHASE_TO_CLI[resume_from_phase]]
    if output_dataplane_dir is not None:
        cmd += ["--output-dataplane-dir", output_dataplane_dir]
    if success_marker_content is not None:
        cmd += ["--success-marker-content", success_marker_content]
    if extra_args:
        cmd += extra_args
    return cmd


def _halve_failing_phase(
    phase_threads: dict[str, int],
    output_dir: Path,
    floor: int,
) -> tuple[bool, str]:
    """Reduce the failing phase's thread count for the next attempt.

    Returns ``(True, "")`` when halving succeeded and the caller should
    spawn another attempt with the updated ``phase_threads``. Returns
    ``(False, reason)`` when no further halving is possible — either the
    phase scan can't identify a failing phase, or the failing phase is
    already at the floor. The caller is expected to surface ``reason`` as
    a ``NonRetryableDiskgraphError``.
    """
    failed = _next_phase_to_run(output_dir)
    if failed is None:
        # All markers present at OOM time — finalize-step failure or a race
        # between marker write and crash. Either way, halving doesn't apply.
        return False, (
            "Graph ingest reported memory pressure after all pipeline "
            "phases had completed. This is unusual — please contact Kumo "
            "with the run ID so we can investigate.")
    current = phase_threads[failed]
    new_count = max(floor, current // 2)
    if new_count == current:
        return False, (
            f"Graph ingest ran out of memory in phase '{failed}' even at "
            f"the lowest thread count we automatically retry with ({floor} "
            "threads). Your data is likely too large for this cluster — try "
            "splitting the largest tables, or contact Kumo about running on "
            "larger hardware.")
    logger.warning(
        "ingest_runner: phase %s OOM at %d threads → halving to %d",
        failed,
        current,
        new_count,
    )
    phase_threads[failed] = new_count
    return True, ""


def run_ingest(
    ingest_config: Optional[dict[str, Any]],
    output_dir: Path,
    *,
    col_stats: bool = True,
    cat_count_strategy: str = "hybrid",
    output_dataplane_dir: Optional[str] = None,
    success_marker_content: Optional[str] = None,
    initial_threads: Optional[int] = None,
    mem_limit_bytes: Optional[int] = _DEFAULT_MEM_LIMIT_BYTES,
    max_inner_attempts: int = _DEFAULT_MAX_INNER_ATTEMPTS,
    heartbeat_interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
    extra_args: Optional[list[str]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
    subprocess_environment: Optional[Mapping[str, Optional[str]]] = None,
    cleanup_paths_after_phase: Optional[Mapping[str, Sequence[str]]] = None,
) -> IngestRunResult:
    """Run the diskgraph ingest binary with retry + resume + per-phase halving.

    The first attempt runs all phases from scratch with `initial_threads`
    workers. Each phase is given its own `--<phase>-threads` override, all
    initially equal to `initial_threads`.

    On OOM-class failures (SIGKILL, SIGABRT-from-rlimit, structured
    `exit_class=oom`), the wrapper:

      1. Identifies the failing phase via the phase-marker scan (lowest
         missing marker). Halves only that phase's thread count.
      2. Picks the resume phase from the marker scan (the same one).
      3. Re-invokes the binary with `--resume-from-phase` plus all five
         `--<phase>-threads` flags reflecting the current per-phase counts.

    Non-OOM failures (data, schema, config) are non-retryable and surface
    as :class:`NonRetryableDiskgraphError`. Transient I/O failures retry
    without changing the thread counts.

    Returns:
        IngestRunResult with parsed stats, final per-phase thread counts,
        attempt count.

    Raises:
        NonRetryableDiskgraphError: schema/data/config error.
        DiskgraphDataplaneNoSpaceError: local dataplane mirror is full.
        DiskgraphRetryExhaustedError: ran out of retry attempts.
    """
    if initial_threads is None:
        initial_threads = _default_worker_threads()
    phase_threads: dict[str, int] = {p: initial_threads for p in PHASE_NAMES}
    effective_floor = min(_THREAD_FLOOR, initial_threads)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # ingest_stats.json is written by ingest_with_stats_from_source after
    # the index phase; it contains the same shape as the pybind's stats_json.
    stats_path = output_dir / "ingest_stats.json"
    ingest_bin = _resolve_ingest_bin()

    # Write config to a sibling file (NOT inside output_dir, since the
    # binary wipes output_dir on fresh attempts after reading the config).
    # Place next to output_dir so retries use the same path.
    output_dir = output_dir.resolve()
    config_path = persisted_ingest_config_path(output_dir)
    if ingest_config is not None:
        config_path.write_text(json.dumps(ingest_config))
    elif not config_path.is_file():
        raise NonRetryableDiskgraphError(
            "Graph ingest cannot resume because its persisted source "
            "configuration is missing.")

    resolved_extra_args = list(extra_args or [])
    cleanup_paths = dict(cleanup_paths_after_phase or {})
    unknown_cleanup_phases = set(cleanup_paths).difference(PHASE_ORDER)
    if unknown_cleanup_phases:
        raise ValueError(
            "cleanup_paths_after_phase contains unknown DiskGraph phases: "
            f"{sorted(unknown_cleanup_phases)!r}")

    progress_callback = heartbeat
    if cleanup_paths:

        def report_progress() -> None:
            _remove_files_for_completed_phases(output_dir, cleanup_paths)
            if heartbeat is not None:
                heartbeat()

        progress_callback = report_progress

    last_diagnostic: Optional[Diagnostic] = None
    for attempt in range(max_inner_attempts):
        # Resume on retries or whenever phase markers from a prior crash
        # are already on disk. The binary validates predecessor markers
        # itself and errors out cleanly if state is inconsistent.
        resume_from = (_resume_phase_for_attempt(output_dir) if attempt > 0
                       or _has_phase_markers(output_dir) else None)
        cmd = _build_ingest_cmd(
            ingest_bin,
            config_path,
            output_dir,
            phase_threads,
            col_stats=col_stats,
            cat_count_strategy=cat_count_strategy,
            resume_from_phase=resume_from,
            output_dataplane_dir=output_dataplane_dir,
            success_marker_content=success_marker_content,
            extra_args=resolved_extra_args,
        )
        logger.info(
            "ingest_runner: attempt %d/%d resume_from=%s phase_threads=%s mem_limit_gb=%s",
            attempt + 1,
            max_inner_attempts,
            resume_from,
            phase_threads,
            mem_limit_bytes // 1024**3 if mem_limit_bytes else "unlimited",
        )

        # Remove only the exact protocol filename before every attempt so an
        # abruptly terminated binary cannot make us consume a prior
        # invocation's diagnostic.
        diagnostic_current = True
        try:
            (output_dir / _DIAGNOSTIC_FILENAME).unlink(missing_ok=True)
        except OSError:
            diagnostic_current = False
            logger.warning(
                "ingest_runner: could not clear stale diagnostic; detail "
                "will be ignored for this attempt")

        run_kwargs = {
            "mem_limit_bytes": mem_limit_bytes,
            "heartbeat_interval_s": heartbeat_interval_s,
        }
        if progress_callback is not None:
            run_kwargs["heartbeat"] = progress_callback
        if subprocess_environment is not None:
            run_kwargs["subprocess_environment"] = subprocess_environment
        rc, output_tail = _run_with_heartbeat(cmd, **run_kwargs)
        if cleanup_paths:
            _remove_files_for_completed_phases(output_dir, cleanup_paths)
        cls = _classify_exit(rc)

        if cls == "ok":
            if not stats_path.exists():
                raise DiskgraphRetryExhaustedError(
                    "Graph ingest reported success but the expected stats "
                    f"output ({stats_path.name}) was not produced. This is "
                    "an unexpected internal state — please contact Kumo "
                    "with the run ID.")
            return IngestRunResult(
                stats=json.loads(stats_path.read_text()),
                final_thread_count=max(phase_threads.values()),
                attempts=attempt + 1,
                phase_threads=dict(phase_threads),
            )

        logger.warning(
            "ingest_runner: attempt %d failed rc=%d class=%s phase_threads=%s",
            attempt,
            rc,
            cls,
            phase_threads,
        )
        output_tail, output_fingerprint = _normalize_private_output_tail(
            output_tail)
        diagnostic = (_read_diagnostic(output_dir)
                      if diagnostic_current else None)
        last_diagnostic = diagnostic
        if output_fingerprint is not None:
            logger.warning(
                "ingest_runner: private subprocess diagnostics bytes=%d sha256=%s",
                len(output_tail.encode("utf-8")),
                output_fingerprint,
            )

        if cls == "cancelled":
            raise DiskgraphCancelledError(
                "Graph ingest was cancelled before it could complete.",
                diagnostic=diagnostic,
            )

        if cls == "public_fatal":
            if diagnostic is not None and diagnostic.public is not None:
                message = diagnostic.public.message
            else:
                message = (
                    "Graph ingest rejected its input, but no trusted public "
                    "error detail was available. Contact Kumo with the run "
                    "ID so the failure can be investigated.")
            raise NonRetryableDiskgraphError(
                message,
                private_output_tail=output_tail,
                diagnostic=diagnostic,
            )

        if cls == "fatal":
            # The binary writes the dataplane mirror inline. When that target
            # is a separate local filesystem, probing only output_dir (pool
            # NVMe) deterministically mislabels dataplane ENOSPC as a schema
            # error. Check the dataplane first: if both volumes are full, this
            # terminal ceiling must not trigger futile larger-node placement.
            dataplane_path = _local_filesystem_path(output_dataplane_dir)
            if dataplane_path is not None:
                dataplane_free_frac = _free_disk_fraction(dataplane_path)
                if (dataplane_free_frac is not None
                        and dataplane_free_frac < _NO_SPACE_FREE_FRACTION):
                    raise DiskgraphDataplaneNoSpaceError(
                        "Graph ingest ran out of space while writing its "
                        "inline dataplane mirror at "
                        f"{output_dataplane_dir} (only "
                        f"{dataplane_free_frac * 100:.1f}% free, exit code "
                        f"{rc}). This data_plane_storage_root is separate "
                        "from the DiskGraph worker's local SSD, so retrying "
                        "on a larger DiskGraph node will not fix it. Free or "
                        "expand the dataplane volume, or configure "
                        "data_plane_storage_root to use object storage, then "
                        "retry the ingest.",
                        diagnostic=diagnostic,
                    )

            if diagnostic is not None and diagnostic.code == "ingest.no_space":
                raise DiskgraphNoSpaceError(
                    "Graph ingest ran out of local SSD space while writing "
                    f"{output_dir} (exit code {rc}). Free local capacity or "
                    "retry with a larger disk reservation.",
                    diagnostic=diagnostic,
                )

            # A disk-full (ENOSPC) exit is non-OOM, so it lands here too and
            # would otherwise be mislabeled as a schema/data error (and marked
            # non-retryable). Disambiguate on the way out: if the SSD is
            # (nearly) full, surface a distinct RETRYABLE no-space error so the
            # ingest workflow can re-place on a larger node.
            free_frac = _free_disk_fraction(output_dir)
            if (free_frac is not None and free_frac < _NO_SPACE_FREE_FRACTION):
                raise DiskgraphNoSpaceError(
                    "Graph ingest ran out of local SSD space (only "
                    f"{free_frac * 100:.1f}% free at {output_dir}, exit code "
                    f"{rc}). Retrying with a larger disk reservation may "
                    "succeed.",
                    diagnostic=diagnostic,
                )
            # Non-OOM failure: data/schema/config error or programming bug.
            # panic=abort means every OOM path exits via SIGABRT/SIGKILL
            # (rc=134/137 or -6/-9), so anything else is application-level
            # and retrying with halved threads cannot fix it.
            fd_limit = _soft_nofile_limit()
            fd_note = ""
            if fd_limit is not None and fd_limit < _LOW_NOFILE_THRESHOLD:
                fd_note = (
                    f" Note: the process open-file limit is only {fd_limit}; "
                    "graph ingest can exceed this on larger graphs — raising "
                    "the nofile ulimit (e.g. to 65535) and retrying may "
                    "resolve this.")
            raise NonRetryableDiskgraphError(
                "Graph ingest failed for a non-recoverable reason "
                f"(exit code {rc}). Common causes: a schema mismatch, an "
                "invalid column type, a data-validation issue, or an OS "
                "resource limit such as the open-file cap. Contact Kumo with "
                f"the run ID for the specific cause.{fd_note}",
                private_output_tail=output_tail,
                diagnostic=diagnostic,
            )

        # cls == "oom" — halve the failing phase or surface as non-retryable
        # if we've exhausted the per-phase halving budget (floor reached).
        ok, reason = _halve_failing_phase(phase_threads, output_dir,
                                          effective_floor)
        if not ok:
            raise NonRetryableDiskgraphError(
                reason,
                diagnostic=diagnostic,
            )

    raise DiskgraphRetryExhaustedError(
        "Graph ingest exhausted automatic memory-recovery retries — we "
        f"tried lowering thread counts {max_inner_attempts} times to recover "
        "from memory pressure but the failure persisted. Final per-phase "
        f"thread counts: {phase_threads}. Likely cause: the dataset size "
        "exceeds what this cluster can ingest; contact Kumo about running "
        "on larger hardware.",
        diagnostic=last_diagnostic,
    )


def run_ingest_configured(
    ingest_config: dict[str, Any],
    output_dir: Path,
    config: IngestExecutionConfig,
    *,
    heartbeat: Optional[Callable[[], None]] = None,
) -> IngestRunResult:
    """Execute ingest using a reusable, serializable policy object."""

    return run_ingest(
        ingest_config,
        output_dir,
        col_stats=config.col_stats,
        cat_count_strategy=config.cat_count_strategy,
        output_dataplane_dir=config.output_dataplane_dir,
        success_marker_content=config.success_marker_content,
        initial_threads=config.initial_threads,
        mem_limit_bytes=config.mem_limit_bytes,
        max_inner_attempts=config.max_attempts,
        heartbeat_interval_s=config.poll_interval_seconds,
        extra_args=list(config.extra_args),
        heartbeat=heartbeat,
    )
