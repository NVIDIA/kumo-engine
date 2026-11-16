"""Scoped Kumo observations, following Arena's operation ownership rules.

Only the creating thread is observed; global wrappers are restored on exit.
The profiler design reference is Arena 220aa187. This smaller adapter has no
Arena imports and never guesses native work from residual host elapsed time.
"""

import inspect
import threading
from collections.abc import Mapping
from functools import wraps
from pathlib import Path

_ACTIVE = threading.Lock()
_ABSENT = object()


class KumoObservers:
    def __init__(self, recorder):
        self.recorder = recorder
        self.thread = threading.get_ident()
        self.patches = []
        self.sources = set()
        self.locked = False

    def patch(self, owner, name, decorate):
        original = getattr(owner, name)
        saved = vars(owner).get(name, _ABSENT)
        replacement = decorate(original)
        setattr(owner, name, replacement)
        self.patches.append((owner, name, saved))

    def timed(self, operation):
        def decorate(original):
            @wraps(original)
            def call(*args, **kwargs):
                if threading.get_ident() != self.thread:
                    return original(*args, **kwargs)
                with self.recorder.span(operation) as details:
                    details["implementation"] = (
                        f"{original.__module__}.{original.__qualname__}"
                    )
                    return original(*args, **kwargs)

            return call

        return decorate

    def __enter__(self):
        if not _ACTIVE.acquire(blocking=False):
            raise RuntimeError(
                "Kumo benchmark observers already active in this process"
            )
        self.locked = True
        try:
            from kumo.pytorch import assembly, encoding, preparation
            from kumo.pytorch.engines import diskgraph

            for name, operation in {
                "_fit_inputs": "input.fit",
                "_fit_targets": "target.fit",
                "_rows_for_inputs": "python.merge",
                "_encoded_column": "python.encode",
                "_token_rows": "python.tokenize",
                "_mask_tokens": "python.masks",
                "_target_outputs": "python.labels",
                "_column_output": "python.assemble",
                "_token_output": "python.assemble",
            }.items():
                self.patch(assembly, name, self.timed(operation))
            self.patch(
                assembly.PreparedBatchPlan,
                "_compose_batches",
                self.timed("python.plan"),
            )
            self.patch(
                assembly.PreparedBatchPlan, "_collate", self.timed("python.postprocess")
            )
            self.patch(assembly, "fit_encoder", self._encoder_fit)
            self.patch(preparation, "fit_from_stats", self._encoder_fit)
            self.patch(
                encoding.FittedEncoder, "encode_arrow", self.timed("python.encode")
            )
            self.patch(diskgraph, "ensure_diskgraph_index", self._index)
            return self
        except BaseException:
            self.close()
            raise

    def _index(self, original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            if threading.get_ident() != self.thread:
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            path = Path(bound.arguments["destination"])
            operation = "index.reopen" if path.exists() else "index.build"
            with self.recorder.span(operation) as details:
                details["timing_scope"] = (
                    "ensure index, including binding checks; nested inside prepare"
                )
                result = original(*args, **kwargs)
            size, inodes = 0, set()
            for item in path.rglob("*"):
                if item.is_symlink() or not item.is_file():
                    continue
                stat = item.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity not in inodes:
                    inodes.add(identity)
                    size += stat.st_size
            self.recorder.measure(
                "index.bytes",
                size,
                method="unique_regular_file_logical_bytes",
                index_action=operation,
                path=str(path),
            )
            return result

        return call

    def _encoder_fit(self, original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            if threading.get_ident() != self.thread:
                return original(*args, **kwargs)
            from kumo.pytorch.api import Buckets, Text, Vocabulary, ZScore

            arguments = signature.bind(*args, **kwargs).arguments
            config = arguments.get("config")
            if config is None:
                config = arguments["encoder"].config
            operation = (
                "vocabulary.fit"
                if isinstance(config, Vocabulary)
                else "statistics.fit"
                if isinstance(config, (ZScore, Buckets))
                else "bpe.fit"
                if isinstance(config, Text) and config.state is None
                else None
            )
            if operation is None:
                return original(*args, **kwargs)
            with self.recorder.span(operation) as details:
                details.update(
                    encoder_identity=arguments["identity"],
                    implementation=original.__qualname__,
                    configuration=repr(config),
                )
                return original(*args, **kwargs)

        return call

    def attach_source(self, source):
        if id(source) in self.sources:
            return
        self.sources.add(id(source))
        if callable(getattr(source, "_graph_output", None)):
            self.patch(source, "_graph_output", self.timed("python.assemble"))
        for method, profile in {
            "_sample_graph_events": "_last_sampling_profile",
            "_feature_arrays": "_last_retrieval_profile",
            "build": "last_temporal_profile",
            "build_model_ordered": "last_temporal_profile",
            "model_lengths": "last_length_profile",
            "sample_graph": "last_graph_profile",
        }.items():
            if not callable(getattr(source, method, None)) or not hasattr(
                source, profile
            ):
                continue
            self.patch(source, method, self._source_call(source, profile))
        if callable(getattr(source, "sample_graph_batch", None)) and hasattr(
            source, "last_graph_profile"
        ):
            self.patch(source, "sample_graph_batch", self._runtime_graph)

    def _runtime_graph(self, original):
        signature = inspect.signature(original)

        @wraps(original)
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if threading.get_ident() == self.thread:
                runtime = signature.bind(*args, **kwargs).arguments.get("runtime", {})
                if runtime.get("native_profile") is not None:
                    self.capture_native(runtime["native_profile"], original.__name__)
                else:
                    self.capture(
                        {"instrumentation_gap": "Native graph profile absent"},
                        original.__name__,
                        "last_graph_profile",
                    )
            return result

        return call

    def _source_call(self, source, profile_name):
        def decorate(original):
            @wraps(original)
            def call(*args, **kwargs):
                if threading.get_ident() != self.thread:
                    return original(*args, **kwargs)
                before = getattr(source, profile_name)
                result = original(*args, **kwargs)
                after = getattr(source, profile_name)
                if after is not None and after is not before:
                    if profile_name.startswith("_last_"):
                        self.capture_native(after, original.__name__)
                    else:
                        if profile_name == "last_graph_profile":
                            after = dict(
                                after,
                                native_profile_available=getattr(
                                    source, "_last_sampling_profile", None
                                )
                                is not None,
                            )
                        self.capture(after, original.__name__, profile_name)
                return result

            return call

        return decorate

    def capture_native(self, profile, method):
        record = self.recorder.write(
            "backend_profile",
            method=method,
            profile_name="native_graph",
            profile=dict(profile),
        )
        for field, metric in {
            "sampling_ns": "native.sample.elapsed_ns",
            "retrieval_ns": "native.retrieve.elapsed_ns",
            "rows_examined": "rows.examined",
            "rows_decoded": "rows.decoded",
        }.items():
            if field in profile:
                self.recorder.measure(
                    metric,
                    profile[field],
                    backend_profile_id=record["id"],
                    timing_scope="native elapsed including output backpressure; fused sampling/retrieval share one interval"
                    if profile.get("fused")
                    else "native elapsed including output backpressure",
                    row_scope="adjacency source-row visits plus feature decode occurrences; index-pruned edges excluded",
                )

    def capture(self, profile, method, profile_name):
        if not isinstance(profile, Mapping):
            raise TypeError("Backend timing profile must be a mapping")
        # Freeze nested scalar metadata, never retain model inputs or native buffers.
        copied = dict(profile)
        timings = dict(copied.get("native_timings_ns", {}))
        stats = dict(copied.get("native_stats", {}))
        copied.update(native_timings_ns=timings, native_stats=stats)
        record = self.recorder.write(
            "backend_profile", method=method, profile_name=profile_name, profile=copied
        )
        for field, operation in {
            "traversal": "native.sample",
            "decode_and_select": "native.retrieve",
            "aggregation": "native.aggregate",
        }.items():
            if field in timings:
                self.recorder.measure(
                    operation + ".elapsed_ns",
                    timings[field],
                    backend_field=field,
                    backend_profile_id=record["id"],
                    timing_scope="native temporal query; overlapping parent timings retained separately",
                )
        if "rows_decoded" in stats:
            self.recorder.measure(
                "rows.decoded",
                stats["rows_decoded"],
                backend_profile_id=record["id"],
                scope="native temporal query row decodes",
            )
        if not stats and copied.get("native_call_ns") == 0:
            self.recorder.measure(
                "rows.examined",
                0,
                backend_profile_id=record["id"],
                scope="no temporal query issued; graph/retrieval calls counted separately",
            )
        elif "rows_considered" in stats:
            self.recorder.measure(
                "rows.examined",
                stats["rows_considered"],
                backend_profile_id=record["id"],
                scope="candidate frame row occurrences before filtering/selection; index-pruned edges excluded",
            )
        elif not copied.get("native_profile_available") and profile_name in {
            "last_temporal_profile",
            "last_length_profile",
            "last_graph_profile",
        }:
            self.recorder.measure(
                "rows.examined",
                status="unavailable",
                reason="DiskGraph does not expose a rows-examined counter for this operation; decoded rows and visited edges are different counts",
                backend_profile_id=record["id"],
            )
        if profile_name == "last_graph_profile" and not copied.get(
            "native_profile_available"
        ):
            for operation in ("native.sample", "native.retrieve"):
                self.recorder.measure(
                    operation + ".elapsed_ns",
                    status="unavailable",
                    reason="Graph profile contains host sampling/fetch boundaries, not Rust-only elapsed times",
                    backend_profile_id=record["id"],
                )

    def close(self):
        try:
            while self.patches:
                owner, name, saved = self.patches.pop()
                if saved is _ABSENT:
                    delattr(owner, name)
                else:
                    setattr(owner, name, saved)
        finally:
            if self.locked:
                self.locked = False
                _ACTIVE.release()

    def __exit__(self, *exc):
        self.close()
