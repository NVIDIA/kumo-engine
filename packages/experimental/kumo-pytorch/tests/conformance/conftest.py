# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A complete run must actually execute every golden, property and corpus obligation."""

import pytest

REQUIRED = pytest.StashKey[set[str]]()
PROPERTY_TESTS = "tests/conformance/test_properties.py"
DTYPE_TESTS = "tests/conformance/test_dtype_matrix.py"
PINNED_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "decimal128",
    "decimal256",
    "bool",
    "string",
    "bytes",
    "date32",
    "timestamp",
)
PINNED_ENCODERS = ("raw", "vocabulary", "zscore", "buckets", "text")
# Independent of the test parametrization, so removing a boundary still fails
# --conformance-complete. These extend existing tests without an Arena runner.
PINNED_BATCH_BOUNDARIES = {
    "test_sample_count_boundaries": tuple(
        f"{drop}-n{n}" for drop in ("keep", "drop") for n in (0, 1, 15, 16, 17, 32, 33)
    ),
    "test_oversized_position_and_exact_budget": tuple(
        f"{layout}-{lengths}"
        for layout in ("padded", "packed")
        for lengths in ("7", "1,7", "7,1", "1,2,7", "6")
    ),
    "test_bucketing_has_exact_padding_cost": ("unbucketed", "bucketed"),
    "test_bucketing_uses_truncated_lengths": ("",),
    "test_bucketing_respects_chunk_scope": ("chunks1", "chunks2"),
}
#: Pinned here, apart from the parametrization: dropping a property or a seed
#: batch from conformance.properties leaves a required ID unpassed.
PINNED_PROPERTIES = {
    "diskgraph": (
        "delivery",
        "aggregate_answers",
        "epoch_trend",
        "irrelevance",
        "partition",
        "permutation",
        "point_in_time",
        "refusal_timing",
        "time_shift",
        "truncated_parquet",
    ),
}
# -- Functional areas: targets, anchors, packing, the model path, fitted state --
PINNED_PROPERTIES["diskgraph"] += (
    "anchor_splits",
    "contract_transfer",
    "fitted_state",
    "label_window",
    "model_delivery",
    "model_epoch_trend",
    "packing",
    "root_availability",
)
_SAMPLING = (
    "graph_determinism",
    "graph_edge_uniqueness",
    "graph_point_in_time",
    "graph_structure",
    "graph_tie_invariance",
    "sequence_permutation",
    "sequence_point_in_time",
)
PINNED_PROPERTIES["diskgraph"] += ("index_layout",)
PINNED_PROPERTIES["diskgraph"] += (
    *_SAMPLING,
    "graph_expected",
    "graph_inference_exact",
    "sequence_expected",
)
PINNED_SEEDS = {"diskgraph": (0,)}
# The nightly's behavior-only scope is explicit; the other workflows retain the
# machinery tests and comparisons. Keep its inventory independent of parametrization.
NATIVE_MODULES = frozenset(
    {
        "test_guarantees.py",
        "test_properties.py",
        "test_dtype_matrix.py",
        "test_field_semantics.py",
        "test_batch_boundaries.py",
    }
)
PINNED_NATIVE_PROPERTIES = PINNED_PROPERTIES["diskgraph"]
PINNED_FIELD_SEMANTICS = (
    "test_calendar_timing_outputs_have_only_explicit_values",
    "test_aligned_columns_follow_parent_row_removal",
    "test_complete_reuse_never_fits_or_grows",
    "test_a_stored_column_fits_on_its_table_rows_not_the_fitting_anchors",
    "test_exposure_warning_names_overlap_and_masks_only_named_input",
    "test_physical_dtype_mismatch_fails_during_binding",
    "test_invalid_output_mode_fails_at_batch_setup",
)
PASSED = pytest.StashKey[set[str]]()


def pytest_addoption(parser):
    add_conformance_options(parser)


def add_conformance_options(parser):
    """Register the conformance options once.

    tests/conftest.py also calls this: pytest before 9.1 only applies options from the
    conftests it loads at startup, and this one isn't among them when pytest runs ``tests``.
    """
    try:
        parser.addoption(
            "--conformance-engine",
            choices=("diskgraph", "all"),
            default="diskgraph",
            help="engine whose goldens and properties must execute (default: diskgraph)",
        )
        parser.addoption(
            "--conformance-no-in-memory",
            action="store_true",
            help="require DiskGraph behavior only; forbid InMemory execution",
        )
        parser.addoption("--conformance-complete", action="store_true", help="require every case")
        parser.addoption(
            "--conformance-seeds",
            type=int,
            help="also run N seed batches from START per engine; pinned ones stay required",
        )
        parser.addoption(
            "--conformance-seed-start",
            type=int,
            default=0,
            help="first extra seed batch for --conformance-seeds (default 0)",
        )
    except ValueError:  # already registered by tests/conftest.py
        return


def pytest_configure(config):
    if (
        config.getoption("--conformance-no-in-memory")
        and config.getoption("--conformance-engine") != "diskgraph"
    ):
        raise pytest.UsageError(
            "--conformance-no-in-memory requires --conformance-engine=diskgraph"
        )
    config.addinivalue_line(
        "markers", "engine(name): a harness test of one engine, run when that engine is selected"
    )


def pytest_ignore_collect(collection_path, config):
    if config.getoption("--conformance-no-in-memory"):
        if (
            collection_path.name.startswith("test_")
            and collection_path.suffix == ".py"
            and collection_path.name not in NATIVE_MODULES
        ):
            return True
    return None


@pytest.fixture(autouse=True)
def _forbid_in_memory():
    from tests.native_fixtures import forbid_deprecated_engine

    with forbid_deprecated_engine(
        message="InMemoryEngine is forbidden in DiskGraph-only conformance"
    ):
        yield


def pytest_collection_modifyitems(config, items):
    """An unselected engine's harness tests are never created, like its obligations.

    A selected engine that can't be imported fails them (require_engine), never skips.
    """
    selected = config.getoption("--conformance-engine")
    engines = ("diskgraph",) if selected == "all" else (selected,)
    items[:] = [
        item
        for item in items
        if all(marker.args[0] in engines for marker in item.iter_markers("engine"))
    ]


def pytest_sessionstart(session):
    if not session.config.getoption("--conformance-complete"):
        return
    from conformance.cases import ENGINES, discover_cases
    from conformance.expectations import load_expectations
    from conformance.properties import COMPARISON_PROPERTIES, obligation_id, obligations
    from conformance.registry import REPLAYED

    cases, expectations = discover_cases(), load_expectations()
    if cases.keys() != expectations.keys():
        raise pytest.UsageError("every golden case must have independent engine expectations")
    native_only = session.config.getoption("--conformance-no-in-memory")
    selected = session.config.getoption("--conformance-engine")
    engines = ENGINES if selected == "all" else (selected,)
    session.config.stash[REQUIRED] = {
        "tests/conformance/test_guarantees.py::test_public_behavior["
        f"{case_id}-{engine}-{'answer' if decisions[engine] == 'answer' else 'refusal'}]"
        for case_id, decisions in expectations.items()
        for engine in engines
    }
    pinned = {
        (name, engine, seed)
        for engine in engines
        for name in (PINNED_NATIVE_PROPERTIES if native_only else PINNED_PROPERTIES[engine])
        for seed in PINNED_SEEDS[engine]
    }
    session.config.stash[REQUIRED] |= {
        f"{PROPERTY_TESTS}::test_property[{obligation_id(*obligation)}]"
        for obligation in pinned
        | set(
            obligations(
                engines,
                session.config.getoption("--conformance-seeds"),
                session.config.getoption("--conformance-seed-start"),
                include_comparisons=not native_only,
            )
        )
    } | {f"{PROPERTY_TESTS}::test_corpus[{engine}]" for engine in engines}
    session.config.stash[REQUIRED] |= {
        f"{PROPERTY_TESTS}::test_known_gap[{gap.id}]"
        for gap in REPLAYED
        if gap.engine in engines and (not native_only or gap.property not in COMPARISON_PROPERTIES)
    }
    session.config.stash[REQUIRED] |= {
        f"{DTYPE_TESTS}::test_scalar_compatibility_and_real_execution[{engine}-{family}-{dtype}]"
        for engine in engines
        for family in PINNED_ENCODERS
        for dtype in PINNED_DTYPES
    }
    if native_only:
        session.config.stash[REQUIRED] |= {
            f"tests/conformance/test_field_semantics.py::{name}[diskgraph]"
            for name in PINNED_FIELD_SEMANTICS
        }
    session.config.stash[PASSED] = set()
    session.config.stash[REQUIRED] |= {
        f"tests/conformance/test_batch_boundaries.py::{test}[{engine}"
        f"{'-' + variant if variant else ''}]"
        for engine in engines
        if engine == "diskgraph"
        for test, variants in PINNED_BATCH_BOUNDARIES.items()
        for variant in variants
    }


@pytest.fixture(autouse=True)
def _fresh_engine_runs(request):
    """Harness tests patch engines in place, so they never reuse a cached engine run."""
    if request.node.path.name != "test_properties.py":
        from conformance.properties import clear_runs

        clear_runs()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
        item.config.stash.get(PASSED, set()).add(report.nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    complete = session.config.getoption("--conformance-complete")
    bad_outcomes = ("skipped", "xfailed", "xpassed", "deselected") if complete else ("skipped",)
    bad = reporter and any(
        "/conformance/" in report.nodeid
        for category in bad_outcomes
        for report in reporter.stats.get(category, ())
    )
    missing = session.config.stash.get(REQUIRED, set()) - session.config.stash.get(PASSED, set())
    if bad and reporter:
        reporter.write_sep("!", "required conformance checks were skipped, filtered, or xfailed")
    if missing and reporter:
        reporter.write_sep("!", "required engine checks did not pass")
        for nodeid in sorted(missing):
            reporter.write_line(nodeid)
    if bad or missing:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
