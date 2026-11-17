"""Fitting scalability reports must not turn partial measurements into acceptance."""

import copy
import json

import pytest

from test.kumo_pytorch.benchmarking import fitting_memory as fitting


def runs(large=190):
    return [
        {
            "count": count,
            "actual_count": count,
            "repeat": repeat,
            "status": "passed",
            "returncode": 0,
            "spec_sha256": "same",
            "peak_host_bytes": peak,
            "peak_anonymous_bytes": peak // 2,
            "anonymous_memory": {"status": "available"},
            "memory": {
                "status": "available",
                "details": {
                    "method": "sampled_cgroup_v2_memory_current",
                    "memory_scope": "cgroup_and_descendants",
                    "completed_operation": True,
                    "cgroup": f"/isolated/{count}-{repeat}",
                },
            },
        }
        for count, peak in ((25000, 100), (400000, large))
        for repeat in range(3)
    ]


@pytest.mark.parametrize("peak", [199, 200, 201, 1600])
def test_growth_is_reported_without_a_limit_and_all_peaks_are_retained(peak):
    result = fitting.compare(runs(peak), counts=(25000, 400000), repetitions=3)
    assert result["status"] == "passed"
    assert result["growth_ratio"] == peak / 100
    assert result["peaks_bytes"]["400000"] == [peak] * 3


@pytest.mark.parametrize(
    "fault",
    [
        "failed",
        "no_memory",
        "wrong_count",
        "missing",
        "duplicate",
        "method",
        "scope",
        "spec",
        "cgroup",
        "zero",
        "exit",
    ],
)
def test_incomplete_or_incomparable_workers_fail(fault):
    selected = copy.deepcopy(runs())
    last = selected[-1]
    if fault == "failed":
        last["status"] = "failed"
    elif fault == "no_memory":
        last["memory"]["status"] = "unavailable"
    elif fault == "wrong_count":
        last["actual_count"] = 399999
    elif fault == "missing":
        selected.pop()
    elif fault == "duplicate":
        last["repeat"] = 0
    elif fault in {"method", "scope", "cgroup"}:
        key = "memory_scope" if fault == "scope" else fault
        last["memory"]["details"][key] = (
            selected[0]["memory"]["details"][key] if fault == "cgroup" else "different"
        )
    elif fault == "spec":
        last["spec_sha256"] = "changed"
    elif fault == "zero":
        last["peak_host_bytes"] = 0
    elif fault == "exit":
        last["returncode"] = 124
    result = fitting.compare(selected, counts=(25000, 400000), repetitions=3)
    assert result["status"] == "failed" and result["growth_ratio"] is None


def test_campaign_continues_after_worker_failure_and_isolates_each_size(
    tmp_path, monkeypatch
):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"id": "any-dataset", "anchors": {"25000": {}, "400000": {}}})
    )
    calls = []

    def launch(command, log_path, *, timeout, isolate=False):
        calls.append((command, isolate))
        assert command[2] == "test.kumo_pytorch.benchmarking.fitting_memory"
        if "--count" in command:
            return 124  # All requested runs must still be attempted.
        return 0

    monkeypatch.setattr(fitting, "launch", launch)
    result = fitting.campaign(spec, tmp_path / "results", repetitions=2)
    assert result["status"] == "failed" and len(result["runs"]) == 4
    assert [isolated for _, isolated in calls] == [False, True, True, True, True]
    assert [
        int(cmd[cmd.index("--count") + 1]) for cmd, isolated in calls if isolated
    ] == [25000, 400000, 400000, 25000]
    assert all(run["returncode"] == 124 for run in result["runs"])
    assert (tmp_path / "results/report.json").is_file()


@pytest.mark.parametrize(
    "counts,repeat",
    [
        ((25000, 25000), 3),
        ((25000, 400000), 0),
    ],
)
def test_invalid_experiments_are_rejected(counts, repeat):
    with pytest.raises(ValueError):
        fitting.compare([], counts=counts, repetitions=repeat)


@pytest.mark.parametrize(
    "repeat,order", [(0, [25000, 400000]), (1, [400000, 25000]), (2, [25000, 400000])]
)
def test_selected_repetition_keeps_both_sizes_and_original_order(
    tmp_path, monkeypatch, repeat, order
):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"id": "any-dataset", "anchors": {"25000": {}, "400000": {}}})
    )
    calls = []

    def launch(command, log_path, *, timeout, isolate=False):
        calls.append((command, isolate))
        return 124 if isolate else 0

    monkeypatch.setattr(fitting, "launch", launch)
    result = fitting.campaign(spec, tmp_path / "result", repetitions=3, repeat=repeat)
    assert result["status"] == "failed" and result["all_repetitions_executed"] is False
    assert [run["repeat"] for run in result["runs"]] == [repeat, repeat]
    assert [
        int(cmd[cmd.index("--count") + 1]) for cmd, isolated in calls if isolated
    ] == order
    assert [isolated for _, isolated in calls] == [False, True, True]


def test_selected_repetition_passes_its_pair_without_claiming_all_repetitions(
    tmp_path, monkeypatch
):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"id": "any-dataset", "anchors": {"25000": {}, "400000": {}}})
    )

    def launch(command, log_path, *, timeout, isolate=False):
        if isolate:
            count = int(command[command.index("--count") + 1])
            output = tmp_path / "results" / f"{count}-repeat-1"
            output.mkdir()
            run = next(r for r in runs() if r["count"] == count and r["repeat"] == 1)
            run["spec_sha256"] = fitting.digest(spec)
            (output / "report.json").write_text(json.dumps(run))
        return 0

    monkeypatch.setattr(fitting, "launch", launch)
    result = fitting.campaign(spec, tmp_path / "results", repetitions=3, repeat=1)
    assert result["status"] == "passed"
    assert result["selected_repeat"] == 1 and not result["all_repetitions_executed"]
    assert [r["repeat"] for r in result["runs"]] == [1, 1]
    assert (
        fitting.compare(result["runs"], counts=(25000, 400000), repetitions=3)["status"]
        == "failed"
    )
