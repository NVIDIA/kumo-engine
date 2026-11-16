"""Original AML objectives resume exactly, including targets and iterator position."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from test.kumo_pytorch.checkpoint_resume import check_resume

from .fixtures import CUTOFF, build_fixture
from .test_aml import CASES
from .test_finetuning import save_pretrained


@pytest.mark.parametrize("checkpoint_after", [1, 2])
@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
def test_original_workload_resumes_mid_epoch(
    tmp_path, selection, family, checkpoint_after
):
    lifecycle, _ = save_pretrained(tmp_path, selection, family)
    fixture = build_fixture()
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    graph = family in {"graphsage", "combined"}
    stamp = int(CUTOFF.timestamp())
    # Five distinct fixture anchors give two complete batches and a short tail.
    requests = tuple(
        replace(
            fixture.training[i % 2],
            sample_id=f"resume-{i}",
            as_of_time=stamp - int(graph),
            targets=() if native else fixture.training[i % 2].targets,
            metadata={"prediction_time_ns": stamp * 10**9} if graph else {},
        )
        for i in range(5)
    )
    result = check_resume(lifecycle, requests, checkpoint_after=checkpoint_after)
    assert result["parent_pid"] == os.getpid() != result["pid"]
    assert result["workload"] == selection["selector"]["workload_id"]
    assert result["checkpoint_after"] == checkpoint_after
    assert result["final_iterator"]["batches"] == 3
    assert len(result["remaining"]) == 3 - checkpoint_after
    assert len(result["remaining"][-1]["sample_ids"]) == 1
    assert all(row["target_weight"] > 0 for row in result["remaining"])
    assert (
        result["remaining"][-1]["model_sha256"] != result["prefix"][-1]["model_sha256"]
    )

    # Mutation probes validate the comparison, not expected package failures.
    if family == "nuformer" and native and checkpoint_after == 1:
        directory = tmp_path / "mid-epoch-resume"
        original = (directory / "checkpoint.pt").read_bytes()
        for part in ("batches", "optimizer", "torch_rng"):
            checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
            if part == "batches":
                checkpoint[part]["batches"] = 0
            elif part == "optimizer":
                checkpoint[part]["state"] = {}
            else:
                checkpoint[part] = torch.Generator().manual_seed(999).get_state()
            torch.save(checkpoint, directory / "checkpoint.pt")
            (directory / "report.json").unlink(missing_ok=True)
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "test.kumo_pytorch.checkpoint_resume",
                    str(directory),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert child.returncode != 0, f"Lost {part} state was silently accepted"
            expected = (
                "duplicated or omitted anchors"
                if part == "batches"
                else "Mid-epoch continuation differs"
            )
            assert expected in child.stderr
            assert not (directory / "report.json").exists()
            (directory / "checkpoint.pt").write_bytes(original)
        (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
