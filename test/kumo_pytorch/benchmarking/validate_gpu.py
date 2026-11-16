"""Exercise real CUDA collectors on the engine CI runner; missing hardware fails.

This is collector acceptance, not full dataset benchmark acceptance. Device
selection follows the runner's visible CUDA devices; no hardware UUID is pinned.
"""

import argparse
import json
from pathlib import Path

from .interface import BatchCounts, Workload
from .reporting import write_json
from .runner import run


def factory(config, directory):
    import torch

    device = config["device"]
    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation requires runner-provided CUDA")
    state = {}

    def prepare(recorder):
        torch.manual_seed(0)
        state["model"] = torch.nn.Linear(1024, 1024).to(device)
        state["optimizer"] = torch.optim.SGD(state["model"].parameters(), lr=0.001)
        state["input"] = torch.randn(128, 1024, pin_memory=True)

    def batches(epoch):
        # A freed allocation must affect this epoch's peak, not later epochs.
        temporary = torch.empty(
            (64 if epoch == 0 else 4) * 1024**2,
            dtype=torch.uint8,
            device=device,
        )
        temporary.fill_(1)
        del temporary
        for _ in range(256):
            yield state["input"]

    def step(batch, recorder):
        with recorder.span("transfer.host"):
            inputs = batch.to(device, non_blocking=True)
        optimizer = state["optimizer"]
        optimizer.zero_grad(set_to_none=True)
        with recorder.span("model.forward"):
            prediction = state["model"](inputs)
        with recorder.span("model.loss"):
            loss = prediction.square().mean()
        with recorder.span("model.backward"):
            loss.backward()
        with recorder.span("model.optimizer"):
            optimizer.step()
        state["loss"] = loss.detach()
        return True

    def close():
        try:
            if "loss" in state and not torch.isfinite(state["loss"]).item():
                raise AssertionError("GPU training produced nonfinite loss")
        finally:
            state.clear()

    return Workload(
        identity={
            "dataset": "generated-collector-fixture",
            "workload": "linear-training",
        },
        prepare=prepare,
        batches=batches,
        count=lambda batch: BatchCounts(
            128,
            0,
            output_elements=batch.numel(),
            output_bytes=batch.numel() * batch.element_size(),
        ),
        step=step,
        synchronize=lambda: torch.cuda.synchronize(device),
        close=close,
        device=device,
    )


def check(output, result):
    if result["execution_status"] != "completed":
        raise AssertionError(f"GPU execution failed: {result['errors']}")
    rows = [
        json.loads(line)
        for line in (output / "observations.jsonl").read_text().splitlines()
    ]
    for epoch in range(2):

        def values(metric, epoch=epoch):
            selected = [
                row
                for row in rows
                if row.get("scope", {}).get("epoch") == epoch
                and row.get("metric") == metric
            ]
            if not selected or any(row["status"] != "available" for row in selected):
                raise AssertionError(f"Missing GPU evidence: {epoch=} {metric=}")
            return [row["value"] for row in selected]

        allocated = values("gpu.peak_allocated_bytes")[0]
        reserved = values("gpu.peak_reserved_bytes")[0]
        if not 0 < allocated <= reserved:
            raise AssertionError("Invalid CUDA allocation peaks")
        transfers = values("transfer.device_ns")
        if len(transfers) != 256 or sum(transfers) <= 0:
            raise AssertionError("Expected 256 measured host-to-device transfers")
        if not 0 <= values("gpu.utilization_percent")[0] <= 100:
            raise AssertionError("Invalid device utilization")
        if epoch == 0:
            first_peak = allocated
        elif allocated >= first_peak - 32 * 1024**2:
            raise AssertionError("Allocated peak was not reset between epochs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(
        factory,
        {"device": args.device},
        args.output,
        repetitions=1,
        epochs=2,
        gpu_interval=0.01,
    )
    validation = {"scope": "GPU collector only", "benchmark_status": result["status"]}
    try:
        check(args.output, result)
        validation["status"] = "passed"
    except Exception as error:  # noqa: BLE001 -- retain validation failures
        validation.update(status="failed", error=str(error))
    write_json(args.output / "gpu-validation.json", validation)
    print(json.dumps(validation))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
