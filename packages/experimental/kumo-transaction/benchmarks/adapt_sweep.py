# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune one pretrained checkpoint several ways, on the proxy's tasks.

Each variant is one `FineTune` (LoRA or full, linear or `joint_fusion` head),
trained on a copy of the checkpoint and scored on the same rows the proxy
probes. A `joint_fusion` variant with `raw: weak` reads the `xgboost_raw` row
(the last event's fields plus the profile, no history aggregates) instead of
the batch's own `numeric_features`, so it is the fine-tuned counterpart of
`xgboost_raw_embedding`. Columns constant on the training side are dropped:
standardizing them would blow up any evaluation value that differs.

    uv run --all-extras python benchmarks/adapt_sweep.py \\
        --spec benchmarks/sweeps/blueprint_v2_adapt.yaml \\
        --out outputs/proxy/adapt.json

Spec format:

    run_dir: outputs/proxy/stage1/adamw_lr1e-3   # seed_config.yaml and model.pt
    tasks: [fraud, churn_6m]
    variants:
      - name: lora_r8_mlp
        finetune: {method: lora, learning_rate: 0.001, validation_fraction: 0.2}
        lora: {rank: 8, alpha: 8, targets: [qkv, out, ff.0, ff.2]}
      - name: lora_fusion_weak
        finetune: {method: lora, head: joint_fusion}
        raw: weak

`--out` is rewritten after every variant; variants already in it are skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from proxy_eval import proxy_tasks, reference_arms
from seed_range import _commit

from tfm.adaptation.finetune import FineTune, finetune_binary
from tfm.config import load_recipe_config
from tfm.evaluation.tabular import TabularFeaturizer
from tfm.runtime.common import apply_device_override
from tfm.runtime.data import PreparedData, prepare_data
from tfm.runtime.evaluate import ProbeTask
from tfm.runtime.pipeline import build_model
from tfm.training.checkpointing import load_checkpoint
from tfm.training.device import resolve_device
from tfm.training.lora import LoRA


def weak_raw_task(task: ProbeTask, data: PreparedData, config: Any) -> tuple[ProbeTask, int]:
    """`task` with each batch's `numeric_features` replaced by the `xgboost_raw` row."""
    profiles = {p.entity_id: p for p in data.profiles}
    raw = TabularFeaturizer.fit(
        task.train_sequences,
        profiles,
        top_k=config.evaluation.tabular_baselines.top_k,
        coded_keys=tuple(config.features.categorical_keys),
        history=False,
    )
    x_tr = raw.transform(task.train_sequences, profiles)
    x_ev = raw.transform(task.eval_sequences, profiles)
    keep = x_tr.std(axis=0) > 0
    train_batch = dataclasses.replace(
        task.train_batch, numeric_features=torch.tensor(x_tr[:, keep], dtype=torch.float32)
    )
    eval_batch = dataclasses.replace(
        task.eval_batch, numeric_features=torch.tensor(x_ev[:, keep], dtype=torch.float32)
    )
    return dataclasses.replace(task, train_batch=train_batch, eval_batch=eval_batch), int(
        keep.sum()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/proxy/cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", default=None, help="Run these variant names only.")
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    run_dir = Path(spec["run_dir"])
    config = apply_device_override(load_recipe_config(run_dir / "seed_config.yaml"), args.device)
    device = resolve_device(config.runtime.device)
    names = tuple(spec.get("tasks") or ("fraud", "churn_6m"))
    variants = [v for v in spec["variants"] if args.only is None or v["name"] in args.only]

    done: dict[str, Any] = {}
    if args.out.exists():
        done = {r["name"]: r for r in json.loads(args.out.read_text())["runs"]}
    started = time.perf_counter()
    data = prepare_data(config, fixture_dir=args.out.parent / f"{args.out.stem}_book")
    tasks = proxy_tasks(data, config, names)
    record: dict[str, Any] = {
        "spec": str(args.spec),
        "run_dir": str(run_dir),
        "commit": _commit(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "prepare_seconds": round(time.perf_counter() - started, 1),
        "tasks": {name: task.summary() for name, task in tasks.items()},
        "reference": reference_arms(tasks, data, config, args.cache_dir),
        "runs": list(done.values()),
    }
    model = build_model(config, data.tokenizer.vocab_size)
    load_checkpoint(model, run_dir / "model.pt")
    weak: dict[str, tuple[ProbeTask, int]] = {}
    for variant in variants:
        name = variant["name"]
        if name in done:
            continue
        lora = LoRA(
            **{k: tuple(v) if k == "targets" else v for k, v in (variant.get("lora") or {}).items()}
        )
        ft = FineTune(lora=lora, **(variant.get("finetune") or {}))
        run: dict[str, Any] = {"name": name, "arm": ft.arm, "spec": variant, "tasks": {}}
        for task_name, task in tasks.items():
            n_raw = None
            if variant.get("raw") == "weak":
                if task_name not in weak:
                    weak[task_name] = weak_raw_task(task, data, config)
                task, n_raw = weak[task_name]
            t0 = time.perf_counter()
            metrics, manifest = finetune_binary(
                model,
                task.train_batch,
                task.eval_batch,
                task.y_train,
                task.y_eval,
                ft,
                device=device,
                seed=config.seed,
                k=config.evaluation.precision_at_k,
                precision=config.training.precision,
            )
            manifest.pop("adapted_modules", None)
            run["tasks"][task_name] = {
                "metrics": metrics,
                "manifest": manifest,
                "n_raw_features": n_raw,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            print(
                json.dumps(
                    {"name": name, "task": task_name, "pr_auc": metrics["pr_auc"], **manifest},
                    default=str,
                ),
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        record["runs"].append(run)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
