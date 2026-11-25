# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pretrain several config variants and score each with the fast proxy.

A spec file names a base config and a list of variants, each a set of dotted
overrides. Variants that share `data` and `features` share one prepared
dataset, so only the first of them pays the data preparation. Each variant is
trained with the recipe's own trainer, checkpointed under `--run-root`, and
scored by `benchmarks/proxy_eval.py`. A variant with `train: false` is scored
untrained: the random-initialization control. A variant with `checkpoint:`
loads that `model.pt` instead of training.

    uv run --all-extras python benchmarks/proxy_sweep.py \\
        --spec benchmarks/sweeps/blueprint_v2_stage1.yaml \\
        --run-root outputs/proxy/stage1 --summary outputs/proxy/stage1.json

Spec format:

    base: path/to/config.yaml
    set: {training.max_steps: 1500}      # applied to every variant
    variants:
      - name: lr1e-3
        set: {training.learning_rate: 0.001}
      - name: random_init
        train: false

`--summary` is rewritten after every variant, so a long sweep leaves partial
evidence. Variants already in the summary are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from proxy_eval import proxy_eval, proxy_tasks, summary_row
from seed_range import _commit

from tfm.config import RecipeConfig, load_recipe_config
from tfm.runtime.common import apply_device_override
from tfm.runtime.data import prepare_data
from tfm.runtime.pipeline import RunArtifacts, build_model, run_train
from tfm.training.checkpointing import load_checkpoint
from tfm.training.device import resolve_device
from tfm.training.reproducibility import seed_everything


def _override(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value


def variant_config(base: Path, common: dict[str, Any], own: dict[str, Any]) -> RecipeConfig:
    raw = load_recipe_config(base).model_dump(mode="json")
    _override(raw, common)
    _override(raw, own)
    return RecipeConfig.model_validate(raw)


def data_key(config: RecipeConfig) -> str:
    raw = config.model_dump(mode="json")
    payload = {"data": raw["data"], "features": raw["features"], "seed": config.seed}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/proxy/cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", default=None, help="Run these variant names only.")
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    base = Path(spec["base"])
    common = dict(spec.get("set") or {})
    tasks = tuple(spec.get("tasks") or ("history_window", "fraud", "churn_6m"))
    xgboost_trials = int(spec.get("xgboost_trials", 8))
    variants = [v for v in spec["variants"] if args.only is None or v["name"] in args.only]
    configs = [(v, variant_config(base, common, dict(v.get("set") or {}))) for v in variants]
    # Group variants by prepared data, in spec order of first appearance.
    order: dict[str, int] = {}
    for _v, cfg in configs:
        order.setdefault(data_key(cfg), len(order))
    configs.sort(key=lambda vc: order[data_key(vc[1])])

    done: dict[str, Any] = {}
    if args.summary.exists():
        done = {r["name"]: r for r in json.loads(args.summary.read_text())["runs"]}
    record: dict[str, Any] = {
        "spec": str(args.spec),
        "base": str(base),
        "common": common,
        "commit": _commit(),
        "hardware": {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "runs": list(done.values()),
    }
    current_key, data, prepare_seconds = None, None, 0.0
    probe_cache: dict[str, Any] = {}
    for variant, config in configs:
        name = variant["name"]
        if name in done:
            continue
        config = apply_device_override(config, args.device)
        key = data_key(config)
        if key != current_key:
            data = probe_cache = None
            gc.collect()
            started = time.perf_counter()
            data = prepare_data(config, fixture_dir=args.run_root / "pragmatiq_book")
            prepare_seconds = round(time.perf_counter() - started, 1)
            probe_cache = {}
            current_key = key
        assert data is not None and probe_cache is not None
        if not probe_cache:
            probe_cache["tasks"] = proxy_tasks(data, config, tasks)
        seed_everything(config.seed)
        device = resolve_device(config.runtime.device)
        run_dir = args.run_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "seed_config.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        model = build_model(config, data.tokenizer.vocab_size)
        artifacts = RunArtifacts(
            run_dir=run_dir, config=config, data=data, model=model, device=device
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        stats: dict[str, Any] = {}
        if variant.get("checkpoint"):
            load_checkpoint(model, Path(variant["checkpoint"]))
        elif variant.get("train", True):
            stats = dict(run_train(artifacts))
        trained = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        result = proxy_eval(
            model,
            data,
            config,
            device,
            tasks=tasks,
            cache_dir=args.cache_dir,
            xgboost_trials=xgboost_trials,
            probe_tasks=probe_cache["tasks"],
        )
        (run_dir / "proxy.json").write_text(json.dumps(result, indent=2, default=str))
        run = {
            "name": name,
            "set": variant.get("set") or {},
            "train": bool(variant.get("train", True)) and not variant.get("checkpoint"),
            "checkpoint": variant.get("checkpoint"),
            "run_dir": str(run_dir),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "train_stats": stats,
            "seconds": {
                "prepare": prepare_seconds,
                "train": round(trained - started, 1),
                "proxy": result["seconds"],
            },
            "peak_gpu_memory_gib": None if peak is None else round(peak, 2),
            "proxy": summary_row(result),
            "next_event_ce_by_key": result["heldout"]["next_event_ce_by_key"],
        }
        prepare_seconds = 0.0
        record["runs"].append(run)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"name": name, **run["seconds"], **run["proxy"]}, indent=1), flush=True)
        del model, artifacts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
