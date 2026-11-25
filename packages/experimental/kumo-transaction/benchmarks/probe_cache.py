# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache one seed's history-window probe rows, so protocol and readout variants
are scored offline without re-embedding.

Three sides. With `--users heldout`, users are divided as `event_entity_split`
divides them (probe split rules); with `--users all`, every side holds every user and
only time separates them, as a production model scores existing customers.

- `train`: training-window events, every fraud plus `--negatives-per-positive`
  legitimate rows each. A smaller ratio is a subset.
- `stop`: validation-window events, a later period for early stopping and
  selection: `--stop-rows` uniform rows at the natural fraud rate, or with
  `--stop-negatives-per-positive`, every fraud plus that many legitimate rows
  each. The checkpoint never saw these events; pretraining stops at the train
  cutoff.
- `eval`: test-window events: every one, or with `--eval-negatives`, every
  fraud plus that many uniform legitimate rows, each weighted (`w`) by the
  inverse of its sampling rate so weighted PR-AUC estimates the full window's.

Each side saves the raw row and the history aggregates (`TabularFeaturizer`,
fitted on `train`), labels, users, event times, and three frozen readouts:
`hist_last` (last token after the history, the recipe's embedding),
`hist_event` (mean over the scored transaction's own tokens, after the
history), and `single_last` (the transaction alone, the recipe's control).
Flat token layouts only.

    uv run python benchmarks/probe_cache.py --config recipes/blueprint_v1/config.yaml \\
      --checkpoint outputs/blueprint_v1/seeds/seed_42/model.pt \\
      --out outputs/probe_cache/blueprint_v1_seed42 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfm.adaptation.batching import minibatches
from tfm.config import load_recipe_config
from tfm.data.sequence_builders.event_rows import EventHistoryRow, build_event_history_rows
from tfm.evaluation.tabular import TabularFeaturizer
from tfm.inference.embedding import trim_padding
from tfm.runtime.common import parse_duration
from tfm.runtime.evaluate import _collate_rows, event_entity_split
from tfm.runtime.pipeline import setup_run
from tfm.training.checkpointing import load_checkpoint

CHUNK_ROWS = 32_768


def _readouts(
    model: torch.nn.Module, batch: Any, *, event_tokens: int, device: torch.device, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Last-token state, and the mean state over each row's last event tokens."""
    last, event = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for chunk in minibatches(batch, size):
            chunk = trim_padding(chunk).to(device)
            out = model(chunk, compute_loss=False)
            states = out.hidden_states.float()
            lengths = chunk.attention_mask.sum(dim=1).long()
            # Flat rows end `... <sep> field_1 .. field_n <eos>`: the scored
            # transaction is the `event_tokens` positions before `<eos>`.
            offsets = torch.arange(event_tokens, device=device)
            idx = (lengths - 1 - event_tokens).clamp(min=0)[:, None] + offsets[None, :]
            gathered = states.gather(1, idx[:, :, None].expand(-1, -1, states.shape[-1]))
            last.append(out.embeddings.float().cpu().numpy().astype(np.float16))
            event.append(gathered.mean(dim=1).cpu().numpy().astype(np.float16))
    return np.concatenate(last), np.concatenate(event)


def _embed(
    model: torch.nn.Module,
    rows: Sequence[EventHistoryRow],
    tokenizer: Any,
    *,
    max_tokens: int,
    event_tokens: int,
    device: torch.device,
    size: int,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {"hist_last": [], "hist_event": [], "single_last": []}
    for start in range(0, len(rows), CHUNK_ROWS):
        chunk = rows[start : start + CHUNK_ROWS]
        hist = _collate_rows([r.history for r in chunk], tokenizer, max_tokens=max_tokens)
        last, event = _readouts(model, hist, event_tokens=event_tokens, device=device, size=size)
        single = _collate_rows([r.event_only for r in chunk], tokenizer, max_tokens=max_tokens)
        alone, _ = _readouts(model, single, event_tokens=event_tokens, device=device, size=size)
        parts["hist_last"].append(last)
        parts["hist_event"].append(event)
        parts["single_last"].append(alone)
        print(f"  embedded {start + len(chunk):,} / {len(rows):,}", flush=True)
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negatives-per-positive", type=float, default=40.0)
    parser.add_argument("--stop-rows", type=int, default=200_000)
    parser.add_argument("--stop-negatives-per-positive", type=float)
    parser.add_argument("--eval-negatives", type=int)
    parser.add_argument("--users", choices=("heldout", "all"), default="heldout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, help="override the config seed (the run's seed)")
    args = parser.parse_args()

    config = load_recipe_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    if config.model.composition.granularity.type != "flat":
        raise SystemExit("probe_cache.py reads flat token layouts only.")
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = setup_run(config=config, run_dir=args.out, device=args.device)
    data, model, device = artifacts.data, artifacts.model, artifacts.device
    load_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    split = event_entity_split(data, config.seed)
    held_out = set(split.evaluation)
    every_user = sorted(set(data.batches[0].entity_ids))
    train_users = every_user if args.users == "all" else sorted(set(every_user) - held_out)
    eval_users = every_user if args.users == "all" else sorted(held_out)
    common = {
        "max_events": config.data.sequence_length,
        "observation_window": parse_duration(config.data.observation_window),
        "seed": config.seed,
    }
    sides = {
        "train": build_event_history_rows(
            data.events,
            entities=train_users,
            window=data.cutoffs.windows["train"],
            label_cutoff=data.cutoffs.train,
            max_rows=None,
            negatives_per_positive=args.negatives_per_positive,
            **common,
        ),
        "stop": build_event_history_rows(
            data.events,
            entities=train_users,
            window=data.cutoffs.windows["validation"],
            label_cutoff=data.cutoffs.validation,
            max_rows=None if args.stop_negatives_per_positive else args.stop_rows,
            negatives_per_positive=args.stop_negatives_per_positive,
            **common,
        ),
        "eval": build_event_history_rows(
            data.events,
            entities=eval_users,
            window=data.cutoffs.windows["test"],
            label_cutoff=data.cutoffs.test,
            max_rows=None,
            **common,
        ),
    }
    weights = {side: np.ones(len(rows), dtype=np.float64) for side, rows in sides.items()}
    if args.eval_negatives is not None:
        rows = sides["eval"]
        negatives = [i for i, r in enumerate(rows) if r.label < 0.5]
        kept = min(args.eval_negatives, len(negatives))
        rng = np.random.default_rng(config.seed)
        keep_neg = set(rng.choice(negatives, size=kept, replace=False).tolist())
        keep = [i for i, r in enumerate(rows) if r.label >= 0.5 or i in keep_neg]
        sides["eval"] = [rows[i] for i in keep]
        weights["eval"] = np.array(
            [1.0 if rows[i].label >= 0.5 else len(negatives) / kept for i in keep]
        )
    profiles = {p.entity_id: p for p in data.profiles}
    baselines = config.evaluation.tabular_baselines
    fit_args = {
        "top_k": baselines.top_k,
        "coded_keys": tuple(config.features.categorical_keys),
        "exclude_keys": tuple(baselines.exclude_keys),
    }
    train_seqs = [r.history for r in sides["train"]]
    featurizers = {
        "raw": TabularFeaturizer.fit(train_seqs, profiles, history=False, **fit_args),
        "full": TabularFeaturizer.fit(train_seqs, profiles, history=True, **fit_args),
    }
    event_tokens = len(config.features.fields)
    meta: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "seed": config.seed,
        "negatives_per_positive": args.negatives_per_positive,
        "stop_negatives_per_positive": args.stop_negatives_per_positive,
        "eval_negatives": args.eval_negatives,
        "users": args.users,
        "split": split.as_dict(),
        "names": {name: list(f.names) for name, f in featurizers.items()},
        "sides": {},
    }
    for side, rows in sides.items():
        t0 = time.time()
        print(f"{side}: {len(rows):,} rows, {int(sum(r.label for r in rows)):,} frauds", flush=True)
        seqs = [r.history for r in rows]
        arrays = {
            name: featurizer.transform(seqs, profiles) for name, featurizer in featurizers.items()
        }
        arrays.update(
            _embed(
                model,
                rows,
                data.tokenizer,
                max_tokens=config.data.max_tokens,
                event_tokens=event_tokens,
                device=device,
                size=config.evaluation.inference_batch_size,
            )
        )
        np.savez(
            args.out / f"{side}.npz",
            y=np.array([r.label for r in rows], dtype=np.float32),
            w=weights[side],
            user=np.array([s.entity_id for s in seqs]),
            time=np.array([int(s.events[-1].event_timestamp.timestamp()) for s in seqs]),
            **arrays,
        )
        meta["sides"][side] = {
            "rows": len(rows),
            "weighted_rows": float(weights[side].sum()),
            "frauds": int(sum(r.label for r in rows)),
            "seconds": round(time.time() - t0, 1),
        }
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
