# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the default Blueprint-v2 model with batches fetched through Kumo."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import kumo.settings
import torch
from adapter import KumoToKVT
from kumo.pytorch import api

from contract import make_anchors, make_contract, make_dataset
from tfm.api.builders import profile_event_decoder_from_config
from tfm.config import load_recipe_config
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.data.splits.temporal import parse_timestamp
from tfm.training import EagerTrainer, seed_everything
from tfm.training.progress import TrainingProgress

ROOT = Path(__file__).resolve().parents[2]


class KumoBatches:
    """Let Kumo order native anchor batches; prefetch one adapted CPU batch."""

    def __init__(
        self,
        task: api.Task,
        adapter: KumoToKVT,
        worker: ThreadPoolExecutor,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.task, self.adapter, self.worker = task, adapter, worker
        self.batch_size, self.seed, self.shuffle = batch_size, seed, shuffle
        self.samples, self.epoch = task.count("train"), 0

    def __len__(self) -> int:
        return (self.samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[KeyValueTimeBatch]:
        source = iter(
            self.task.batches(
                split="train",
                batch_size=self.batch_size,
                seed=self.seed,
                epoch=self.epoch,
                shuffle=self.shuffle,
                drop_last=False,
                output_format="model",
            )
        )
        self.epoch += 1

        def fetch() -> KeyValueTimeBatch | None:
            batch = next(source, None)
            return None if batch is None else self.adapter(batch)

        pending = self.worker.submit(fetch)
        while (batch := pending.result()) is not None:
            # Build the next CPU batch while the trainer processes this one.
            pending = self.worker.submit(fetch)
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/kumo-engine/data")
    parser.add_argument("--anchors", type=Path, help="Override the prepared anchor table")
    parser.add_argument("--scratch-dir", type=Path, help="Override Kumo's managed scratch root")
    parser.add_argument(
        "--index", type=Path, help="Reuse a completed DiskGraph index under the Kumo scratch root"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kumo-engine/train")
    parser.add_argument("--config", type=Path, default=ROOT / "recipes/blueprint_v2/config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, help="Override training duration")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    config = load_recipe_config(args.config)
    if config.training.backend != "eager":
        parser.error("This example supports the single-process eager trainer")
    steps = config.training.max_steps if args.steps is None else args.steps
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    seed_everything(config.seed)
    kumo.settings.configure(scratch_dir=args.scratch_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    contract = make_contract(config.data.sequence_length, config.training.batch_size)
    anchors = make_anchors((args.anchors or args.data / "anchors.parquet").resolve())
    if args.index is None:
        dataset = make_dataset(args.data.resolve())
        # Precompute column statistics at the train boundary during ingestion.
        # Fitting still uses the latest selected train anchor as its cutoff.
        encoder_cutoff = int(parse_timestamp(config.data.temporal_split.train_end).timestamp()) - 1
        engine = api.DiskGraphEngine(num_threads=args.threads, encoder_cutoffs=(encoder_cutoff,))
    else:
        dataset = None
        # The completed index supplies its stored build identity and schema.
        snapshot = SimpleNamespace(index_path=args.index, id=None)
        engine = api.DiskGraphEngine.from_snapshot(snapshot, num_threads=args.threads)
    with ExitStack() as stack:
        print("Preparing DiskGraph and fitting input encoders", flush=True)
        plan = stack.enter_context(api.prepare(contract, dataset, engine))
        print(f"DiskGraph index: {engine.index}", flush=True)
        task = stack.enter_context(plan.materialize(anchors, fit_split="train"))
        adapter = KumoToKVT(task)
        # Close the producer before the task/index, including when training fails.
        worker = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        batches = KumoBatches(
            task,
            adapter,
            worker,
            config.training.batch_size,
            config.seed,
            config.training.shuffle,
        )
        model = profile_event_decoder_from_config(config, vocab_size=adapter.vocab_size)
        # Preserve the configured learning-rate schedule when overriding duration.
        optimizer = config.training.optimizer.model_copy(
            update={
                "schedule_steps": config.training.optimizer.schedule_steps
                or config.training.max_steps,
            }
        )
        progress = TrainingProgress(log_every=config.training.log_every)
        result = EagerTrainer().fit(
            model,
            batches,
            max_steps=steps,
            learning_rate=config.training.learning_rate,
            optimizer=optimizer,
            seed=config.seed,
            device=device,
            shuffle=False,  # Kumo owns seeded anchor shuffling and epoch order.
            precision=config.training.precision,
            progress=progress,
        )
        task.to_artifact().save(args.output / "fitted")
        api.save_weights(args.output / "model.pt", model.state_dict(), task.fitted_state)
        (args.output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
        (args.output / "training.json").write_text(
            json.dumps(
                {
                    "result": result,
                    "curve": progress.curve,
                    "anchors": str(args.anchors or args.data / "anchors.parquet"),
                    "train_samples": batches.samples,
                    "steps": steps,
                    "device": str(device),
                    "vocab_size": adapter.vocab_size,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
