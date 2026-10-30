# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded real Kumo iterable/DataLoader/adapter diagnostic for a runtime manifest.

Run once per backend in a fresh interpreter. --preinitialize exercises the
known PR424 fork-after-I/O failure; the loader timeout and finally block reap
the worker. Use an isolated environment containing the candidate DiskGraph
wheel and the checkout's kumo_core plus its existing dependencies.
"""

import argparse
import json
import os
import sys

import diskgraph
import kumo_core.diskgraph.lightning_data as lightning_data
import kumo_core.diskgraph.runtime_adapter as runtime_adapter
from torch.utils.data import DataLoader


def iterator_options() -> dict:
    fanout = {
        "fanout": 2,
        "mode": "deterministic_uniform",
        "absent_key_policy": "uniform_fallback",
    }
    return {
        "stage": "train",
        "sampling_plan": {"layers": [
            {"fanouts": {}, "default_fanout": fanout},
            {"fanouts": {}, "default_fanout": fanout},
        ]},
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "epoch": 0,
        "batch_size": 3,
        "entity_node_type": "users",
        "swap_edge_direction": False,
    }


def identity(batch):
    return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('mmap', 'buffered', 'direct'), required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--preinitialize', action='store_true')
    args = parser.parse_args()
    assert diskgraph._c.__file__.startswith(sys.prefix + os.sep), diskgraph._c.__file__
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    core_root = os.path.join(repo_root, 'packages/kumo-core/src') + os.sep
    assert lightning_data.__file__.startswith(core_root), lightning_data.__file__
    assert runtime_adapter.__file__.startswith(core_root), runtime_adapter.__file__
    os.environ['DG_EDGE_BACKEND'] = args.backend
    if args.preinitialize:
        index = diskgraph.RuntimeIndex(args.manifest, rnd_seed=42, num_threads=1)
        native, count = index.iter(**iterator_options(), fused=False)
        assert len(list(native)) == 2 and count == 6
    dataset = lightning_data.DiskgraphTaskBatchIterable(
        manifest_path=args.manifest,
        stage='train',
        sampling_plan=iterator_options()['sampling_plan'],
        batch_size=3,
        shuffle=False,
        drop_last=False,
        base_seed=42,
        shuffle_state=lightning_data._ShuffleStreamState(),
        num_seeds=6,
        replay=False,
        runtime_options={
            'num_threads': 1,
            'fused': False,
            'entity_node_type': 'users',
            'swap_edge_direction': False,
        },
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        multiprocessing_context='fork',
        collate_fn=identity,
        timeout=3,
    )
    iterator = iter(loader)
    batches = []
    outcome = {}
    try:
        batches = list(iterator)
        assert len(batches) == 2, f"expected two batches, got {len(batches)}"
        stage_ids = [sid for b in batches for sid in b['task']['tt_stage_node_ids'].to_pylist()]
        assert stage_ids == list(range(6)), f"unexpected stage IDs: {stage_ids}"
        converted = [runtime_adapter.runtime_batch_to_hetero(
            b,
            train_table_name='tt_e2e/train_seeds',
            seed_node_type='users',
            task_type='regression',
            index_np_time_unit='s',
        ) for b in batches]
        assert all(hetero.num_nodes > 0 and raw for hetero, raw, _ in converted), (
            f"empty converted output: {[(h.num_nodes, list(r)) for h, r, _ in converted]}"
        )
        outcome = {'result': 'passed', 'batches': len(batches), 'converted': len(converted)}
    except Exception as error:
        outcome = {'result': 'failed', 'error': str(error), 'error_type': type(error).__name__}
    finally:
        iterator._shutdown_workers()
    print(json.dumps({
        'backend': args.backend,
        'preinitialized': args.preinitialize,
        'native_module': diskgraph._c.__file__,
        'core_module': lightning_data.__file__,
        **outcome,
    }))
    return int(outcome['result'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
