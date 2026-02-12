#!/usr/bin/env python3
"""Run E2E Kumo training with diskgraph sampling on social_sampling_50m.

Usage (from kumo repo root):
  python ../kumo-diskgraph/scripts/e2e_train.py \
    --data-root /path/to/social_sampling_50m \
    --server grpc://localhost:8815
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure we import the local sibling `kumo` repo (this workspace), not some
# other editable install from a different checkout.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_KUMO_ROOT = _WORKSPACE_ROOT / "kumo"
if str(_LOCAL_KUMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_KUMO_ROOT))

import pandas as pd
import pyarrow.flight as flight
import pyarrow.parquet as pq
import torch
from torch_geometric.sampler import NodeSamplerInput

import kumo
from kumo.engine.diskgraph_sampler import DiskgraphGraphStore
from kumo.typing import TaskType
from kumoapi.encoder import Datetime, Index, Numerical
from kumoml.common.config import EncodersConfig, TargetEncoderConfig, TaskConfig
from kumoml.common.data import InMemoryFeatureStore
from kumoml.enterprise.nn import HeteroGNN, HeteroTensorFrameEncoder, NodePredHead


EdgeType = Tuple[str, str, str]


def fetch_schema_mapping(server_address: str) -> dict:
    client = flight.connect(server_address)
    result = list(client.do_action(flight.Action("get_schema_mapping", b"")))
    if not result:
        raise RuntimeError("Gateway returned no schema mapping payload")
    return json.loads(result[0].body.to_pybytes())


def edge_types_from_mapping(mapping: dict) -> List[EdgeType]:
    return [
        (rel["src_type"], rel["name"], rel["dst_type"])
        for rel in mapping.get("edge_relations", [])
        if rel.get("name")
    ]


def resolve_parquet_path(data_root: Path, table_name: str) -> Path:
    file_path = data_root / f"{table_name}.parquet"
    dir_path = data_root / table_name
    if file_path.exists():
        return file_path
    if dir_path.exists() and dir_path.is_dir():
        return dir_path
    raise FileNotFoundError(
        f"Missing parquet table: expected {file_path} or directory {dir_path}"
    )


def describe_parquet_table(data_root: Path, table_name: str) -> tuple[Path, list[str]]:
    path = resolve_parquet_path(data_root, table_name)
    sample_file = path
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {path}")
        sample_file = parquet_files[0]
    schema = pq.read_schema(sample_file)
    return path, schema.names


def load_parquet_table(
    data_root: Path,
    table_name: str,
    row_limit: int | None = None,
) -> pd.DataFrame:
    path = resolve_parquet_path(data_root, table_name)
    df = pd.read_parquet(path)
    if row_limit is not None and row_limit > 0:
        return df.head(row_limit).copy()
    return df


def infer_col(columns: List[str], candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(f"Could not infer required column from candidates={candidates}")


def infer_time_col(df: pd.DataFrame, table_name: str) -> str:
    candidates = [
        "time", "timestamp", "ts", "created_at", "created_ts",
        "event_time", f"{table_name}_ts",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    dt_cols = [
        col for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    if dt_cols:
        return dt_cols[0]
    raise ValueError(f"No time-like column found for table '{table_name}'")


def to_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", utc=False)


def create_user_post_train_table(
    users_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    user_key: str,
    post_user_key: str,
    post_time_col: str,
    num_windows: int,
    window_seconds: int,
) -> pd.DataFrame:
    post_df = posts_df[[post_user_key, post_time_col]].copy()
    post_df[post_time_col] = to_datetime_series(post_df[post_time_col])
    post_df = post_df.dropna(subset=[post_time_col]).sort_values(post_time_col)

    offset = pd.Timedelta(seconds=window_seconds)
    max_t = post_df[post_time_col].max()
    cur_t = max_t - offset

    rows: List[pd.DataFrame] = []
    for i in range(num_windows):
        hist_start = cur_t - offset
        hist_end = cur_t
        fut_end = cur_t + offset

        hist_mask = (post_df[post_time_col] > hist_start) & (post_df[post_time_col] <= hist_end)
        fut_mask = (post_df[post_time_col] > cur_t) & (post_df[post_time_col] <= fut_end)

        active_users = post_df.loc[hist_mask, post_user_key].dropna().drop_duplicates()
        if active_users.empty:
            cur_t = cur_t - offset
            continue

        targets = post_df.loc[fut_mask, post_user_key].dropna().drop_duplicates()
        target_set = set(targets.tolist())

        frame = pd.DataFrame({
            user_key: active_users.values,
            "target": [u in target_set for u in active_users.values],
            "time": cur_t,
            "train_mask": i >= 2,
            "val_mask": i == 1,
            "test_mask": i == 0,
        })
        rows.append(frame)
        cur_t = cur_t - offset

    if not rows:
        raise RuntimeError("Failed to build training table: no windows produced rows")

    train_df = pd.concat(rows, ignore_index=True)
    # Ensure bool masks:
    for col in ["train_mask", "val_mask", "test_mask", "target"]:
        train_df[col] = train_df[col].astype(bool)
    return train_df


def build_encoders(df_dict: Dict[str, pd.DataFrame], user_key: str) -> EncodersConfig:
    encoders = EncodersConfig()

    def add_table_encoders(table_name: str, skip_cols: set[str]) -> None:
        for col in df_dict[table_name].columns:
            if col in skip_cols:
                continue
            dtype = df_dict[table_name][col].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                encoders[table_name][col] = Datetime()
            elif pd.api.types.is_numeric_dtype(dtype):
                encoders[table_name][col] = Numerical()
            else:
                encoders[table_name][col] = Index()

    add_table_encoders("users", {user_key})
    for table in ["posts", "comments", "likes", "follows"]:
        if table in df_dict:
            add_table_encoders(table, set())

    encoders.target = TargetEncoderConfig("target", Index())
    return encoders


def build_time_col_dict(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for table, df in df_dict.items():
        # Keep time columns for graph tables. We intentionally do not register
        # train_table time here so the first E2E path runs with non-temporal
        # sampling semantics.
        if table == "train_table":
            continue
        dt_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if dt_cols:
            out[table] = dt_cols[0]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diskgraph E2E training script")
    parser.add_argument("--server", type=str, default="grpc://localhost:8815")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-neighbors", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps-per-epoch", type=int, default=500)
    parser.add_argument("--num-windows", type=int, default=12)
    parser.add_argument("--window-seconds", type=int, default=3600)
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional cap per table for smoke tests (loads head(N) rows)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def assert_edge_type_alignment(trainer: kumo.Trainer, input_node_type: str) -> None:
    train_table_name = trainer.feature_store.train_table_name
    fkey = trainer.feature_store[train_table_name, "fkey", :]
    train_mask = trainer.feature_store[train_table_name, "train_mask", :]
    seed_time = None

    seed_nodes = fkey[train_mask]
    if seed_nodes.dim() > 1:
        seed_nodes = seed_nodes[:, 0]
    seed_nodes = seed_nodes.view(-1).to(torch.long)

    if trainer.feature_store.has_train_table_time_col:
        time_v = trainer.feature_store[train_table_name, "time", :]
        seed_time = time_v[train_mask].view(-1)

    if seed_nodes.numel() == 0:
        raise RuntimeError("No train seeds available for edge-type alignment assertion")

    sample_size = min(256, seed_nodes.numel())
    seed_nodes = seed_nodes[:sample_size]
    if seed_time is not None:
        seed_time = seed_time[:sample_size]
    input_id = torch.arange(sample_size, dtype=torch.long)

    sample_input = NodeSamplerInput(
        input_id=input_id,
        node=seed_nodes,
        time=seed_time,
        input_type=input_node_type,
    )
    result = trainer.graph_sampler.sample_from_nodes(sample_input)

    graph_edge_types = {
        attr.edge_type
        for attr in trainer.graph_store.get_all_edge_attrs()
    }
    sampled_edge_types = set(result.row.keys())

    assert sampled_edge_types, "Sampler returned no edges"
    missing = sampled_edge_types - graph_edge_types
    assert not missing, (
        f"Sampled edge types missing from graph store attrs: {missing}"
    )
    print(
        "Edge-type alignment check passed: "
        f"sampled={len(sampled_edge_types)}, graph_store={len(graph_edge_types)}"
    )


def main() -> None:
    args = parse_args()
    data_root = Path(os.path.expanduser(args.data_root))

    mapping = fetch_schema_mapping(args.server)
    node_types = mapping.get("node_types", [])
    edge_types = edge_types_from_mapping(mapping)

    if args.dry_run:
        print("Dry-run schema check:")
        for table_name in ["users", "posts", "comments", "likes", "follows"]:
            path, columns = describe_parquet_table(data_root, table_name)
            print(f"  {table_name}: path={path}")
            print(f"    columns={columns}")
        print(f"  node_types={node_types}")
        print(f"  edge_types={len(edge_types)}")
        return

    # Keep full users table to ensure train fkeys map to valid user indices.
    users_df = load_parquet_table(data_root, "users", row_limit=None)
    posts_df = load_parquet_table(data_root, "posts", row_limit=None)

    user_key = infer_col(list(users_df.columns), ["user_id", "id"])
    post_user_key = infer_col(list(posts_df.columns), ["user_id", "author_id"])
    post_time_col = infer_time_col(posts_df, "posts")

    train_df = create_user_post_train_table(
        users_df=users_df,
        posts_df=posts_df,
        user_key=user_key,
        post_user_key=post_user_key,
        post_time_col=post_time_col,
        num_windows=args.num_windows,
        window_seconds=args.window_seconds,
    )

    df_dict: Dict[str, pd.DataFrame] = {
        "users": users_df,
        "posts": posts_df,
        "train_table": train_df,
    }

    encoders = build_encoders(df_dict, user_key=user_key)
    time_col_dict = build_time_col_dict(df_dict)

    feature_store = InMemoryFeatureStore.from_encoders(
        df_dict=df_dict,
        encoders=encoders,
        pkey_name_dict={"users": user_key},
        time_col_dict=time_col_dict,
        end_time_col_dict={},
        train_table_name="train_table",
        train_fkeys=[(user_key, "users")],
        train_mask_names=("train_mask", "val_mask", "test_mask"),
    )

    graph_store = DiskgraphGraphStore(
        server_address=args.server,
        edge_types=edge_types,
        node_types=node_types,
    )

    trainer = kumo.Trainer(
        task=TaskConfig(
            task_type=TaskType.BINARY_CLASSIFICATION,
            name="social_sampling_50m_diskgraph_e2e",
            metrics=["auroc", "auprc"],
        ),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=128, num_mp_layers=2),
        head=NodePredHead(),
        sampler_kwargs=dict(
            num_neighbors=[args.num_neighbors, args.num_neighbors],
            temporal_strategy="uniform",
        ),
        loader_kwargs=dict(batch_size=args.batch_size),
        max_epochs=args.epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
    ).init()

    assert_edge_type_alignment(trainer=trainer, input_node_type="users")

    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    main()
