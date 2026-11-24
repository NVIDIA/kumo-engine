# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tfm CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from tfm.cli.recipe import COMMANDS, resolve_config, run_command


def _prepare_tabformer(root: Path, archive: Path | None) -> int:
    import json

    from tfm.data.adapters.tabformer_splits import (
        CSV_NAME,
        extract_tabformer,
        write_tabformer_splits,
    )

    csv_path = root / "raw" / CSV_NAME
    if not csv_path.exists():
        csv_path = extract_tabformer(archive or root / "transactions.tgz", root)
    manifest = write_tabformer_splits(csv_path, root / "temporal_split")
    print(json.dumps(manifest.as_dict(), indent=2))
    print(f"export TFM_TABFORMER_ROOT={root.resolve()}")
    return 0


def _cache_tokens(config_path: Path, workers: int, rows_per_block: int) -> int:
    import json

    from tfm.config import load_recipe_config
    from tfm.runtime.data import tokenizer_for
    from tfm.runtime.token_cache import build_recipe_cache

    config = load_recipe_config(config_path)
    manifest = build_recipe_cache(
        config, tokenizer_for(config), workers=workers, rows_per_block=rows_per_block
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "field_kind"}, indent=2))
    return 0


def _migrate_config(source: Path, out: Path | None) -> int:
    import sys

    import yaml

    from tfm.config_migration import migrate_config

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    migration = migrate_config(raw)
    migration.validated()
    text = (
        f"# Migrated from {source.name} (2.0.0a1 keys) by `tfm migrate-config`;\n"
        "# the original's comments are not carried over.\n"
        + yaml.safe_dump(migration.config, sort_keys=False)
    )
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")
    for note in migration.notes or ["nothing to migrate"]:
        print(note, file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tfm", description="Transaction foundation model reference-architecture kit"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        stage_help = (
            "Train, embed, evaluate, and report on config.smoke.yaml (CPU)."
            if name == "smoke"
            else f"Run the {name} stage and the stages before it."
        )
        p = sub.add_parser(name, help=stage_help)
        p.add_argument(
            "--recipe",
            default=None,
            help="Recipe slug from recipes/catalog.yaml, e.g. blueprint_v1.",
        )
        p.add_argument(
            "--config",
            default=None,
            help=(
                "A config path, or with --recipe a variant name: `smoke` reads "
                "config.smoke.yaml. Defaults to config.yaml (smoke: config.smoke.yaml)."
            ),
        )
        p.add_argument("--run-dir", type=Path, default=None)
        p.add_argument(
            "--device",
            default=None,
            help="Override runtime.device: cpu, cuda, cuda:N, or auto.",
        )
        p.add_argument(
            "--reuse-checkpoint",
            action="store_true",
            help="With --run-dir holding model.pt, skip training and load it instead.",
        )
    tabformer = sub.add_parser(
        "prepare-tabformer",
        help="Extract IBM's transactions.tgz and write the temporal_split/ parquet files.",
    )
    tabformer.add_argument("--root", type=Path, required=True, help="TabFormer root to write.")
    tabformer.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="transactions.tgz; defaults to <root>/transactions.tgz. Skipped if the CSV exists.",
    )
    cache = sub.add_parser(
        "cache-tokens",
        help="Tokenize a book once into data.token_cache: a pragmatiq book (key_value_time) "
        "or TabFormer's train split (transaction_field_tokenizer) (ADR 0016).",
    )
    cache.add_argument("--recipe", default=None, help="Recipe slug, e.g. blueprint_v2.")
    cache.add_argument("--config", default=None, help="A config path, or a variant of --recipe.")
    cache.add_argument("--workers", type=int, default=8, help="Worker processes (1 runs inline).")
    cache.add_argument(
        "--rows-per-block",
        type=int,
        default=1_000_000,
        help="Book rows per worker task; each worker's peak memory grows with it.",
    )
    migrate = sub.add_parser(
        "migrate-config",
        help="Rewrite a 2.0.0a1 config (such as a checkpoint's config.yaml) into the "
        "current keys, and check that it loads.",
    )
    migrate.add_argument("config", type=Path, help="The 2.0.0a1 config to read.")
    migrate.add_argument(
        "--out", type=Path, default=None, help="Where to write it; defaults to stdout."
    )
    args = parser.parse_args(argv)
    if args.command == "migrate-config":
        return _migrate_config(args.config, args.out)
    if args.command == "prepare-tabformer":
        return _prepare_tabformer(args.root, args.archive)
    if args.command == "cache-tokens":
        config_path = resolve_config(args.recipe, args.config, command="train")
        if config_path is None:
            parser.error("cache-tokens needs --recipe or --config")
        return _cache_tokens(config_path, args.workers, args.rows_per_block)
    run_command(
        args.command,
        config_path=resolve_config(args.recipe, args.config, command=args.command),
        run_dir=args.run_dir,
        device=args.device,
        reuse_checkpoint=args.reuse_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
