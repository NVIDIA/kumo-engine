# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the two archives the Blueprint v2 lab notebooks download (`tfm.runtime.lab`).

    uv run python scripts/package_lab_assets.py \
        --run-dir outputs/ablations/blueprint_v2_final_d \
        --tokenizer ~/data/tfm_cache/synth_100k_gen2_m31_seed0_lab_hkey/tokenizer.json \
        --book ~/data/pragmatiq/data/synth_100k_gen2_m31_seed0 \
        --out dist/lab

writes `blueprint_v2_lab_tuned/` (model.pt, tokenizer.json, config.yaml,
manifests), the two `.tar.gz` archives, and prints each archive's SHA-256 for
`LAB_BOOK` / `LAB_CHECKPOINT`. The extracted checkpoint directory also works
directly as `TFM_V2_CHECKPOINT`. Only synthetic data may be packaged.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from tfm.runtime.common import repo_root
from tfm.runtime.lab import LAB_BOOK, LAB_CHECKPOINT, LAB_CONFIG, sha256_file

RUN_FILES = ("model.pt", "checkpoint_manifest.json", "model.manifest.json", "train_stats.json")


def _archive(source: Path, folder: str, out: Path) -> Path:
    archive = out / f"{folder}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=folder)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / LAB_CONFIG,
    )
    parser.add_argument("--book", type=Path, help="pragmatiq book directory to archive too")
    parser.add_argument("--out", type=Path, default=Path("dist/lab"))
    args = parser.parse_args()

    bundle = args.out / LAB_CHECKPOINT.folder
    bundle.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        source = args.run_dir / name
        if source.exists():
            shutil.copy2(source, bundle / name)
    if not (bundle / "model.pt").exists():
        raise FileNotFoundError(f"{args.run_dir} has no model.pt")
    shutil.copy2(args.tokenizer.expanduser(), bundle / "tokenizer.json")
    shutil.copy2(args.config, bundle / "config.yaml")
    archives = [_archive(bundle, LAB_CHECKPOINT.folder, args.out)]
    if args.book is not None:
        archives.append(_archive(args.book.expanduser(), LAB_BOOK.folder, args.out))
    for archive in archives:
        print(f"{archive}  sha256={sha256_file(archive)}")


if __name__ == "__main__":
    main()
