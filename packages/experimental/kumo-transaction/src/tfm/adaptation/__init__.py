# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slot 4, adaptation: how a pretrained embedding is used for a task (building-block contracts).

Shared by every stack; nothing here knows which tokenizer or history model
produced the embedding. Import from the submodules:

- `tfm.adaptation.batching`: row slicing for any batch layout.
- `tfm.adaptation.probe`: frozen probe arms (raw, embedding, combined).
- `tfm.adaptation.finetune`: LoRA or full fine-tuning with a linear or fused head.
- `tfm.adaptation.fusion`: PLR numeric embeddings and the DCNv2 joint-fusion head.

The package imports nothing itself, so low-level modules such as
`tfm.inference` can use `batching` without pulling in training code.
"""
