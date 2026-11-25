# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tfm import Binary, Decoder, EagerAttention, Flat, LastTokenPool, NextEvent, NoPE, TrxFM

model = TrxFM(
    granularity=Flat(Decoder(depth=2, mask="causal", hidden_size=64, num_heads=4, vocab_size=256)),
    position=NoPE(),
    attention=EagerAttention(),
    objective=NextEvent(),
    readout=LastTokenPool(),
    heads={"fraud": Binary()},
)
print(model.to_config())
