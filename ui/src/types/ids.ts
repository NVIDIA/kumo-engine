// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Primitive resource-ID aliases shared across type modules.
//
// Kept in a dependency-free leaf module so mutually-referencing type files
// (e.g. graphs.ts ⇄ tables.ts) can share ID types without importing each
// other. Do not add runtime code or imports here — keeping this a leaf is what
// prevents circular dependencies (enforced by the `lint:circular` madge gate).
export type GraphID = string;
