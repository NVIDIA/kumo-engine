// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Ambient declarations for eslint plugins that don't ship type definitions.
// Lets eslint.config.ts import them without a per-line @ts-expect-error,
// which would become an "unused expect-error" lint failure if these
// packages ever publish types upstream.
//
// @typescript-eslint/parser ships real types at dist/index.d.ts, but the
// project's `moduleResolution: node` (classic) doesn't follow the `exports`
// field in package.json. The shim avoids changing moduleResolution for the
// whole project just to satisfy this single tooling file.
declare module "eslint-plugin-jsx-a11y";
declare module "@typescript-eslint/parser";
