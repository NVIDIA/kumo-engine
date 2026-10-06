// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Application entry point, referenced by ui/index.html.
 *
 * Import order here is load-bearing: the Trusted-Types policy bootstrap
 * must evaluate before any other application module so the
 * "sanitizeScriptURL" policy exists before any consumer runs. It was
 * moved out of an inline <script> in index.html so the delivery CSP
 * script-src needs no inline hash (kumo-ai/kumo#30121).
 *
 * Do not add imports above the policy bootstrap. Both imports below are
 * side-effect-only, and simple-import-sort keeps side-effect imports in
 * their written order. scripts/check-built-index-csp.mjs enforces the
 * resulting ordering on the built entry chunk during `npm run build`.
 */
import "utils/sanitizeScriptUrlPolicy";
import "./main";
