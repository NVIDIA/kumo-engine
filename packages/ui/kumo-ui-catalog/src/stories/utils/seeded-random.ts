// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Returns a deterministic pseudo-random number generator seeded at `initialSeed`.
 * Each call to the returned function advances the sequence by one step.
 * Use this in story render functions instead of Math.random() so that
 * generated mock data is identical on every run, keeping visual regression
 * screenshots stable.
 *
 * Usage:
 *   // Module-level data (generated once):
 *   const rng = createSeededRandom();
 *   const data = Array.from({ length: 20 }, () => rng());
 *
 *   // Render-function data (must be identical each render):
 *   render: () => {
 *     const rng = createSeededRandom(); // fresh instance every render
 *     const rows = Array.from({ length: 100 }, () => Math.floor(rng() * 200));
 *     return <MyTable data={rows} />;
 *   }
 */
export function createSeededRandom(initialSeed = 12345): () => number {
    let seed = initialSeed;
    return () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
    };
}
