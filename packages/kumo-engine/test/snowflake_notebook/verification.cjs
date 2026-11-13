// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
class VerificationError extends Error {}

function requireThat(value, message) {
  if (!value) throw new VerificationError(message);
}

async function until(
  page,
  predicate,
  timeout,
  message = "Rendered UI did not reach the required state before its deadline",
) {
  const deadline = Date.now() + timeout;
  do {
    if (await predicate()) return;
    await page.waitForTimeout(250);
  } while (Date.now() < deadline);
  throw new VerificationError(message);
}

module.exports = { VerificationError, requireThat, until };
