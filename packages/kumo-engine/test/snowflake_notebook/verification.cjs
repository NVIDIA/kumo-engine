// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
class VerificationError extends Error {}

function requireThat(value, message) {
  if (!value) throw new VerificationError(message);
}

module.exports = { VerificationError, requireThat };
