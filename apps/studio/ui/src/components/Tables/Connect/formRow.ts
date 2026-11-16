// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Two-column form layout shared by the Add Table fields: a fixed label column
 * beside the control, so this page lines up with Add connector and Add
 * identity, which declare the same measure locally. The column width lives in
 * `--form-label-col`. Promote to the catalog when a second feature needs it.
 */

export const formLabelClassName =
    "text-sm font-medium leading-5 text-title-secondary";

/**
 * TextInput emits its label, input and error as bare siblings, so each lands
 * directly in this grid; the error is pushed to column 2 to sit under the
 * field rather than under the label.
 */
export const formRowClassName =
    "grid grid-cols-[var(--form-label-col)_1fr] items-center gap-2 [&>[id^=error]]:col-start-2";

/** For rows whose control stacks extra content below it, so the label pins to the top. */
export const formRowTopAlignedClassName =
    "grid grid-cols-[var(--form-label-col)_1fr] items-start gap-2";
