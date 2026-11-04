// Local patch: replace maligned with built-in byte alignment types.
/*
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

mod aligned_cursor;
pub use aligned_cursor::AlignedCursor;
mod alignment;
pub use alignment::{Alignment, A16, A64};
