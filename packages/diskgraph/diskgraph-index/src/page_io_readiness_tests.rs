// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Known error-path regression, selected explicitly during readiness review.

use super::{DirectCache, PageKey, CACHE_LOADING_BIT};
use std::sync::atomic::Ordering;

#[test]
#[ignore = "known PR424 blocker: failed-plan Drop cancels a replacement reservation"]
fn failed_plan_preserves_replacement() {
    let cache = Box::leak(Box::new(DirectCache::new(1, 1).unwrap()));
    let failed = cache.plan_read(101, &[0]);
    let slot = failed.reads[0].slot.unwrap();
    failed.fail("injected read failure");
    let replacement = cache.plan_read(101, &[0]);
    assert_eq!(replacement.reads[0].slot, Some(slot));
    drop(failed);
    assert_eq!(
        cache.tags[slot].load(Ordering::Acquire),
        PageKey {
            file_id: 101,
            page_number: 0,
        }
        .packed()
            | CACHE_LOADING_BIT,
        "dropping the failed plan canceled a new owner's reservation"
    );
    drop(replacement);
}
