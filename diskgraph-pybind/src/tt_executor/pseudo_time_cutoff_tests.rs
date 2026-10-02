// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn omitted_cutoffs_follow_manifest_stages_for_generated_and_custom_tables() {
    pyo3::Python::initialize();
    let dir = tempfile::tempdir().unwrap();
    let (index, _) = build_index(dir.path());
    generated_tt(&index, "generated", split("RandomLinkSplit"));
    custom_tt(
        dir.path(),
        &index,
        "custom",
        &[(1, vec![10], 0), (2, vec![11], 1), (3, vec![12], 2)],
        manual_split(),
    );
    for tt_id in ["generated", "custom"] {
        let reader = GraphIndexReader::open_with_tt(&index, tt_id).unwrap();
        for (table, time) in STAGES.iter().zip([1, 1, 2]) {
            let cutoff = PseudoTimeSeed::for_stage(&reader, table, None, false)
                .unwrap()
                .unwrap();
            assert_eq!(cutoff.filter().seed_time, time);
            let cutoff = PseudoTimeSeed::for_stage(&reader, table, Some(0), false)
                .unwrap()
                .unwrap();
            assert_eq!(cutoff.filter().seed_time, 0);
            assert!(PseudoTimeSeed::for_stage(&reader, table, None, true)
                .unwrap()
                .is_none());
        }
        assert!(PseudoTimeSeed::for_stage(&reader, "unknown", None, false).is_err());
        assert!(PseudoTimeSeed::for_stage(&reader, STAGES[0], Some(1), true).is_err());
    }
}
