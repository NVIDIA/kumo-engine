// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn ingest_rebuilds_fact_visibility_on_destination_graph() {
    pyo3::Python::initialize();
    for custom in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (source, source_orders) = build_index(&dir.path().join("source"));
        let (destination, destination_orders) =
            build_index_order(&dir.path().join("destination"), true);
        assert_eq!(source_orders, destination_orders);
        let staging = if custom {
            let path = dir.path().join("custom.parquet");
            let mut targets = ListBuilder::new(Int64Builder::new());
            for user in 0..USERS {
                for (_, item) in source_orders.iter().filter(|(u, _)| *u == user) {
                    targets.values().append_value(*item);
                }
                targets.append(true);
            }
            write_parquet(
                &path,
                vec![
                    ("ENTITY1", Arc::new(Int64Array::from_iter_values(0..USERS))),
                    (
                        "TIMESTAMP",
                        Arc::new(TimestampNanosecondArray::from(vec![0; USERS as usize])),
                    ),
                    ("TARGET1", Arc::new(targets.finish())),
                ],
            );
            custom_artifacts(&path, &source, "tt", split("RandomLinkSplit")).unwrap()
        } else {
            pyo3::Python::attach(|py| {
                let plan =
                    serde_json::to_string(&generated_plan(split("RandomLinkSplit"))).unwrap();
                let result =
                    super::super::execute_tt_gen(py, &plan, source.to_str().unwrap(), "tt")
                        .unwrap();
                PathBuf::from(
                    result
                        .bind(py)
                        .get_item("staging_dir")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                )
            })
        };
        let source_bytes = std::fs::read(staging.join("fact_pseudo_time.bin")).unwrap();
        pyo3::Python::attach(|py| {
            super::super::execute_tt_ingest(
                py,
                staging.to_str().unwrap(),
                destination.to_str().unwrap(),
                "tt",
                None,
            )
            .unwrap();
        });
        assert_eq!(
            source_bytes,
            std::fs::read(staging.join("fact_pseudo_time.bin")).unwrap()
        );
        assert_ne!(
            source_bytes,
            std::fs::read(destination.join("train_tables/tt/fact_pseudo_time.bin")).unwrap()
        );
        pyo3::Python::attach(|py| {
            super::super::execute_tt_ingest(
                py,
                staging.to_str().unwrap(),
                source.to_str().unwrap(),
                "tt",
                None,
            )
            .unwrap();
        });
        let source_index = Sampled::open(&source, "tt");
        let destination_index = Sampled::open(&destination, "tt");
        assert!(source_index.order_of_node.iter().any(|(node, row)| {
            destination_index
                .order_of_node
                .get(node)
                .is_some_and(|other| row != other)
        }));
        assert_k2_visibility(&destination, &destination_orders);
        let spec_path = staging.join(super::super::TT_INGEST_SPEC_FILENAME);
        let mut spec: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&spec_path).unwrap()).unwrap();
        spec.as_object_mut().unwrap().remove("fact_pseudo_time");
        std::fs::write(spec_path, serde_json::to_vec(&spec).unwrap()).unwrap();
        pyo3::Python::attach(|py| {
            let error = super::super::execute_tt_ingest(
                py,
                staging.to_str().unwrap(),
                destination.to_str().unwrap(),
                "missing_spec",
                None,
            )
            .unwrap_err();
            assert!(error.to_string().contains("reconstruction spec"), "{error}");
        });
        assert!(!destination.join("train_tables/missing_spec").exists());
    }
}
