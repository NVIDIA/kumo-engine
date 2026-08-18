    use super::*;

    fn publication_paths(dir: &tempfile::TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
        let train_tables = dir.path().join("index").join("train_tables");
        std::fs::create_dir_all(&train_tables).unwrap();
        (train_tables.join("published"), train_tables.join(".published.attempt"))
    }

    fn write_complete_tt_extension(root: &Path, target: &Path, generation: &[u8]) {
        let graph_index = target.parent().unwrap().parent().unwrap();
        let stage_part = root.join("stage").join("part-00000.parquet");
        std::fs::create_dir_all(stage_part.parent().unwrap()).unwrap();
        std::fs::write(&stage_part, b"parquet-part").unwrap();
        std::fs::write(root.join("generation"), generation).unwrap();
        let manifest = json!({
            "schema_version": "diskgraph.tt_execute_result.v1",
            "tt_id": target.file_name().unwrap().to_str().unwrap(),
            "graph_index_path": graph_index,
            "tt_extension_path": target,
            "diskgraph_manifest_path": target.join("diskgraph_manifest.json"),
            "task_dataset": {
                "schema_version": "diskgraph.task_dataset.v1",
                "graph_index_path": graph_index,
                "tt_extension_path": target,
                "is_temporal": false,
                "rhs_node_type": null,
                "input_row_count": 1,
                "stages": {
                    "train": {
                        "table_name": "stage",
                        "index_path": target,
                        "row_count": 1,
                        "parts": ["stage/part-00000.parquet"],
                        "seed_time": null,
                    }
                },
                "columns": {
                    "primary_entity": PRIMARY_ENTITY_COL,
                    "primary_target": PRIMARY_TARGET_COL,
                    "timestamp": TIMESTAMP_COL,
                    "weight": null,
                },
            },
        });
        std::fs::write(
            root.join("diskgraph_manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn test_build_prediction_table_edges_bundle_value_basic() {
        let root = std::env::temp_dir().join(format!("dg_pred_edges_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();

        let fkey = root.join("custom_prediction_table_fkey_cols.parquet");
        std::fs::create_dir_all(&fkey).unwrap();
        std::fs::write(fkey.join("part-0000.parquet"), b"fkey").unwrap();

        let edge = root
            .join("users__rev_user_id__custom_prediction_table")
            .join("edge.parquet");
        std::fs::create_dir_all(&edge).unwrap();
        std::fs::write(edge.join("part-0000.parquet"), b"edge").unwrap();

        let payload = build_prediction_table_edges_bundle_value(
            &root,
            "custom_prediction_table",
            false,
            Some("mask_not_required"),
            None,
        )
        .unwrap();

        assert_eq!(
            payload.get("format").and_then(|v| v.as_str()),
            Some("tt_pred_edge_bundle_v1")
        );
        assert_eq!(
            payload.get("kind").and_then(|v| v.as_str()),
            Some("tt_prediction_table_edges")
        );
        assert_eq!(
            payload
                .get("mask")
                .and_then(|v| v.get("present"))
                .and_then(|v| v.as_bool()),
            Some(false)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_read_tt_parquet_batches_roundtrip() {
        use arrow_array::{Int64Array, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc as StdArc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.parquet");
        let schema = StdArc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(Int64Array::from(vec![1i64, 2, 3])),
                StdArc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .unwrap();
        let f = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        // Construct iterator (lazy) and immediately drop. Should not panic.
        let mut it = TTParquetBatchIterator {
            parquet_path: path.to_string_lossy().into_owned(),
            columns: None,
            reader: None,
            schema: None,
        };
        it.ensure_reader().unwrap();
        // After ensure_reader the schema is populated.
        assert!(it.schema.is_some());
        let s = it.schema.as_ref().unwrap();
        assert_eq!(s.fields().len(), 2);
        assert_eq!(s.field(0).name(), "a");
        assert_eq!(s.field(1).name(), "b");
    }

    #[test]
    fn test_read_tt_parquet_batches_column_projection() {
        use arrow_array::{Int64Array, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc as StdArc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.parquet");
        let schema = StdArc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                StdArc::new(Int64Array::from(vec![1i64, 2])),
                StdArc::new(StringArray::from(vec!["x", "y"])),
                StdArc::new(Int64Array::from(vec![10i64, 20])),
            ],
        )
        .unwrap();
        let f = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let mut it = TTParquetBatchIterator {
            parquet_path: path.to_string_lossy().into_owned(),
            columns: Some(vec!["a".to_string(), "c".to_string()]),
            reader: None,
            schema: None,
        };
        it.ensure_reader().unwrap();
        // Iterate via the underlying reader to confirm projection works.
        let r = it.reader.as_mut().unwrap();
        let batch = r.next().unwrap().unwrap();
        let cols = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>();
        // The reader returns ALL root columns (projection is metadata-pushdown);
        // the projection mask drops them at the leaf level. Just check that
        // the requested columns are present.
        assert!(cols.contains(&"a".to_string()));
        assert!(cols.contains(&"c".to_string()));
    }

    #[test]
    fn test_read_tt_parquet_batches_missing_column_errors() {
        // Constructing a `DiskgraphDataError` requires the Python interpreter,
        // so we test the underlying behavior by initializing the interpreter
        // here. The pyo3 fixture handles this via `Python::attach` —
        // but that requires `auto-initialize`. Use `Python::initialize`
        // to spin up the interpreter just for this test.
        use arrow_array::Int64Array;
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc as StdArc;

        pyo3::Python::initialize();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.parquet");
        let schema = StdArc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![StdArc::new(Int64Array::from(vec![1i64]))],
        )
        .unwrap();
        let f = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let mut it = TTParquetBatchIterator {
            parquet_path: path.to_string_lossy().into_owned(),
            columns: Some(vec!["does_not_exist".to_string()]),
            reader: None,
            schema: None,
        };
        let result = pyo3::Python::attach(|_py| it.ensure_reader());
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("does_not_exist"));
    }

    // ─── split-optional / stage-name-flexible gen+ingest ───────────────────
    //
    // These exercise the predict-unification path: a split-less plan must
    // produce `tt.parquet` + `tt_ingest_spec.json` (NO `split.bin`), and the
    // ingest phase must publish a SINGLE stage under a caller-chosen name
    // (default "predict") using the SAME FK-free resolution path as train.

    use diskgraph_core::tt_builder::{
        AggFunc, EdgeRef, EntitySpec, Op, PredictionType, RefResolveKind, RefResolveSpec, SplitSpec,
        Stage, TemporalSpec,
    };
    use diskgraph_core::time::{Abs, Seconds};
    use diskgraph_ingest::graph_config::{
        ColumnSpec, DataSource, FKeySpec, GraphConfig, IngestedType, TableSpec, TimeSpec,
    };
    use diskgraph_ingest::io::Dest;
    use std::sync::Arc as StdArc2;

    fn t_mk_col(name: &str, dtype: IngestedType) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            source_dtype: None,
            ingested_dtype: dtype,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }
    }

    fn t_write_parquet(path: &Path, batch: &RecordBatch) {
        use parquet::arrow::ArrowWriter;
        let file = File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
    }

    fn t_write_stage_rows(path: &Path, rows: usize) {
        use arrow_array::Int64Array;
        use arrow_schema::{DataType, Field, Schema};

        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let schema = StdArc2::new(Schema::new(vec![Field::new(
            "ENTITY1",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![StdArc2::new(Int64Array::from_iter_values(
                (0..rows).map(|value| value as i64),
            ))],
        )
        .unwrap();
        t_write_parquet(path, &batch);
    }

    fn t_stage_info(
        part: &str,
        row_count: u64,
        artifact_inventory: Option<
            diskgraph_train_table::ttgen_executor::StageArtifactInventory,
        >,
    ) -> StageGenInfo {
        StageGenInfo {
            key: "train".to_string(),
            table_name: "train_seeds".to_string(),
            parts: vec![part.to_string()],
            row_count,
            seed_time: None,
            artifact_inventory,
        }
    }

    fn t_stage_inventory(
        root: &Path,
        part: &str,
        row_count: u64,
    ) -> diskgraph_train_table::ttgen_executor::StageArtifactInventory {
        use diskgraph_train_table::ttgen_executor::{
            StageArtifactInventory, StagePartArtifact,
        };

        StageArtifactInventory {
            schema_version: STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION.to_string(),
            parts: vec![StagePartArtifact {
                path: part.to_string(),
                size_bytes: std::fs::metadata(root.join(part)).unwrap().len(),
                row_count,
            }],
        }
    }

    fn t_write_fixed_staging_artifacts(root: &Path) {
        std::fs::create_dir_all(root).unwrap();
        for name in [TT_INGEST_SPEC_FILENAME, "tt.parquet", "manifest.json"] {
            std::fs::write(root.join(name), b"artifact").unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn test_staging_boundary_rejects_symlink_root_and_fixed_artifacts() {
        use std::os::unix::fs::symlink;

        let parent = tempfile::tempdir().unwrap();
        let real_root = parent.path().join("real-root");
        t_write_fixed_staging_artifacts(&real_root);
        let linked_root = parent.path().join("linked-root");
        symlink(&real_root, &linked_root).unwrap();
        let error = validate_staging_artifact_types(&linked_root).unwrap_err();
        assert!(error.to_string().contains("non-symlink directory"));

        for artifact in [TT_INGEST_SPEC_FILENAME, "tt.parquet", "manifest.json"] {
            let root = parent.path().join(format!("root-{artifact}"));
            t_write_fixed_staging_artifacts(&root);
            let external = parent.path().join(format!("external-{artifact}"));
            std::fs::rename(root.join(artifact), &external).unwrap();
            symlink(&external, root.join(artifact)).unwrap();
            let error = validate_staging_artifact_types(&root).unwrap_err();
            assert!(error.to_string().contains("symlink"));
        }

        let nonregular_root = parent.path().join("root-nonregular");
        t_write_fixed_staging_artifacts(&nonregular_root);
        std::fs::remove_file(nonregular_root.join("manifest.json")).unwrap();
        std::fs::create_dir(nonregular_root.join("manifest.json")).unwrap();
        let error = validate_staging_artifact_types(&nonregular_root).unwrap_err();
        assert!(error.to_string().contains("regular file"));
    }

    #[cfg(unix)]
    #[test]
    fn test_resolve_ingest_stages_rejects_symlink_part_and_parent_escape() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let part = "train_seeds/part-00000.parquet";
        let part_path = root.path().join(part);
        t_write_stage_rows(&part_path, 1);
        let inventory = t_stage_inventory(root.path(), part, 1);
        let external = root.path().join("external-part.parquet");
        std::fs::rename(&part_path, &external).unwrap();
        symlink(&external, &part_path).unwrap();
        let info = t_stage_info(part, 1, Some(inventory));
        let error = resolve_ingest_stages(
            root.path(),
            &root.path().join("tt.parquet"),
            "predict",
            Some(std::slice::from_ref(&info)),
        )
        .unwrap_err();
        assert!(error.to_string().contains("symlink"));

        std::fs::remove_file(&part_path).unwrap();
        std::fs::remove_dir(root.path().join("train_seeds")).unwrap();
        let external_dir = root.path().join("external-stage");
        std::fs::create_dir(&external_dir).unwrap();
        t_write_stage_rows(&external_dir.join("part-00000.parquet"), 1);
        symlink(&external_dir, root.path().join("train_seeds")).unwrap();
        let error = resolve_ingest_stages(
            root.path(),
            &root.path().join("tt.parquet"),
            "predict",
            Some(&[info]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("symlink"));
    }

    #[test]
    fn test_v2_inventory_is_additive_and_requires_fleet_wide_consumer_rollout() {
        use diskgraph_train_table::ttgen_executor::{
            StageArtifactInventory, StagePartArtifact,
        };

        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct LegacyStageInfo {
            parts: Vec<String>,
            row_count: u64,
        }
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        struct LegacySpec {
            schema_version: String,
            split_stages: Option<Vec<LegacyStageInfo>>,
        }

        let mut spec = TtIngestSpec::from_plan("tt", &t_node_plan(true));
        spec.split_stages = Some(vec![t_stage_info(
            "train_seeds/part-00000.parquet",
            1,
            Some(StageArtifactInventory {
                schema_version: STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION.to_string(),
                parts: vec![StagePartArtifact {
                    path: "train_seeds/part-00000.parquet".to_string(),
                    size_bytes: 123,
                    row_count: 1,
                }],
            }),
        )]);
        let encoded = serde_json::to_vec(&spec).unwrap();

        // This mirrors the old serde shape: unknown additive inventory fields
        // are silently dropped. A new producer must therefore not be enabled
        // until every ingest consumer runs the v2 validator.
        let legacy: LegacySpec = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(legacy.schema_version, TT_INGEST_SPEC_SCHEMA_VERSION);
        let stage = &legacy.split_stages.unwrap()[0];
        assert_eq!(stage.parts, ["train_seeds/part-00000.parquet"]);
        assert_eq!(stage.row_count, 1);
    }

    #[test]
    fn test_resolve_ingest_stages_rejects_missing_artifact_inventory() {
        let root = tempfile::tempdir().unwrap();
        let part = "train_seeds/part-00000.parquet";
        t_write_stage_rows(&root.path().join(part), 1);
        let info = t_stage_info(part, 2, None);

        let error = resolve_ingest_stages(
            root.path(),
            &root.path().join("tt.parquet"),
            "predict",
            Some(&[info]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("missing artifact inventory"));
    }

    #[test]
    fn test_resolve_ingest_stages_enforces_v2_part_size() {
        use diskgraph_train_table::ttgen_executor::{
            StageArtifactInventory, StagePartArtifact,
        };

        let root = tempfile::tempdir().unwrap();
        let part = "train_seeds/part-00000.parquet";
        let path = root.path().join(part);
        t_write_stage_rows(&path, 1);
        let actual_size = std::fs::metadata(&path).unwrap().len();
        let info = t_stage_info(
            part,
            1,
            Some(StageArtifactInventory {
                schema_version: STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION.to_string(),
                parts: vec![StagePartArtifact {
                    path: part.to_string(),
                    size_bytes: actual_size + 1,
                    row_count: 1,
                }],
            }),
        );

        let error = resolve_ingest_stages(
            root.path(),
            &root.path().join("tt.parquet"),
            "predict",
            Some(&[info]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("size mismatch"));
    }

    #[test]
    fn test_resolve_ingest_stages_enforces_v2_part_row_count() {
        use diskgraph_train_table::ttgen_executor::{
            StageArtifactInventory, StagePartArtifact,
        };

        let root = tempfile::tempdir().unwrap();
        let part = "train_seeds/part-00000.parquet";
        let path = root.path().join(part);
        t_write_stage_rows(&path, 1);
        let info = t_stage_info(
            part,
            1,
            Some(StageArtifactInventory {
                schema_version: STAGE_ARTIFACT_INVENTORY_SCHEMA_VERSION.to_string(),
                parts: vec![StagePartArtifact {
                    path: part.to_string(),
                    size_bytes: std::fs::metadata(&path).unwrap().len(),
                    row_count: 2,
                }],
            }),
        );

        let error = resolve_ingest_stages(
            root.path(),
            &root.path().join("tt.parquet"),
            "predict",
            Some(&[info]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("row_count mismatch"));
    }

    #[test]
    fn test_ingest_spec_v2_requires_stage_artifact_inventory() {
        let mut spec = TtIngestSpec::from_plan("tt", &t_node_plan(true));
        spec.split_stages = Some(vec![t_stage_info(
            "train_seeds/part-00000.parquet",
            1,
            None,
        )]);

        let error = validate_ingest_spec(&spec).unwrap_err();
        assert!(error.to_string().contains("missing its artifact inventory"));
        spec.schema_version = "diskgraph.tt_ingest_spec.v1".to_string();
        let error = validate_ingest_spec(&spec).unwrap_err();
        assert!(error.to_string().contains("unsupported tt_ingest_spec"));

        let missing_version = serde_json::json!({
            "tt_id": "tt",
            "entity_table": "users",
            "split_stages": null,
        });
        assert!(serde_json::from_value::<TtIngestSpec>(missing_version).is_err());
    }

    /// Build a tiny `users`/`orders` graph (5 users, 2 orders each) so the
    /// `count_orders` op below resolves and `ENTITY1` values map to real
    /// `users` node-ids during resolution-mode ingest. Returns the index dir.
    fn t_build_graph(root: &Path) -> std::path::PathBuf {
        t_build_graph_with_users(root, &[0, 1, 2, 3, 4])
    }

    /// Like [`t_build_graph`] but with a caller-chosen `users` PK set —
    /// used to build a SECOND graph missing some users for the cross-graph
    /// TT-reuse orphan tests.
    fn t_build_graph_with_users(root: &Path, user_ids: &[i64]) -> std::path::PathBuf {
        use arrow_array::{Int64Array, TimestampNanosecondArray};
        use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};

        let data_dir = root.join("data");
        let index_dir = root.join("index");
        std::fs::create_dir_all(&data_dir).unwrap();

        let users_schema = StdArc2::new(Schema::new(vec![Field::new(
            "user_id",
            DataType::Int64,
            false,
        )]));
        let users_batch = RecordBatch::try_new(
            users_schema,
            vec![StdArc2::new(Int64Array::from(user_ids.to_vec()))],
        )
        .unwrap();
        t_write_parquet(&data_dir.join("users.parquet"), &users_batch);

        // orders: each user buys 2 orders at distinct timestamps.
        let mut oids = Vec::new();
        let mut ouser = Vec::new();
        let mut ots = Vec::new();
        let mut oid = 0i64;
        for &uid in user_ids {
            for j in 0..2i64 {
                oids.push(oid);
                ouser.push(uid);
                ots.push((1000 + uid * 500 + j * 200) * 1_000_000_000);
                oid += 1;
            }
        }
        let orders_schema = StdArc2::new(Schema::new(vec![
            Field::new("order_id", DataType::Int64, false),
            Field::new("user_id", DataType::Int64, false),
            Field::new(
                "ts",
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                false,
            ),
        ]));
        let orders_batch = RecordBatch::try_new(
            orders_schema,
            vec![
                StdArc2::new(Int64Array::from(oids)),
                StdArc2::new(Int64Array::from(ouser)),
                StdArc2::new(TimestampNanosecondArray::from(ots)),
            ],
        )
        .unwrap();
        t_write_parquet(&data_dir.join("orders.parquet"), &orders_batch);

        let orders_time = TimeSpec {
            table: "orders".to_string(),
            column: "ts".to_string(),
            bitwidth: Some(16),
            anchor_time: Some(0),
            unit: "s".to_string(),
        };
        let config = GraphConfig {
            dataset_name: "predict_unify_test".to_string(),
            tables: vec![
                TableSpec {
                    name: "users".to_string(),
                    columns: vec![t_mk_col("user_id", IngestedType::Int64)],
                    primary_key: Some("user_id".to_string()),
                    create_time: None,
                    end_time: None,
                    weight: None,
                    edge_table: None,
                    rowpack: None,
                    source: DataSource::Parquet {
                        paths: vec![data_dir
                            .join("users.parquet")
                            .to_string_lossy()
                            .to_string()],
                    },
                    source_size: None,
                    fkeys: vec![],
                    data_version: None,
                    kumo_connector_id: None,
                    kumo_source_table_name: None,
                    group_by_timestamp: false,
                },
                TableSpec {
                    name: "orders".to_string(),
                    columns: vec![
                        t_mk_col("order_id", IngestedType::Int64),
                        t_mk_col("user_id", IngestedType::Int64),
                        t_mk_col("ts", IngestedType::Timestamp),
                    ],
                    primary_key: Some("order_id".to_string()),
                    create_time: None,
                    end_time: None,
                    weight: None,
                    edge_table: None,
                    rowpack: None,
                    source: DataSource::Parquet {
                        paths: vec![data_dir
                            .join("orders.parquet")
                            .to_string_lossy()
                            .to_string()],
                    },
                    source_size: None,
                    fkeys: vec![FKeySpec {
                        src_column: "user_id".to_string(),
                        dst_table: "users".to_string(),
                        forward: None,
                        reverse: Some(diskgraph_ingest::graph_config::DirectionSpec {
                            time: Some(orders_time.clone()),
                            weight: None,
                        }),
                        ..Default::default()
                    }],
                    data_version: None,
                    kumo_connector_id: None,
                    kumo_source_table_name: None,
                    group_by_timestamp: false,
                },
            ],
            rowpack: None,
            num_shards: Some(1),
            edge_format: None,
            coercion_mode: Default::default(),
        };
        let config_path = data_dir.join("graph_config.json");
        std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
        diskgraph_ingest::ingest(
            &config_path,
            index_dir.to_str().unwrap(),
            Dest::Local,
            diskgraph_ingest::config::IngestOptions {
                parallel: false,
                ..Default::default()
            },
        )
        .unwrap();
        index_dir
    }

    /// Node-prediction plan over `users` counting orders, no split. Optionally
    /// carries an `lp_resolution` that resolves `ENTITY1` → `users` node-ids.
    fn t_node_plan(with_resolution: bool) -> DiskGraphTTPlan<Seconds> {
        DiskGraphTTPlan::<Seconds> {
            entity: EntitySpec::<Seconds> {
                table: "users".to_string(),
                column: "user_id".to_string(),
                output_time_col_name: None,
                default_time_value: None,
                time_col: None,
                valid_to_col: None,
                filter: None,
            },
            temporal: Some(TemporalSpec::<Seconds> {
                timestamps: vec![1000i64, 2000, 3000]
                    .into_iter()
                    .map(Abs::from_raw)
                    .collect(),
                forecast_length: 1,
                lag_timesteps: None,
            }),
            prediction_type: PredictionType::Node,
            is_prediction: false,
            split: None,
            stages: vec![Stage::<Seconds> {
                ops: vec![Op::Aggregate {
                    alias: "cnt".to_string(),
                    path: vec![EdgeRef {
                        src_table: "users".to_string(),
                        src_col: "user_id".to_string(),
                        dst_table: "orders".to_string(),
                        dst_col: "user_id".to_string(),
                    }],
                    func: AggFunc::Count,
                    target_col: None,
                    time_col: None,
                    window: Some((0, 5000)),
                    window_unit: Some(diskgraph_core::time::TimeUnit::Seconds),
                    row_filter: None,
                    limit: None,
                    target_entity_table: None,
                }],
                filter: None,
            }],
            target_stype: Some("numerical".to_string()),
            weight_col: None,
            fkey_mapping: None,
            lp_resolution: if with_resolution {
                Some(vec![RefResolveSpec {
                    src_col: PRIMARY_ENTITY_COL.to_string(),
                    dst_table: "users".to_string(),
                    kind: RefResolveKind::Scalar,
                    node_id_col: "entity_node_id".to_string(),
                }])
            } else {
                None
            },
            lp_rhs_is_temporal: false,
        }
    }

    fn t_time_range_split() -> SplitSpec<Seconds> {
        SplitSpec::<Seconds> {
            ratio: None,
            kind: Some("TimeRangeSplit".to_string()),
            rnd_seed: None,
            is_temporal: Some(true),
            offsets: None,
            offset_unit: None,
            unit: None,
            timeframe: None,
            anchor_start_offset: None,
            ranges: Some(vec![
                (Abs::from_raw(0), Abs::from_raw(1500)),
                (Abs::from_raw(1501), Abs::from_raw(2500)),
                (Abs::from_raw(2501), Abs::from_raw(4000)),
            ]),
            backtesting_timestamp: None,
            split_column: None,
            split_output_column: None,
            manual_train_values: None,
            manual_val_values: None,
            manual_test_values: None,
            month_offsets: None,
            flexible_boundaries: None,
            timeframe_step: None,
        }
    }

    /// Read the `stages` map out of a published `diskgraph_manifest.json`.
    fn read_manifest_stages(tt_extension_path: &str) -> serde_json::Map<String, Value> {
        let p = Path::new(tt_extension_path).join("diskgraph_manifest.json");
        let raw = std::fs::read_to_string(&p).unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        v["task_dataset"]["stages"]
            .as_object()
            .expect("stages object present")
            .clone()
    }

    /// Read the `task_dataset` object out of a published manifest (for asserting
    /// the top-level `is_temporal`/`rhs_node_type` fields).
    fn read_task_dataset(tt_extension_path: &str) -> serde_json::Map<String, Value> {
        let p = Path::new(tt_extension_path).join("diskgraph_manifest.json");
        let raw = std::fs::read_to_string(&p).unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        v["task_dataset"]
            .as_object()
            .expect("task_dataset object present")
            .clone()
    }

    // ─── stage-keyed self-describing manifest ───────────────────────────────
    //
    // The `task_dataset` manifest is self-describing so materialize can ingest
    // without re-deriving task metadata from the pquery: it carries `is_temporal`
    // + `rhs_node_type` and a per-stage `parts` list, derived from the plan at
    // gen time and round-tripped through `tt_ingest_spec.json`.

    #[test]
    fn test_tt_ingest_spec_derives_is_temporal_and_rhs_node_type() {
        // No split + scalar-only resolution → not temporal, no RHS node type.
        let mut plan = t_node_plan(true); // split=None, one Scalar entity resolve
        let spec = TtIngestSpec::from_plan("tt", &plan);
        assert!(!spec.is_temporal, "no split ⇒ is_temporal=false");
        assert_eq!(
            rhs_node_type_from_resolution(&spec.lp_resolution),
            None,
            "scalar-only resolution ⇒ no RHS node type"
        );

        // Temporal split + a List target resolution → temporal + RHS=dst_table.
        plan.split = Some(t_time_range_split()); // is_temporal = Some(true)
        plan.lp_resolution = Some(vec![
            RefResolveSpec {
                src_col: PRIMARY_ENTITY_COL.to_string(),
                dst_table: "users".to_string(),
                kind: RefResolveKind::Scalar,
                node_id_col: "entity_node_id".to_string(),
            },
            RefResolveSpec {
                src_col: PRIMARY_TARGET_COL.to_string(),
                dst_table: "orders".to_string(),
                kind: RefResolveKind::List,
                node_id_col: "target_node_ids".to_string(),
            },
        ]);
        let spec = TtIngestSpec::from_plan("tt", &plan);
        assert!(spec.is_temporal, "temporal split ⇒ is_temporal=true");
        assert_eq!(
            rhs_node_type_from_resolution(&spec.lp_resolution),
            Some("orders".to_string()),
            "RHS node type = the List-kind resolution's dst_table"
        );
    }

    /// Decode one stage-table node's rowpack columns back as JSON (mirrors the
    /// resolution_ingest.rs read-back: `edges_for_node` → `decode_node_data_json`).
    fn read_stage_node_json(
        tt_extension_path: &str,
        table_name: &str,
        node_id: u64,
    ) -> serde_json::Value {
        let ext = Path::new(tt_extension_path);
        let reader = diskgraph_index::graph::GraphIndexReader::open(
            &ext.join("index_metadata.json"),
            ext,
        )
        .expect("open produced extension");
        let table = reader.table(table_name).expect("stage table present");
        let view = table.edges_for_node(node_id).expect("edges_for_node");
        table
            .decode_node_data_json(&view)
            .expect("decode_node_data_json")
            .expect("node data present")
    }

    #[test]
    fn test_execute_tt_gen_no_split_writes_no_split_bin() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let plan = t_node_plan(false);
        let plan_json = serde_json::to_string(&plan).unwrap();

        let result = pyo3::Python::attach(|py| {
            execute_tt_gen(py, &plan_json, index.to_str().unwrap(), "ttpred").map(|d| {
                let d = d.bind(py);
                let staging: String =
                    d.get_item("staging_dir").unwrap().extract().unwrap();
                let parquet: String =
                    d.get_item("tt_parquet_path").unwrap().extract().unwrap();
                let row_count: u64 = d.get_item("row_count").unwrap().extract().unwrap();
                (staging, parquet, row_count)
            })
        })
        .unwrap();

        let (staging, parquet, row_count) = result;
        // TT-NG never writes split.bin (gone entirely from the artifact set).
        assert!(
            !Path::new(&staging).join("split.bin").exists(),
            "split.bin must not be written"
        );
        // tt.parquet + tt_ingest_spec.json present.
        assert!(Path::new(&parquet).exists());
        assert!(Path::new(&staging).join(TT_INGEST_SPEC_FILENAME).exists());
        assert!(row_count > 0);
    }

    #[test]
    fn test_execute_tt_ingest_no_split_single_predict_stage() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        // Split-less plan WITH resolution → predict seed table topologically
        // identical to a resolution-mode train table (entity_node_id rowpack,
        // no FK edge).
        let plan = t_node_plan(true);
        let plan_json = serde_json::to_string(&plan).unwrap();

        let (staging, expected_rows) = pyo3::Python::attach(|py| {
            let d = execute_tt_gen(py, &plan_json, index.to_str().unwrap(), "ttpred").unwrap();
            let d = d.bind(py);
            let staging: String = d.get_item("staging_dir").unwrap().extract().unwrap();
            let n: u64 = d.get_item("row_count").unwrap().extract().unwrap();
            (staging, n)
        });

        let tt_ext = pyo3::Python::attach(|py| {
            // NEW dir-API: ingest takes the stage-set staging DIRECTORY.
            let d = execute_tt_ingest(
                py,
                &staging,
                index.to_str().unwrap(),
                "ttpred",
                Some("predict".to_string()),
            )
            .unwrap();
            let d = d.bind(py);
            // Same-graph ingest: the orphan skip counters must exist and be 0.
            let kf = d.get_item("key_failures").unwrap();
            let membership: u64 = kf
                .get_item("membership_failures")
                .unwrap()
                .extract()
                .unwrap();
            let lookup: u64 = kf.get_item("lookup_failures").unwrap().extract().unwrap();
            assert_eq!(
                (membership, lookup),
                (0, 0),
                "same-graph ingest must skip nothing"
            );
            let ext: String = d.get_item("tt_extension_path").unwrap().extract().unwrap();
            ext
        });

        // Manifest stages must be exactly {"predict": {..., row_count==N}}.
        let stages = read_manifest_stages(&tt_ext);
        assert_eq!(
            stages.keys().collect::<Vec<_>>(),
            vec!["predict"],
            "expected a single 'predict' stage, got {:?}",
            stages.keys().collect::<Vec<_>>()
        );
        let predict = &stages["predict"];
        assert_eq!(
            predict["row_count"].as_u64(),
            Some(expected_rows),
            "predict stage row_count mismatch"
        );
        assert_eq!(
            predict["table_name"].as_str(),
            Some("predict_seeds"),
            "predict stage table_name"
        );
        // Self-describing: per-stage `parts` under `<table>/part-NNNNN.parquet`
        // (single part on the predict path).
        assert_eq!(
            predict["parts"].as_array().map(|a| a
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<Vec<_>>()),
            Some(vec!["predict_seeds/part-00000.parquet".to_string()]),
            "predict stage must list its single part"
        );
        // The declared part must actually exist on disk in the extension.
        assert!(
            Path::new(&tt_ext)
                .join("predict_seeds/part-00000.parquet")
                .exists(),
            "predict stage part must be published on disk"
        );
        // Self-describing task-level fields: non-temporal node plan, no RHS.
        let td = read_task_dataset(&tt_ext);
        assert_eq!(
            td["is_temporal"].as_bool(),
            Some(false),
            "split-less node plan ⇒ is_temporal=false"
        );
        assert!(
            td["rhs_node_type"].is_null(),
            "node plan (no List resolution) ⇒ rhs_node_type=null, got {:?}",
            td["rhs_node_type"]
        );

        // Resolution-mode: the predict seed table carries the resolved
        // `entity_node_id` rowpack column (same as a resolution-mode train
        // table) for node 0.
        let node0 = read_stage_node_json(&tt_ext, "predict_seeds", 0);
        assert!(
            node0.get("entity_node_id").and_then(|v| v.as_u64()).is_some(),
            "predict_seeds must carry resolved entity_node_id column, got {node0}"
        );

        // NO FK edge: the published metadata must have no edge_relation
        // touching `predict_seeds`, and every node must have zero edges.
        let meta: diskgraph_core::metadata::DatasetMetadata = serde_json::from_slice(
            &std::fs::read(Path::new(&tt_ext).join("index_metadata.json")).unwrap(),
        )
        .unwrap();
        let touching: Vec<_> = meta
            .edge_relations
            .iter()
            .filter(|r| r.src_type == "predict_seeds" || r.dst_type == "predict_seeds")
            .collect();
        assert!(
            touching.is_empty(),
            "resolution mode must emit NO edge relation touching predict_seeds; got {:?}",
            touching
                .iter()
                .map(|r| (&r.name, &r.src_type, &r.dst_type))
                .collect::<Vec<_>>()
        );
        let ext = Path::new(&tt_ext);
        let reader = diskgraph_index::graph::GraphIndexReader::open(
            &ext.join("index_metadata.json"),
            ext,
        )
        .unwrap();
        let table = reader.table("predict_seeds").unwrap();
        for node_id in 0..expected_rows {
            let view = table.edges_for_node(node_id).unwrap();
            let segs = view.relation_segments().expect("relation_segments");
            let total_edges: u64 = segs.iter().map(|s| s.count as u64).sum();
            assert_eq!(
                total_edges, 0,
                "resolution mode must emit ZERO edges for predict_seeds node {node_id}"
            );
        }
    }

    #[test]
    fn test_manifest_staging_failure_preserves_prior_final() {
        let dir = tempfile::tempdir().unwrap();
        let (published, attempt) = publication_paths(&dir);
        write_complete_tt_extension(&published, &published, b"old");
        std::fs::create_dir_all(&attempt).unwrap();
        std::fs::write(attempt.join("generation"), b"new").unwrap();
        let blocked_manifest = attempt.join("diskgraph_manifest.json");
        std::fs::create_dir(&blocked_manifest).unwrap();

        let write_error =
            write_unpublished_tt_manifest(&blocked_manifest, b"new-manifest").unwrap_err();
        assert_ne!(write_error.kind(), std::io::ErrorKind::NotFound);
        let error = publish_tt_extension(&attempt, &published).unwrap_err();
        assert!(error.to_string().contains("incomplete"));
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(is_complete_tt_extension_for(&published, &published));
        assert!(attempt.exists());
    }

    #[test]
    fn test_publish_tt_extension_rejects_target_mismatched_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let (published, attempt) = publication_paths(&dir);
        write_complete_tt_extension(&published, &published, b"old");
        write_complete_tt_extension(&attempt, &published, b"new");
        let manifest_path = attempt.join("diskgraph_manifest.json");
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["tt_id"] = json!("different-target");
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        let error = publish_tt_extension(&attempt, &published).unwrap_err();
        assert!(error.to_string().contains("target-mismatched"));
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(attempt.exists());
    }

    #[test]
    fn test_publish_tt_extension_moves_complete_attempt_with_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let (published, attempt) = publication_paths(&dir);
        write_complete_tt_extension(&published, &published, b"old");
        write_complete_tt_extension(&attempt, &published, b"new");

        publish_tt_extension(&attempt, &published).unwrap();

        assert!(!attempt.exists());
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"new");
        assert!(is_complete_tt_extension_for(&published, &published));
    }

    #[test]
    fn test_publish_tt_extension_never_exposes_incomplete_final() {
        let dir = tempfile::tempdir().unwrap();
        let (published, attempt) = publication_paths(&dir);
        write_complete_tt_extension(&published, &published, b"old");
        write_complete_tt_extension(&attempt, &published, b"new");

        publish_tt_extension_with_rename(&attempt, &published, |src, dst| {
            std::fs::rename(src, dst)?;
            assert!(
                !published.exists()
                    || is_complete_tt_extension_for(&published, &published),
                "a final path must be absent or contain its manifest"
            );
            Ok(())
        })
        .unwrap();
        assert!(is_complete_tt_extension_for(&published, &published));
    }

    #[test]
    fn test_publish_tt_extension_rename_failure_restores_prior_final() {
        let dir = tempfile::tempdir().unwrap();
        let (published, attempt) = publication_paths(&dir);
        write_complete_tt_extension(&published, &published, b"old");
        write_complete_tt_extension(&attempt, &published, b"new");
        let mut rename_count = 0;

        let error = publish_tt_extension_with_rename(&attempt, &published, |src, dst| {
            rename_count += 1;
            if rename_count == 2 {
                return Err(std::io::Error::other("injected publish failure"));
            }
            std::fs::rename(src, dst)
        })
        .unwrap_err();

        assert!(error.to_string().contains("injected publish failure"));
        assert_eq!(rename_count, 3, "park, failed publish, rollback");
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(is_complete_tt_extension_for(&published, &published));
        assert_eq!(std::fs::read(attempt.join("generation")).unwrap(), b"new");

        publish_tt_extension(&attempt, &published).unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"new");
        assert!(!attempt.exists());
    }

    #[test]
    fn test_publish_tt_extension_recovers_crash_after_parking_prior() {
        let dir = tempfile::tempdir().unwrap();
        let (published, attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        write_complete_tt_extension(&published, &published, b"old");
        write_complete_tt_extension(&attempt, &published, b"new");

        // Simulate a process crash immediately after final -> backup.
        std::fs::rename(&published, &backup).unwrap();
        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(!backup.exists());

        // Recovery is idempotent, and the same attempt can then retry safely.
        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();
        publish_tt_extension(&attempt, &published).unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"new");
        assert!(!backup.exists());
    }

    #[test]
    fn test_publish_tt_extension_recovers_crash_after_new_final() {
        let dir = tempfile::tempdir().unwrap();
        let (published, _attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        let retry = published.with_file_name(".published.retry");
        write_complete_tt_extension(&published, &published, b"new");
        write_complete_tt_extension(&backup, &published, b"old");

        // Simulate a crash immediately after attempt -> final and before the
        // now-stale prior generation was removed.
        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"new");
        assert!(!backup.exists());
        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"new");

        write_complete_tt_extension(&retry, &published, b"newer");
        publish_tt_extension(&retry, &published).unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"newer");
        assert!(!backup.exists());
    }

    #[test]
    fn test_publish_tt_extension_rolls_back_incomplete_final() {
        let dir = tempfile::tempdir().unwrap();
        let (published, _attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        std::fs::create_dir_all(&published).unwrap();
        std::fs::write(published.join("generation"), b"partial-new").unwrap();
        write_complete_tt_extension(&backup, &published, b"old");

        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(!backup.exists());
    }

    #[test]
    fn test_recovery_rolls_back_final_with_missing_declared_part() {
        let dir = tempfile::tempdir().unwrap();
        let (published, _attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        write_complete_tt_extension(&published, &published, b"new");
        write_complete_tt_extension(&backup, &published, b"old");
        std::fs::remove_file(published.join("stage/part-00000.parquet")).unwrap();

        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();

        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(is_complete_tt_extension_for(&published, &published));
        assert!(!backup.exists());
    }

    #[test]
    fn test_recovery_rolls_back_final_with_wrong_graph_path() {
        let dir = tempfile::tempdir().unwrap();
        let (published, _attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        write_complete_tt_extension(&published, &published, b"new");
        write_complete_tt_extension(&backup, &published, b"old");
        let manifest_path = published.join("diskgraph_manifest.json");
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["graph_index_path"] = json!(dir.path().join("wrong-index"));
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();

        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(is_complete_tt_extension_for(&published, &published));
        assert!(!backup.exists());
    }

    #[cfg(unix)]
    #[test]
    fn test_recovery_rejects_symlink_final_without_following_it() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let (published, _attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        let external = dir.path().join("external");
        write_complete_tt_extension(&external, &published, b"external");
        write_complete_tt_extension(&backup, &published, b"old");
        symlink(&external, &published).unwrap();

        assert!(!is_complete_tt_extension_for(&published, &published));
        recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap();
        assert_eq!(std::fs::read(published.join("generation")).unwrap(), b"old");
        assert!(is_complete_tt_extension_for(&published, &published));
        assert_eq!(std::fs::read(external.join("generation")).unwrap(), b"external");
        assert!(!backup.exists());
    }

    #[test]
    fn test_publish_tt_extension_fails_closed_on_ambiguous_or_invalid_backup() {
        let dir = tempfile::tempdir().unwrap();
        let (published, _attempt) = publication_paths(&dir);
        let backup = tt_extension_backup_path(&published).unwrap();
        let ambiguous = backup.with_file_name(".published.previous-stale");
        write_complete_tt_extension(&published, &published, b"new");
        write_complete_tt_extension(&backup, &published, b"old");
        write_complete_tt_extension(&ambiguous, &published, b"stale");

        let error = recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap_err();
        assert!(error.to_string().contains("ambiguous"));
        assert!(published.exists());
        assert!(backup.exists());
        assert!(ambiguous.exists());

        std::fs::remove_dir_all(&ambiguous).unwrap();
        std::fs::remove_file(backup.join("diskgraph_manifest.json")).unwrap();
        let error = recover_tt_extension_publication(&published, &mut |src, dst| {
            std::fs::rename(src, dst)
        })
        .unwrap_err();
        assert!(error.to_string().contains("incomplete TT extension backup"));
        assert!(published.exists());
        assert!(backup.exists());
    }

    // The cross-graph TT-reuse case that motivated ingest-time membership
    // verification: a staging bundle generated against G1 is re-ingested into
    // a DIFFERENT graph G2 missing some of G1's entities. Orphan seeds must
    // resolve to NULL (resolution mode) and be counted in the result dict's
    // `key_failures` — never silently collide onto a wrong G2 node.
    #[test]
    fn test_execute_tt_ingest_cross_graph_orphans_skipped_and_counted() {
        pyo3::Python::initialize();
        let dir_g1 = tempfile::tempdir().unwrap();
        let dir_g2 = tempfile::tempdir().unwrap();
        let g1 = t_build_graph_with_users(dir_g1.path(), &[0, 1, 2, 3, 4]);
        // G2 drops users 3 and 4 → any G1-gen'd seed for them is an orphan.
        let g2 = t_build_graph_with_users(dir_g2.path(), &[0, 1, 2]);

        let plan = t_node_plan(true);
        let plan_json = serde_json::to_string(&plan).unwrap();
        let (staging, total_rows) = pyo3::Python::attach(|py| {
            let d = execute_tt_gen(py, &plan_json, g1.to_str().unwrap(), "ttx").unwrap();
            let d = d.bind(py);
            let staging: String = d.get_item("staging_dir").unwrap().extract().unwrap();
            let n: u64 = d.get_item("row_count").unwrap().extract().unwrap();
            (staging, n)
        });
        assert_eq!(total_rows % 5, 0, "gen emits the same seed count per user");
        let rows_per_user = total_rows / 5;

        let (tt_ext, skipped) = pyo3::Python::attach(|py| {
            let d = execute_tt_ingest(
                py,
                &staging,
                g2.to_str().unwrap(),
                "ttx",
                Some("predict".to_string()),
            )
            .unwrap();
            let d = d.bind(py);
            let kf = d.get_item("key_failures").unwrap();
            let membership: u64 = kf
                .get_item("membership_failures")
                .unwrap()
                .extract()
                .unwrap();
            let lookup: u64 = kf.get_item("lookup_failures").unwrap().extract().unwrap();
            let ext: String = d.get_item("tt_extension_path").unwrap().extract().unwrap();
            (ext, membership + lookup)
        });
        assert_eq!(
            skipped,
            2 * rows_per_user,
            "every seed of the two dropped users is skipped + counted"
        );

        // The manifest persists the counters too (kumo can re-read them from
        // disk without the live result dict).
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(Path::new(&tt_ext).join("diskgraph_manifest.json")).unwrap(),
        )
        .unwrap();
        let mf_skipped = manifest["key_failures"]["membership_failures"]
            .as_u64()
            .unwrap()
            + manifest["key_failures"]["lookup_failures"].as_u64().unwrap();
        assert_eq!(mf_skipped, skipped, "manifest mirrors the result counters");
        // The denominator stays the PRE-drop seed total, so kumo's orphan-% is
        // "% of original seeds" even though the orphan rows are now dropped.
        assert_eq!(
            manifest["task_dataset"]["input_row_count"].as_u64(),
            Some(total_rows),
            "input_row_count is the pre-drop seed total"
        );

        // Golden: orphan-anchor seed rows are DROPPED at ingest — the stage holds
        // only the survivors, every one resolving to a G2 node whose PK is a
        // surviving user {0,1,2}. No null entity_node_id ever reaches disk.
        let ext = Path::new(&tt_ext);
        let reader = diskgraph_index::graph::GraphIndexReader::open(
            &ext.join("index_metadata.json"),
            ext,
        )
        .unwrap();
        let g2_reader =
            diskgraph_index::graph::GraphIndexReader::open(&g2.join("index_metadata.json"), &g2)
                .unwrap();
        let g2_users = g2_reader.table("users").unwrap();
        let valid_ids: std::collections::HashSet<u64> = [0u64, 1, 2]
            .iter()
            .map(|k| {
                g2_users
                    .node_id_from_pk(&diskgraph_index::pkmap::Key::Int(*k))
                    .unwrap()
                    .expect("surviving user resolves in G2")
            })
            .collect();
        let table = reader.table("predict_seeds").unwrap();
        let survivors = total_rows - skipped;
        assert_eq!(
            table.node_count(),
            survivors,
            "orphan-anchor rows dropped; only survivors kept"
        );
        for node_id in 0..survivors {
            let view = table.edges_for_node(node_id).unwrap();
            let json = table
                .decode_node_data_json(&view)
                .unwrap()
                .expect("node data present");
            let id = json
                .get("entity_node_id")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("survivor must carry a resolved id: {json}"));
            assert!(
                valid_ids.contains(&id),
                "resolved id {id} must be a surviving G2 user node: {json}"
            );
        }
    }

    #[test]
    fn test_execute_tt_ingest_split_path_still_three_stages() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let mut plan = t_node_plan(false);
        plan.split = Some(t_time_range_split());
        let plan_json = serde_json::to_string(&plan).unwrap();

        let staging = pyo3::Python::attach(|py| {
            let d = execute_tt_gen(py, &plan_json, index.to_str().unwrap(), "tttrain").unwrap();
            let d = d.bind(py);
            let staging: String = d.get_item("staging_dir").unwrap().extract().unwrap();
            staging
        });
        // Split plan produces per-stage seed parts `<table>/part-NNNNN.parquet`
        // (no split.bin); one part per stage on a single-row-group monolith.
        for table in ["train_seeds", "val_seeds", "test_seeds"] {
            assert!(
                Path::new(&staging)
                    .join(format!("{table}/part-00000.parquet"))
                    .exists(),
                "gen must write {table}/part-00000.parquet for a split plan"
            );
        }
        assert!(
            !Path::new(&staging).join("split.bin").exists(),
            "split.bin must not be written"
        );

        let tt_ext = pyo3::Python::attach(|py| {
            // No predict_stage_name → split path. NEW dir-API: pass the dir.
            let d =
                execute_tt_ingest(py, &staging, index.to_str().unwrap(), "tttrain", None).unwrap();
            let d = d.bind(py);
            let ext: String = d.get_item("tt_extension_path").unwrap().extract().unwrap();
            ext
        });

        let stages = read_manifest_stages(&tt_ext);
        let mut keys: Vec<_> = stages.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec!["test", "train", "val"]);
        assert_eq!(stages["train"]["table_name"].as_str(), Some("train_seeds"));
        assert_eq!(stages["val"]["table_name"].as_str(), Some("val_seeds"));
        assert_eq!(stages["test"]["table_name"].as_str(), Some("test_seeds"));
        // Self-describing: each stage lists its part(s) under
        // `<table>/part-NNNNN.parquet` (one part per stage on Phase-1 gen), and
        // each declared part must exist on disk in the extension.
        for (key, table) in [
            ("train", "train_seeds"),
            ("val", "val_seeds"),
            ("test", "test_seeds"),
        ] {
            assert_eq!(
                stages[key]["parts"].as_array().map(|a| a
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>()),
                Some(vec![format!("{table}/part-00000.parquet")]),
                "stage {key} must list its single part"
            );
            assert!(
                Path::new(&tt_ext)
                    .join(format!("{table}/part-00000.parquet"))
                    .exists(),
                "stage {key} part must be published on disk"
            );
        }
        // Self-describing task-level fields: TimeRangeSplit is temporal; the
        // node plan carries no List resolution ⇒ rhs_node_type=null.
        let td = read_task_dataset(&tt_ext);
        assert_eq!(
            td["is_temporal"].as_bool(),
            Some(true),
            "TimeRangeSplit (is_temporal=true) ⇒ manifest is_temporal=true"
        );
        assert!(
            td["rhs_node_type"].is_null(),
            "node plan ⇒ rhs_node_type=null, got {:?}",
            td["rhs_node_type"]
        );
    }

    // LP VAL/TEST single-timeframe guard (FIX 3): a >1-bucket val/test stage
    // (multiple distinct seed timestamps) must fail loud; <=1 is fine, and
    // non-eval stages (train) are unconstrained.
    #[test]
    fn enforce_single_timestamp_eval_stage_rejects_multi_ts_val_test() {
        // Multi-timestamp val / test -> error.
        let err = enforce_single_timestamp_eval_stage("val", 2).unwrap_err();
        assert!(
            err.to_string().contains("one timeframe"),
            "unexpected error: {err}"
        );
        assert!(enforce_single_timestamp_eval_stage("test", 3).is_err());
    }

    #[test]
    fn enforce_single_timestamp_eval_stage_allows_single_and_train() {
        // <=1 bucket on val/test is allowed.
        assert!(enforce_single_timestamp_eval_stage("val", 1).is_ok());
        assert!(enforce_single_timestamp_eval_stage("val", 0).is_ok());
        assert!(enforce_single_timestamp_eval_stage("test", 1).is_ok());
        // train is unconstrained even with many buckets.
        assert!(enforce_single_timestamp_eval_stage("train", 9).is_ok());
    }

    // ─── custom-TT source-mode == normal-path task_dataset parity ───────────
    //
    // The source-mode entry (`build_train_table_artifacts_for_existing_table`)
    // must publish into `index/train_tables/<tt_id>/` AND emit the SAME
    // `task_dataset` block the normal `execute_tt_plan` path produces, so kumo
    // can stamp the block verbatim (no hand-synthesis). This pins both the
    // returned dict shape and the on-disk manifest.

    /// Write a node-prediction custom-TT parquet (ENTITY1/TIMESTAMP/TARGET1)
    /// over `users` PKs that all exist in `t_build_graph`'s index.
    fn t_write_custom_tt_node_parquet(path: &Path) -> u64 {
        use arrow_array::{Float64Array, Int64Array, TimestampNanosecondArray};
        use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
        let entity = vec![0i64, 1, 2, 3];
        let ts: Vec<i64> = entity.iter().map(|_| 1_000i64 * 1_000_000_000).collect();
        let target = vec![1.0f64, 0.0, 1.0, 0.0];
        let schema = StdArc2::new(Schema::new(vec![
            Field::new(PRIMARY_ENTITY_COL, DataType::Int64, false),
            Field::new(
                TIMESTAMP_COL,
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                false,
            ),
            Field::new(PRIMARY_TARGET_COL, DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                StdArc2::new(Int64Array::from(entity.clone())),
                StdArc2::new(TimestampNanosecondArray::from(ts)),
                StdArc2::new(Float64Array::from(target)),
            ],
        )
        .unwrap();
        t_write_parquet(path, &batch);
        entity.len() as u64
    }

    #[test]
    fn test_train_binary_label_index_is_emitted_by_stage_ingest() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let stage_pq = dir.path().join("train_seed_rows.parquet");
        t_write_custom_tt_node_parquet(&stage_pq);
        let tt_ext = dir.path().join("tt_ext");
        let columns = diskgraph_ingest::infer_stage_columns(&stage_pq).unwrap();
        let result = diskgraph_ingest::ingest_stage_extension_with_resolution_and_sidecars(
            "binary_label_index_test",
            &tt_ext,
            &index,
            &[diskgraph_ingest::StageSpec {
                stage_table_name: "train_seeds".to_string(),
                parquet_paths: vec![stage_pq],
            }],
            std::collections::HashMap::from([(
                "train_seeds".to_string(),
                diskgraph_ingest::StageSidecarColumns {
                    binary_label_column: Some(PRIMARY_TARGET_COL.to_string()),
                    sample_weight_column: None,
                },
            )]),
            columns,
            vec![],
            &[diskgraph_ingest::ResolveSpec {
                src_col: PRIMARY_ENTITY_COL.to_string(),
                dst_table: "users".to_string(),
                kind: diskgraph_ingest::ResolveKind::Scalar,
                node_id_col: "entity_node_id".to_string(),
            }],
            Some(2),
            false,
            false,
        )
        .unwrap();

        let index = result
            .binary_label_indexes
            .get("train_seeds")
            .expect("bool-like TARGET1 must emit sidecar");

        assert_eq!(index.false_count, 2);
        assert_eq!(index.true_count, 2);

        fn read_node_ids(path: &Path) -> Vec<u64> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(8)
                .map(|chunk| {
                    let mut raw = [0u8; 8];
                    raw.copy_from_slice(chunk);
                    u64::from_le_bytes(raw)
                })
                .collect()
        }

        assert_eq!(
            read_node_ids(&tt_ext.join(&index.false_node_ids)),
            vec![2, 3],
            "false rows are source rows 1/3, which become shard-1 node ids"
        );
        assert_eq!(
            read_node_ids(&tt_ext.join(&index.true_node_ids)),
            vec![0, 1],
            "true rows are source rows 0/2, which become shard-0 node ids"
        );
    }

    #[test]
    fn test_train_sample_weight_index_is_emitted_by_stage_ingest() {
        use arrow_array::{Float64Array, Int64Array, TimestampNanosecondArray};
        use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};

        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let stage_pq = dir.path().join("train_seed_rows.parquet");
        let entity = vec![0i64, 1, 2, 3];
        let ts = vec![1000i64, 2000, 3000, 4000];
        let target = vec![1.0f64, 0.0, 1.0, 0.0];
        let weights = vec![0.5f64, 2.0, 4.0, 8.0];
        let schema = StdArc2::new(Schema::new(vec![
            Field::new(PRIMARY_ENTITY_COL, DataType::Int64, false),
            Field::new(
                TIMESTAMP_COL,
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                false,
            ),
            Field::new(PRIMARY_TARGET_COL, DataType::Float64, false),
            Field::new(
                diskgraph_train_table::ttgen_output::WEIGHT_COL,
                DataType::Float64,
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                StdArc2::new(Int64Array::from(entity)),
                StdArc2::new(TimestampNanosecondArray::from(ts)),
                StdArc2::new(Float64Array::from(target)),
                StdArc2::new(Float64Array::from(weights)),
            ],
        )
        .unwrap();
        t_write_parquet(&stage_pq, &batch);

        let tt_ext = dir.path().join("tt_ext");
        let columns = diskgraph_ingest::infer_stage_columns(&stage_pq).unwrap();
        let result = diskgraph_ingest::ingest_stage_extension_with_resolution_and_sidecars(
            "sample_weight_index_test",
            &tt_ext,
            &index,
            &[diskgraph_ingest::StageSpec {
                stage_table_name: "train_seeds".to_string(),
                parquet_paths: vec![stage_pq],
            }],
            std::collections::HashMap::from([(
                "train_seeds".to_string(),
                diskgraph_ingest::StageSidecarColumns {
                    binary_label_column: Some(PRIMARY_TARGET_COL.to_string()),
                    sample_weight_column: Some(
                        diskgraph_train_table::ttgen_output::WEIGHT_COL.to_string(),
                    ),
                },
            )]),
            columns,
            vec![],
            &[diskgraph_ingest::ResolveSpec {
                src_col: PRIMARY_ENTITY_COL.to_string(),
                dst_table: "users".to_string(),
                kind: diskgraph_ingest::ResolveKind::Scalar,
                node_id_col: "entity_node_id".to_string(),
            }],
            Some(2),
            false,
            false,
        )
        .unwrap();

        let index = result
            .sample_weight_indexes
            .get("train_seeds")
            .expect("valid WEIGHT must emit sidecar");
        assert_eq!(index.all.count, 4);
        assert_eq!(index.all.total_weight, 14.5);

        fn read_u64s(path: &Path) -> Vec<u64> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(8)
                .map(|chunk| {
                    let mut raw = [0u8; 8];
                    raw.copy_from_slice(chunk);
                    u64::from_le_bytes(raw)
                })
                .collect()
        }

        fn read_f64s(path: &Path) -> Vec<f64> {
            std::fs::read(path)
                .unwrap()
                .chunks_exact(8)
                .map(|chunk| {
                    let mut raw = [0u8; 8];
                    raw.copy_from_slice(chunk);
                    f64::from_le_bytes(raw)
                })
                .collect()
        }

        assert_eq!(
            read_u64s(&tt_ext.join(&index.all.node_ids)),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            read_f64s(&tt_ext.join(&index.all.cdf)),
            vec![0.5, 4.5, 6.5, 14.5]
        );
        let false_bucket = index.false_bucket.as_ref().unwrap();
        assert_eq!(read_u64s(&tt_ext.join(&false_bucket.node_ids)), vec![2, 3]);
        assert_eq!(read_f64s(&tt_ext.join(&false_bucket.cdf)), vec![2.0, 10.0]);
        let true_bucket = index.true_bucket.as_ref().unwrap();
        assert_eq!(read_u64s(&tt_ext.join(&true_bucket.node_ids)), vec![0, 1]);
        assert_eq!(read_f64s(&tt_ext.join(&true_bucket.cdf)), vec![0.5, 4.5]);
    }

    #[test]
    fn test_binary_label_index_is_train_only() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let train_pq = dir.path().join("train_seed_rows.parquet");
        let val_pq = dir.path().join("val_seed_rows.parquet");
        t_write_custom_tt_node_parquet(&train_pq);
        t_write_custom_tt_node_parquet(&val_pq);
        let tt_ext = dir.path().join("tt_ext");
        let columns = diskgraph_ingest::infer_stage_columns(&train_pq).unwrap();
        let result = diskgraph_ingest::ingest_stage_extension_with_resolution_and_sidecars(
            "binary_label_index_train_only_test",
            &tt_ext,
            &index,
            &[
                diskgraph_ingest::StageSpec {
                    stage_table_name: "train_seeds".to_string(),
                    parquet_paths: vec![train_pq],
                },
                diskgraph_ingest::StageSpec {
                    stage_table_name: "val_seeds".to_string(),
                    parquet_paths: vec![val_pq],
                },
            ],
            std::collections::HashMap::from([(
                "train_seeds".to_string(),
                diskgraph_ingest::StageSidecarColumns {
                    binary_label_column: Some(PRIMARY_TARGET_COL.to_string()),
                    sample_weight_column: None,
                },
            )]),
            columns,
            vec![],
            &[diskgraph_ingest::ResolveSpec {
                src_col: PRIMARY_ENTITY_COL.to_string(),
                dst_table: "users".to_string(),
                kind: diskgraph_ingest::ResolveKind::Scalar,
                node_id_col: "entity_node_id".to_string(),
            }],
            Some(2),
            false,
            false,
        )
        .unwrap();

        assert!(result.binary_label_indexes.contains_key("train_seeds"));
        assert!(!result.binary_label_indexes.contains_key("val_seeds"));
    }

    /// Write a PREDICTION custom-TT parquet (ENTITY1/TIMESTAMP, NO target) over
    /// `users` PKs that all exist in `t_build_graph`'s index. Mirrors
    /// `t_write_custom_tt_node_parquet` minus the TARGET1 column.
    fn t_write_custom_tt_predict_parquet(path: &Path) -> u64 {
        use arrow_array::{Int64Array, TimestampNanosecondArray};
        use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
        let entity = vec![0i64, 1, 2, 3];
        let ts: Vec<i64> = entity.iter().map(|_| 1_000i64 * 1_000_000_000).collect();
        let schema = StdArc2::new(Schema::new(vec![
            Field::new(PRIMARY_ENTITY_COL, DataType::Int64, false),
            Field::new(
                TIMESTAMP_COL,
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                StdArc2::new(Int64Array::from(entity.clone())),
                StdArc2::new(TimestampNanosecondArray::from(ts)),
            ],
        )
        .unwrap();
        t_write_parquet(path, &batch);
        entity.len() as u64
    }

    // Gen-only result-dict + durable-bundle shape (split path). Retargeted from
    // the old fused "publishes train_tables/<tt_id>/ + task_dataset" test: the
    // custom entry is now GEN-ONLY, so it returns bundle-describing keys
    // (`staging_dir`/`tt_parquet_path`/`ingest_spec_path`/`split_counts`) and
    // does NOT publish an extension. The gen→ingest publish parity is covered by
    // `test_custom_gen_ingest_parity_with_fused_path`.
    #[test]
    fn test_custom_tt_source_mode_gen_only_result_dict_and_bundle() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());

        // Customer-supplied parquet (already materialized): node-pred shape.
        let custom_pq = dir.path().join("custom_tt.parquet");
        let n_rows = t_write_custom_tt_node_parquet(&custom_pq);

        // RandomSplit so the source-mode pass emits train/val/test seed parquets.
        let split_json = serde_json::json!({
            "kind": "RandomSplit",
            "ratio": [0.5, 0.25, 0.25],
            "rnd_seed": 42,
            "is_temporal": false,
            "offsets": null,
            "offset_unit": null,
            "unit": null,
            "timeframe": null,
            "anchor_start_offset": null,
            "ranges": null,
            "backtesting_timestamp": null,
            "split_column": null,
            "split_output_column": null,
            "manual_train_values": null,
            "manual_val_values": null,
            "manual_test_values": null,
            "month_offsets": null,
            "flexible_boundaries": null,
            "timeframe_step": null,
        })
        .to_string();

        let tt_id = "custom_src_tt";
        let (staging_dir, tt_parquet_path, ingest_spec_path, row_count, split_counts_total) =
            pyo3::Python::attach(|py| {
                let d = crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
                    py,
                    custom_pq.to_str().unwrap(),
                    index.to_str().unwrap(),
                    tt_id,
                    "users",
                    "user_id",
                    "numerical",
                    "nanoseconds",
                    Some(&split_json),
                    None,
                    None,
                    None,
                    false,
                    None,
                    None,
                )
                .unwrap();
                let d = d.bind(py);
                // Gen-only: NO published-extension keys.
                assert!(
                    !d.contains("task_dataset").unwrap(),
                    "gen-only result must NOT carry a task_dataset block"
                );
                assert!(
                    !d.contains("tt_extension_path").unwrap(),
                    "gen-only result must NOT carry tt_extension_path"
                );
                assert!(
                    !d.contains("extension_path").unwrap(),
                    "gen-only result must NOT carry extension_path"
                );
                let staging_dir: String =
                    d.get_item("staging_dir").unwrap().extract().unwrap();
                let tt_parquet_path: String =
                    d.get_item("tt_parquet_path").unwrap().extract().unwrap();
                let ingest_spec_path: String =
                    d.get_item("ingest_spec_path").unwrap().extract().unwrap();
                let row_count: u64 = d.get_item("row_count").unwrap().extract().unwrap();
                // split_counts is dict[int,int]; sum the stage counts.
                let sc = d.get_item("split_counts").unwrap();
                let split_counts_total: u64 = sc
                    .try_iter()
                    .unwrap()
                    .map(|k| {
                        let k = k.unwrap();
                        sc.get_item(&k).unwrap().extract::<u64>().unwrap()
                    })
                    .sum();
                (
                    staging_dir,
                    tt_parquet_path,
                    ingest_spec_path,
                    row_count,
                    split_counts_total,
                )
            });

        // Result dict points at the durable bundle.
        let bundle = t_custom_bundle_dir(&custom_pq);
        assert_eq!(
            Path::new(&staging_dir),
            bundle,
            "staging_dir == the kept bundle dir"
        );
        assert_eq!(row_count, n_rows, "row_count == input rows");
        assert_eq!(
            split_counts_total, n_rows,
            "split_counts must partition the input rows"
        );

        // Bundle is durable + self-describing: spec + per-stage seed parquets.
        assert_eq!(
            Path::new(&ingest_spec_path),
            bundle.join(TT_INGEST_SPEC_FILENAME),
            "ingest_spec_path is the bundle's tt_ingest_spec.json"
        );
        assert!(Path::new(&tt_parquet_path).exists(), "tt.parquet exists in bundle");
        let spec: TtIngestSpec =
            serde_json::from_slice(&std::fs::read(&ingest_spec_path).unwrap()).unwrap();
        let mut names: Vec<String> = spec
            .split_stages
            .expect("split spec must carry split_stages")
            .iter()
            .map(|s| s.table_name.clone())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "test_seeds".to_string(),
                "train_seeds".to_string(),
                "val_seeds".to_string(),
            ],
            "RandomSplit gen must emit train/val/test seed tables"
        );
        for table in ["train_seeds", "val_seeds", "test_seeds"] {
            assert!(
                bundle.join(format!("{table}/part-00000.parquet")).exists(),
                "seed part {table}/part-00000.parquet must be durable in the bundle"
            );
        }

        // Gen-only: no extension published under train_tables/<tt_id>/.
        assert!(
            !index
                .join("train_tables")
                .join(tt_id)
                .join("diskgraph_manifest.json")
                .exists(),
            "gen-only must NOT publish train_tables/<tt_id>/"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // custom-TT GEN-ONLY DURABLE BUNDLE CONTRACT (spec §DG 1-5)
    //
    // Contract: `build_train_table_artifacts_for_existing_table` must become
    // GEN-ONLY — it produces a durable, self-describing staging bundle
    // (`tt.parquet` + `tt_ingest_spec.json` [+ per-stage seed parquets]) that the
    // normal `execute_tt_ingest` can consume on ANY pod, any number of times. It
    // must NOT fuse ingest+publish and NOT delete the bundle.
    //
    // These tests express the TARGET (post-fix) behavior. Against the CURRENT
    // fused code they are RED for the RIGHT reason: the custom path never writes
    // `tt_ingest_spec.json` into the bundle, so the bundle is not re-ingestable.
    //
    // The bundle dir is where `execute_tt_pass_with_source` writes its outputs:
    // for a `<name>.parquet` input that is `<name>.artifacts` (see
    // `custom_tt_artifact_base`); `execute_tt_ingest` reads `tt.parquet` +
    // `tt_ingest_spec.json` as siblings from that dir.
    // ═══════════════════════════════════════════════════════════════════════

    /// The gen bundle dir for a single-file `<name>.parquet` custom-TT input.
    /// Mirrors `custom_tt_entry::custom_tt_artifact_base` (private) + the
    /// `.artifacts` suffix `execute_tt_pass_with_source` writes into.
    fn t_custom_bundle_dir(custom_pq: &Path) -> std::path::PathBuf {
        let stem = custom_pq
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_suffix(".parquet"))
            .expect("custom parquet must be a <name>.parquet file");
        custom_pq.with_file_name(format!("{stem}.artifacts"))
    }

    /// Write a link-prediction custom-TT parquet (ENTITY1 over `users`, TARGET1
    /// = List<Int64> of `orders` PKs) so the LP resolution/fkey_mapping path in
    /// `build_train_table_artifacts_for_existing_table` is exercised.
    fn t_write_custom_tt_lp_parquet(path: &Path) -> u64 {
        use arrow_array::builder::{Int64Builder, ListBuilder};
        use arrow_array::{Array, Int64Array, TimestampNanosecondArray};
        use arrow_schema::{DataType, Field, Schema, TimeUnit as ArrowTimeUnit};
        let entity = vec![0i64, 1, 2, 3];
        // Each entity's TARGET1 references real `orders` node keys (0..10 exist).
        let targets: Vec<Vec<i64>> = vec![vec![0], vec![1, 2], vec![3], vec![4, 5]];
        let ts: Vec<i64> = entity.iter().map(|_| 1_000i64 * 1_000_000_000).collect();
        let mut tb = ListBuilder::new(Int64Builder::new());
        for row in &targets {
            for &t in row {
                tb.values().append_value(t);
            }
            tb.append(true);
        }
        let schema = StdArc2::new(Schema::new(vec![
            Field::new(PRIMARY_ENTITY_COL, DataType::Int64, false),
            Field::new(
                TIMESTAMP_COL,
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                false,
            ),
            Field::new(
                PRIMARY_TARGET_COL,
                DataType::List(StdArc2::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
        ]));
        let cols: Vec<StdArc2<dyn Array>> = vec![
            StdArc2::new(Int64Array::from(entity.clone())),
            StdArc2::new(TimestampNanosecondArray::from(ts)),
            StdArc2::new(tb.finish()),
        ];
        let batch = RecordBatch::try_new(schema, cols).unwrap();
        t_write_parquet(path, &batch);
        entity.len() as u64
    }

    // TDD (spec §DG 1): custom gen must leave a durable bundle with BOTH
    // `tt.parquet` AND `tt_ingest_spec.json`, and must NOT publish an extension.
    // Impl change required: make the entry gen-only (write the spec, keep the
    // bundle, drop the fused ingest/publish + `remove_dir_all`).
    #[test]
    fn test_custom_gen_bundle_completeness_node_pred_no_split() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let custom_pq = dir.path().join("custom_tt.parquet");
        let _n_rows = t_write_custom_tt_node_parquet(&custom_pq);

        let tt_id = "custom_bundle_ns";
        pyo3::Python::attach(|py| {
            let _ = crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
                py,
                custom_pq.to_str().unwrap(),
                index.to_str().unwrap(),
                tt_id,
                "users",
                "user_id",
                "numerical",
                "nanoseconds",
                None, // no split
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .expect("custom gen build should succeed");
        });

        let bundle = t_custom_bundle_dir(&custom_pq);
        assert!(
            bundle.join("tt.parquet").exists(),
            "gen bundle must contain tt.parquet at {}",
            bundle.display()
        );
        // RED today: the custom path never writes tt_ingest_spec.json.
        assert!(
            bundle.join(TT_INGEST_SPEC_FILENAME).exists(),
            "gen bundle must contain {TT_INGEST_SPEC_FILENAME} at {} (durable, \
             re-ingestable bundle) — MISSING today because the fused custom path \
             never writes it",
            bundle.display()
        );
        // Gen-only: the bundle dir must NOT be deleted.
        assert!(bundle.exists(), "gen must KEEP the bundle dir (no remove_dir_all)");
        // Gen-only: the gen call must NOT publish a samplable extension.
        let published_ext = index.join("train_tables").join(tt_id);
        assert!(
            !published_ext.join("diskgraph_manifest.json").exists(),
            "gen-only must NOT publish train_tables/<tt_id>/ (found extension at {})",
            published_ext.display()
        );
    }

    // TDD (spec §DG 2): with a SplitSpec the bundle must carry per-stage
    // `<stage>/part-*.parquet` AND `tt_ingest_spec.json` whose `split_stages`
    // lists them. Impl change required: gen-only + write spec (with split_stages
    // from the pass output, like `execute_tt_gen`).
    #[test]
    fn test_custom_gen_bundle_completeness_split() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let custom_pq = dir.path().join("custom_tt.parquet");
        let _n_rows = t_write_custom_tt_node_parquet(&custom_pq);

        // RandomSplit → train/val/test per-stage seed parquets.
        let split_json = serde_json::json!({
            "kind": "RandomSplit",
            "ratio": [0.5, 0.25, 0.25],
            "rnd_seed": 42,
            "is_temporal": false,
            "offsets": null, "offset_unit": null, "unit": null, "timeframe": null,
            "anchor_start_offset": null, "ranges": null, "backtesting_timestamp": null,
            "split_column": null, "split_output_column": null,
            "manual_train_values": null, "manual_val_values": null, "manual_test_values": null,
            "month_offsets": null, "flexible_boundaries": null, "timeframe_step": null,
        })
        .to_string();

        let tt_id = "custom_bundle_split";
        pyo3::Python::attach(|py| {
            let _ = crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
                py,
                custom_pq.to_str().unwrap(),
                index.to_str().unwrap(),
                tt_id,
                "users",
                "user_id",
                "numerical",
                "nanoseconds",
                Some(&split_json),
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .expect("custom gen build (split) should succeed");
        });

        let bundle = t_custom_bundle_dir(&custom_pq);
        // Per-stage seed parts must be durable in the bundle.
        for table in ["train_seeds", "val_seeds", "test_seeds"] {
            assert!(
                bundle.join(format!("{table}/part-00000.parquet")).exists(),
                "split gen bundle must contain {table}/part-00000.parquet at {}",
                bundle.display()
            );
        }
        // RED today: no spec ⇒ cannot list split_stages.
        let spec_path = bundle.join(TT_INGEST_SPEC_FILENAME);
        assert!(
            spec_path.exists(),
            "split gen bundle must contain {TT_INGEST_SPEC_FILENAME} at {}",
            bundle.display()
        );
        let spec: TtIngestSpec =
            serde_json::from_slice(&std::fs::read(&spec_path).unwrap()).unwrap();
        let stages = spec
            .split_stages
            .expect("split spec must carry split_stages");
        let mut names: Vec<String> = stages.iter().map(|s| s.table_name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "test_seeds".to_string(),
                "train_seeds".to_string(),
                "val_seeds".to_string(),
            ],
            "split_stages must list the three per-stage seed tables"
        );
    }

    // TDD (spec §DG 3): for `target_entity_table` set (LP), the written spec
    // must carry the LP resolution (entity Scalar `entity_node_id` + target List
    // `target_node_ids`), the fkey_mapping for entity+target, and the correct
    // entity_table. Impl change required: gen-only writes the spec derived from
    // the (LP) plan, exactly like `execute_tt_gen`/`TtIngestSpec::from_plan`.
    #[test]
    fn test_custom_gen_spec_content_lp() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let custom_pq = dir.path().join("custom_tt.parquet");
        let _n_rows = t_write_custom_tt_lp_parquet(&custom_pq);

        let tt_id = "custom_bundle_lp";
        // The entry may or may not fully succeed for LP gen here, but the DURABLE
        // spec must exist regardless (gen-only writes it as part of the bundle).
        // Ignore the call result so this test isolates the bundle-artifact
        // contract from any downstream LP-gen specifics.
        let _ = pyo3::Python::attach(|py| {
            crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
                py,
                custom_pq.to_str().unwrap(),
                index.to_str().unwrap(),
                tt_id,
                "users",
                "user_id",
                "numerical",
                "nanoseconds",
                None,
                None,
                Some("orders"), // target_entity_table ⇒ LP mode
                None,           // task_type
                false,
                None,
                None,
            )
        });

        let bundle = t_custom_bundle_dir(&custom_pq);
        let spec_path = bundle.join(TT_INGEST_SPEC_FILENAME);
        // RED today: the custom LP path never writes the spec.
        assert!(
            spec_path.exists(),
            "LP gen bundle must contain {TT_INGEST_SPEC_FILENAME} at {}",
            bundle.display()
        );
        let spec: TtIngestSpec =
            serde_json::from_slice(&std::fs::read(&spec_path).unwrap()).unwrap();

        assert_eq!(spec.entity_table, "users", "spec.entity_table");

        let res = spec.lp_resolution.expect("LP spec must carry lp_resolution");
        assert_eq!(res.len(), 2, "lp_resolution = [entity Scalar, target List]");
        let entity_res = res
            .iter()
            .find(|r| r.kind == RefResolveKind::Scalar)
            .expect("entity Scalar resolution");
        assert_eq!(entity_res.src_col, PRIMARY_ENTITY_COL);
        assert_eq!(entity_res.dst_table, "users");
        assert_eq!(entity_res.node_id_col, "entity_node_id");
        let target_res = res
            .iter()
            .find(|r| r.kind == RefResolveKind::List)
            .expect("target List resolution");
        assert_eq!(target_res.src_col, PRIMARY_TARGET_COL);
        assert_eq!(target_res.dst_table, "orders");
        assert_eq!(target_res.node_id_col, "target_node_ids");

        let fkeys = spec.fkey_mapping.expect("LP spec must carry fkey_mapping");
        assert!(
            fkeys
                .iter()
                .any(|f| f.src_col == PRIMARY_ENTITY_COL && f.dst_table == "users"),
            "fkey_mapping must include entity → users, got {fkeys:?}"
        );
        assert!(
            fkeys
                .iter()
                .any(|f| f.src_col == PRIMARY_TARGET_COL && f.dst_table == "orders"),
            "fkey_mapping must include target → orders, got {fkeys:?}"
        );
    }

    // TDD (spec §DG 4 — THE BUG): custom gen bundle must be re-ingestable by the
    // normal `execute_tt_ingest` (pod-independent), publishing
    // `train_tables/<tt_id>/` with the 4 required manifest files. Today this
    // fails because the bundle the custom path leaves has no
    // `tt_ingest_spec.json` (execute_tt_ingest hard-requires it at :443-455).
    // Impl change required: gen-only leaves a complete, re-ingestable bundle.
    #[test]
    fn test_custom_gen_bundle_roundtrips_through_execute_tt_ingest() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());
        let custom_pq = dir.path().join("custom_tt.parquet");
        let _n_rows = t_write_custom_tt_node_parquet(&custom_pq);

        // Gen phase: build the durable bundle (gen-only). TRAIN contract:
        // source==target — training re-ingests its OWN gen bundle under the
        // SAME id (no fan-out). Only PREDICT ingests one gen bundle under a
        // different per-BP-job id (see
        // `test_custom_predict_bundle_reingests_under_distinct_target_id`).
        let tt_id = "custom_rt";
        pyo3::Python::attach(|py| {
            let _ = crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
                py,
                custom_pq.to_str().unwrap(),
                index.to_str().unwrap(),
                tt_id,
                "users",
                "user_id",
                "numerical",
                "nanoseconds",
                None,
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .expect("custom gen build should succeed");
        });

        let bundle = t_custom_bundle_dir(&custom_pq);
        let bundle_str = bundle.to_string_lossy().into_owned();

        // Ingest phase: the normal, pod-independent consumer, re-ingesting the
        // bundle under the SAME id (train contract: source==target).
        let ingest_tt_id = tt_id;
        let tt_ext = pyo3::Python::attach(|py| {
            let d = execute_tt_ingest(
                py,
                &bundle_str,
                index.to_str().unwrap(),
                ingest_tt_id,
                Some("predict".to_string()),
            )
            .expect(
                "execute_tt_ingest must consume the custom gen bundle — fails today \
                 because the fused custom path never wrote tt_ingest_spec.json",
            );
            let d = d.bind(py);
            d.get_item("tt_extension_path")
                .unwrap()
                .extract::<String>()
                .unwrap()
        });

        // The published extension must carry the 4 required files.
        let ext = Path::new(&tt_ext);
        for f in [
            "manifest.json",
            "diskgraph_manifest.json",
            "stats.json",
            "train_table_col_stats.json",
        ] {
            assert!(
                ext.join(f).exists(),
                "published extension must contain {f} at {}",
                ext.display()
            );
        }
    }

    // TDD (spec §DG 5 — parity): the extension produced by (custom gen →
    // execute_tt_ingest) must be a well-formed, samplable split extension. The
    // old fused-publish reference is GONE (custom gen is now gen-only), so per
    // the TDD author's note we drop the fused-reference half and assert the
    // gen→ingest half directly: a 3-stage (train/val/test) extension whose
    // per-stage table_names + parts are correct and whose row_counts partition
    // the input rows. This is the same manifest shape the normal split ingest
    // path (`test_execute_tt_ingest_split_path_still_three_stages`) produces.
    #[test]
    fn test_custom_gen_ingest_parity_with_fused_path() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());

        // Split so there are multiple stages.
        let split_json = serde_json::json!({
            "kind": "RandomSplit",
            "ratio": [0.5, 0.25, 0.25],
            "rnd_seed": 42,
            "is_temporal": false,
            "offsets": null, "offset_unit": null, "unit": null, "timeframe": null,
            "anchor_start_offset": null, "ranges": null, "backtesting_timestamp": null,
            "split_column": null, "split_output_column": null,
            "manual_train_values": null, "manual_val_values": null, "manual_test_values": null,
            "month_offsets": null, "flexible_boundaries": null, "timeframe_step": null,
        })
        .to_string();

        // Gen-only bundle → normal execute_tt_ingest.
        let gen_pq = dir.path().join("custom_tt_gen.parquet");
        let n_rows = t_write_custom_tt_node_parquet(&gen_pq);
        // TRAIN contract: source==target. Training re-ingests its own gen
        // bundle under the SAME id (no per-BP-job fan-out — that's predict).
        let tt_id = "parity";
        pyo3::Python::attach(|py| {
            let _ = crate::custom_tt_entry::build_train_table_artifacts_for_existing_table(
                py,
                gen_pq.to_str().unwrap(),
                index.to_str().unwrap(),
                tt_id,
                "users",
                "user_id",
                "numerical",
                "nanoseconds",
                Some(&split_json),
                None,
                None,
                None,
                false,
                None,
                None,
            )
            .expect("gen build should succeed");
        });
        let bundle = t_custom_bundle_dir(&gen_pq);
        let bundle_str = bundle.to_string_lossy().into_owned();
        let ingest_tt_id = tt_id;
        let ingest_ext = pyo3::Python::attach(|py| {
            let d = execute_tt_ingest(
                py,
                &bundle_str,
                index.to_str().unwrap(),
                ingest_tt_id,
                None, // split path
            )
            .expect(
                "execute_tt_ingest must consume the custom gen bundle — fails \
                 without tt_ingest_spec.json in the bundle",
            );
            let d = d.bind(py);
            d.get_item("tt_extension_path")
                .unwrap()
                .extract::<String>()
                .unwrap()
        });
        let ingest_stages = read_manifest_stages(&ingest_ext);

        // 3-stage extension with the canonical seed-table names.
        let mut keys: Vec<String> = ingest_stages.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec!["test".to_string(), "train".to_string(), "val".to_string()],
            "gen→ingest must publish train/val/test stages"
        );
        for (key, table) in [
            ("train", "train_seeds"),
            ("val", "val_seeds"),
            ("test", "test_seeds"),
        ] {
            assert_eq!(
                ingest_stages[key]["table_name"].as_str(),
                Some(table),
                "stage {key} table_name"
            );
            assert_eq!(
                ingest_stages[key]["parts"].as_array().map(|a| a
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>()),
                Some(vec![format!("{table}/part-00000.parquet")]),
                "stage {key} must list its single part"
            );
            assert!(
                Path::new(&ingest_ext)
                    .join(format!("{table}/part-00000.parquet"))
                    .exists(),
                "stage {key} part must be published on disk"
            );
        }
        // Row counts must partition the input rows across the three stages.
        let total: u64 = ["train", "val", "test"]
            .iter()
            .map(|k| ingest_stages[*k]["row_count"].as_u64().unwrap())
            .sum();
        assert_eq!(
            total, n_rows,
            "train+val+test row_counts must sum to the input row count"
        );
    }

    // TDD (custom-TT PREDICT sibling): the gen-only PREDICT entry
    // (`build_prediction_table_artifacts_for_existing_table`) must produce a
    // durable single-`predict`-stage bundle from a NO-TARGET parquet, and that
    // bundle must be re-ingestable under a DIFFERENT (per-BP-job-style) target
    // id. This exercises the PREDICT fan-out honestly: unlike train
    // (source==target), predict re-ingests one gen bundle under a per-BP-job
    // extension id (source≠target). Asserts: (a) gen writes a single `predict`
    // stage + `tt_ingest_spec.json` with `is_prediction` and NO published
    // extension, (b) `execute_tt_ingest(..., predict_stage_name="predict")`
    // under a DISTINCT target id publishes `train_tables/<target>/`.
    #[test]
    fn test_custom_predict_bundle_reingests_under_distinct_target_id() {
        pyo3::Python::initialize();
        let dir = tempfile::tempdir().unwrap();
        let index = t_build_graph(dir.path());

        // Customer prediction parquet: ENTITY1 + TIMESTAMP, NO target.
        let predict_pq = dir.path().join("custom_predict.parquet");
        let n_rows = t_write_custom_tt_predict_parquet(&predict_pq);

        // Gen phase: build the durable PREDICT bundle (gen-only). The gen id is
        // the bundle's own `spec.tt_id`.
        let gen_tt_id = "custom_pred_gen";
        pyo3::Python::attach(|py| {
            let _ =
                crate::custom_tt_entry::build_prediction_table_artifacts_for_existing_table(
                    py,
                    predict_pq.to_str().unwrap(),
                    index.to_str().unwrap(),
                    gen_tt_id,
                    "users",
                    "user_id",
                    "numerical",
                    "nanoseconds",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("custom predict gen build should succeed");
        });

        let bundle = t_custom_bundle_dir(&predict_pq);

        // (a) The bundle holds a single `predict` seed table (SplitMode::None):
        // exactly one `tt.parquet`, no train/val/test seed parquets.
        assert!(
            bundle.join("tt.parquet").exists(),
            "predict bundle must hold the single tt.parquet at {}",
            bundle.display()
        );
        for stage in ["train_seeds", "val_seeds", "test_seeds"] {
            assert!(
                !bundle.join(stage).exists(),
                "predict bundle must NOT emit split stage {stage}"
            );
        }

        // (a) `tt_ingest_spec.json` present, self-describing, and marks the
        // gen id + NO split stages (the prediction path leaves `split_stages`
        // null so `execute_tt_ingest` takes the single-`predict`-stage path).
        let spec_path = bundle.join(TT_INGEST_SPEC_FILENAME);
        assert!(
            spec_path.exists(),
            "predict bundle must write tt_ingest_spec.json"
        );
        let spec_val: Value =
            serde_json::from_slice(&std::fs::read(&spec_path).unwrap()).unwrap();
        assert_eq!(
            spec_val["tt_id"].as_str(),
            Some(gen_tt_id),
            "spec.tt_id names the gen bundle id"
        );
        assert!(
            spec_val["split_stages"].is_null(),
            "predict spec must have no split_stages (single predict stage)"
        );

        // (a) No extension published by the gen-only entry itself.
        assert!(
            !index.join("train_tables").join(gen_tt_id).exists(),
            "gen-only predict entry must NOT publish an extension"
        );

        // (b) PREDICT fan-out: ingest the SAME bundle under a DIFFERENT
        // per-BP-job target id (source≠target — the train contract never does
        // this). This must publish `train_tables/<target>/` with the required
        // manifest files.
        let bundle_str = bundle.to_string_lossy().into_owned();
        let target_tt_id = "custom_pred_bpjob_0007";
        assert_ne!(
            target_tt_id, gen_tt_id,
            "predict target id must differ from the gen bundle id"
        );
        let tt_ext = pyo3::Python::attach(|py| {
            let d = execute_tt_ingest(
                py,
                &bundle_str,
                index.to_str().unwrap(),
                target_tt_id,
                Some("predict".to_string()),
            )
            .expect("execute_tt_ingest must publish the predict bundle under the target id");
            let d = d.bind(py);
            d.get_item("tt_extension_path")
                .unwrap()
                .extract::<String>()
                .unwrap()
        });

        // The published extension lives under the TARGET id, not the gen id.
        assert!(
            tt_ext.ends_with(target_tt_id),
            "extension must publish under the per-BP-job target id, got {tt_ext}"
        );
        let ext = Path::new(&tt_ext);
        for f in [
            "manifest.json",
            "diskgraph_manifest.json",
            "stats.json",
            "train_table_col_stats.json",
        ] {
            assert!(
                ext.join(f).exists(),
                "published predict extension must contain {f} at {}",
                ext.display()
            );
        }

        // The single `predict` stage must carry all input rows.
        let stages = read_manifest_stages(&tt_ext);
        assert_eq!(
            stages.keys().cloned().collect::<Vec<_>>(),
            vec!["predict".to_string()],
            "predict ingest must publish exactly one `predict` stage"
        );
        assert_eq!(
            stages["predict"]["row_count"].as_u64(),
            Some(n_rows),
            "predict stage must carry every input row"
        );
    }
