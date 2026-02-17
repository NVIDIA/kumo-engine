#[cfg(test)]
mod tests {
    use crate::dataset::{DatasetHandle, DatasetId, DatasetReservation};
    use crate::errors::{DiskgraphError, Result};
    use crate::features::BatchFeatureRequest;
    use crate::metadata::DatasetMetadata;
    use crate::sampling::*;
    use crate::storage::GraphStorage;
    use arrow_array::{Array, Int64Array, Int8Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// A mock implementation of GraphStorage to verify the trait and protocol.
    struct MockStorage;

    #[async_trait]
    impl GraphStorage for MockStorage {
        async fn reserve(&self, _m: &DatasetMetadata) -> Result<DatasetReservation> {
            unimplemented!()
        }
        async fn ingest(&self, _h: DatasetHandle) -> Result<DatasetHandle> {
            unimplemented!()
        }
        async fn unload(&self, _id: DatasetId) -> Result<()> {
            unimplemented!()
        }
        async fn get_features(&self, _r: BatchFeatureRequest) -> Result<RecordBatch> {
            unimplemented!()
        }

        async fn sample(&self, request: BatchSamplingRequest) -> Result<RecordBatch> {
            // --- SERVER-SIDE SIMULATION ---
            let mut local_index_maps: HashMap<String, HashMap<u64, i64>> = HashMap::new();
            let mut next_local_ids: HashMap<String, i64> = HashMap::new();

            let assign_local = |node_type: &str,
                                global_id: u64,
                                maps: &mut HashMap<String, HashMap<u64, i64>>,
                                next_ids: &mut HashMap<String, i64>|
             -> i64 {
                let map = maps.entry(node_type.to_string()).or_default();
                if let Some(&local) = map.get(&global_id) {
                    local
                } else {
                    let local = *next_ids.entry(node_type.to_string()).or_insert(0);
                    map.insert(global_id, local);
                    next_ids.insert(node_type.to_string(), local + 1);
                    local
                }
            };

            // 1. Initialize Seeds
            for (node_type, globals) in &request.seed_nodes {
                for &g in globals {
                    assign_local(node_type, g, &mut local_index_maps, &mut next_local_ids);
                }
            }

            // 2. Mock Edge Sampling (User 100/200 -> Item 1/2/3)
            let edge_rows = [
                (
                    0,
                    assign_local("Item", 1, &mut local_index_maps, &mut next_local_ids),
                    "User__Click__Item",
                    1001,
                ),
                (
                    0,
                    assign_local("Item", 2, &mut local_index_maps, &mut next_local_ids),
                    "User__Click__Item",
                    1002,
                ),
                (
                    1,
                    assign_local("Item", 2, &mut local_index_maps, &mut next_local_ids),
                    "User__Click__Item",
                    2001,
                ),
                (
                    1,
                    assign_local("Item", 3, &mut local_index_maps, &mut next_local_ids),
                    "User__Click__Item",
                    2002,
                ),
            ];

            // 3. Build Multiplexed Schema
            let schema = Arc::new(Schema::new(vec![
                Field::new("row_type", DataType::Int8, false),
                Field::new("id_a", DataType::Int64, false),
                Field::new("id_b", DataType::Int64, false),
                Field::new("meta_type", DataType::Utf8, false),
                Field::new("meta_val", DataType::Int64, true),
            ]));

            let id_a = Int64Array::from(edge_rows.iter().map(|r| r.0).collect::<Vec<_>>());
            let id_b = Int64Array::from(edge_rows.iter().map(|r| r.1).collect::<Vec<_>>());
            let meta_type = StringArray::from(edge_rows.iter().map(|r| r.2).collect::<Vec<_>>());
            let meta_val = Int64Array::from(edge_rows.iter().map(|r| r.3).collect::<Vec<_>>());
            let row_type = Int8Array::from(vec![0; edge_rows.len()]);

            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(row_type),
                    Arc::new(id_a),
                    Arc::new(id_b),
                    Arc::new(meta_type),
                    Arc::new(meta_val),
                ],
            )
            .map_err(|e| DiskgraphError::Generic(e.to_string()))
        }
    }

    #[tokio::test]
    async fn test_sampling_trait_usage() {
        let storage = MockStorage;
        let dataset_id = DatasetId::new_v4();

        // 1. CONSTRUCT REQUEST
        let request = BatchSamplingRequest {
            dataset_id,
            seed_nodes: [("User".to_string(), vec![100, 200])].into(),
            seed_timestamps: None,
            plan: SamplingPlan {
                layers: vec![LayerSpec {
                    fanouts: [(
                        "User__Click__Item".to_string(),
                        FanoutSpec {
                            fanout: 2,
                            mode: SamplingMode::Uniform,
                            absent_key_policy: AbsentKeyPolicy::RejectPlan,
                        },
                    )]
                    .into(),
                    default_fanout: None,
                }],
            },
            flags: SamplingFlags {
                replace: false,
                directed: true,
                disjoint: true,
                return_edge_id: false,
                adaptive: false,
                fused_feature_fetching: false,
                swap_edge_direction: true,
            },
        };

        // 2. CALL TRAIT
        let batch = storage.sample(request).await.expect("Sampling failed");

        // 3. VERIFY OUTPUT
        assert_eq!(batch.num_rows(), 4);
        let id_as = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let id_bs = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(id_bs.value(1), id_bs.value(2));
        assert_eq!(id_as.value(0), 0);
        assert_eq!(id_as.value(2), 1);
    }
}
