#[cfg(test)]
mod tests {
    use crate::dataset::DatasetId;
    use crate::sampling::*;
    use arrow_array::{Array, Int64Array, Int8Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_sampling_flow_simulation() {
        // --- 1. SETUP MOCK GRAPH & REQUEST ---
        let dataset_id = DatasetId::new_v4();
        let seed_nodes = [("User".to_string(), vec![100, 200])].into(); // 2 User seeds
        
        let plan = SamplingPlan {
            layers: vec![
                LayerSpec {
                    fanouts: [
                        ("User__Click__Item".to_string(), FanoutSpec {
                            fanout: 2,
                            mode: SamplingMode::Uniform,
                            absent_key_policy: AbsentKeyPolicy::RejectPlan,
                        }),
                    ].into(),
                }
            ],
        };

        let request = BatchSamplingRequest {
            dataset_id,
            seed_nodes,
            seed_timestamps: None,
            plan,
            flags: SamplingFlags {
                replace: false,
                directed: true,
                disjoint: false, // Merged for this test
                return_edge_id: true,
                adaptive: false,
            },
        };

        // --- 2. SIMULATE SERVER (RE-INDEXING) ---
        // Mock storage: 
        // User 100 -> Item 1, Item 2
        // User 200 -> Item 2, Item 3
        
        let mut local_index_maps: HashMap<String, HashMap<u64, i64>> = HashMap::new();
        let mut next_local_ids: HashMap<String, i64> = HashMap::new();

        let mut assign_local = |node_type: &str, global_id: u64, maps: &mut HashMap<String, HashMap<u64, i64>>, next_ids: &mut HashMap<String, i64>| -> i64 {
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

        // Step 2a: Initialize Seeds
        for (node_type, globals) in &request.seed_nodes {
            for &g in globals {
                assign_local(node_type, g, &mut local_index_maps, &mut next_local_ids);
            }
        }

        // Step 2b: Sample Layer 0 -> 1
        // We'll manually emit Edges for User 100 and 200
        let mut edge_rows = vec![];
        // User 100 (Local 0) -> Item 1 (Local 0), Item 2 (Local 1)
        edge_rows.push((0, assign_local("Item", 1, &mut local_index_maps, &mut next_local_ids), "User__Click__Item", 1001));
        edge_rows.push((0, assign_local("Item", 2, &mut local_index_maps, &mut next_local_ids), "User__Click__Item", 1002));
        // User 200 (Local 1) -> Item 2 (Local 1), Item 3 (Local 2)
        edge_rows.push((1, assign_local("Item", 2, &mut local_index_maps, &mut next_local_ids), "User__Click__Item", 2001));
        edge_rows.push((1, assign_local("Item", 3, &mut local_index_maps, &mut next_local_ids), "User__Click__Item", 2002));

        // --- 3. CONSTRUCT ARROW RESPONSE ---
        let schema = Arc::new(Schema::new(vec![
            Field::new("row_type", DataType::Int8, false),
            Field::new("id_a", DataType::Int64, false),
            Field::new("id_b", DataType::Int64, false),
            Field::new("meta_type", DataType::Utf8, false), // Simplified for simulation
            Field::new("meta_val", DataType::Int64, true),
        ]));

        let row_type = Int8Array::from(vec![0, 0, 0, 0]); // All Edges
        let id_a = Int64Array::from(edge_rows.iter().map(|r| r.0).collect::<Vec<_>>());
        let id_b = Int64Array::from(edge_rows.iter().map(|r| r.1).collect::<Vec<_>>());
        let meta_type = StringArray::from(edge_rows.iter().map(|r| r.2).collect::<Vec<_>>());
        let meta_val = Int64Array::from(edge_rows.iter().map(|r| r.3).collect::<Vec<_>>());

        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(row_type), Arc::new(id_a), Arc::new(id_b), Arc::new(meta_type), Arc::new(meta_val)
        ]).unwrap();

        // --- 4. SIMULATE CLIENT RECONSTRUCTION ---
        // Goal: Build row_dict, col_dict, node_dict
        
        let mut row_dict: HashMap<String, Vec<i64>> = HashMap::new();
        let mut col_dict: HashMap<String, Vec<i64>> = HashMap::new();

        let row_types = batch.column(0).as_any().downcast_ref::<Int8Array>().unwrap();
        let id_as = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let id_bs = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let meta_types = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();

        for i in 0..batch.num_rows() {
            if row_types.value(i) == 0 {
                let etype = meta_types.value(i).to_string();
                row_dict.entry(etype.clone()).or_default().push(id_as.value(i));
                col_dict.entry(etype).or_default().push(id_bs.value(i));
            }
        }

        // Verify "User__Click__Item"
        let rows = row_dict.get("User__Click__Item").unwrap();
        let cols = col_dict.get("User__Click__Item").unwrap();
        
        assert_eq!(rows.len(), 4);
        assert_eq!(rows, &vec![0, 0, 1, 1]); // Local User IDs
        assert_eq!(cols, &vec![0, 1, 1, 2]); // Local Item IDs (Note: Item 2 is shared at Local 1)
        
        // --- 5. VERIFY NODE MAPPING ---
        // Node mapping would be sent as row_type=1. 
        // User map: {100:0, 200:1}
        // Item map: {1:0, 2:1, 3:2}
        assert_eq!(local_index_maps["User"][&100], 0);
        assert_eq!(local_index_maps["Item"][&2], 1);
    }
}
