use super::{key_from_access_into_buf, BitWriter, OutputSink};
use crate::graph_config::{ColumnSpec, DataSource, IngestedType};
use arrow_array::{ArrayRef, Decimal128Array, Decimal256Array, Int64Array};
use diskgraph_rowpack::{BinaryStringHint, RowAccess, RowAccessHints, RowAccessStats};
use std::io::{BufWriter, Read};
use std::sync::Arc;

#[test]
fn test_negative_int64_key() {
    let col = ColumnSpec {
        name: "id".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Int64,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: None,
    };

    let val: i64 = -1;
    let array = Int64Array::from(vec![val]);
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];

    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
        binary_string_hint: BinaryStringHint::Utf8,
    }];

    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut key_buf = Vec::new();
    key_from_access_into_buf(&mut access, 0, &col, &col.ingested_dtype, &mut key_buf).unwrap();

    let expected = (val as u64).to_le_bytes();
    assert_eq!(key_buf, expected);
}

#[test]
fn test_negative_decimal128_key() {
    let col = ColumnSpec {
        name: "id".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Decimal128,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: Some(38),
        scale: Some(0),
        semantic_type: None,
    };

    let val: i128 = -1;
    let array = Decimal128Array::from_iter_values([val])
        .with_precision_and_scale(38, 0)
        .unwrap();
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];

    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: Some(0),
        binary_string_hint: BinaryStringHint::Utf8,
    }];

    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut key_buf = Vec::new();
    key_from_access_into_buf(&mut access, 0, &col, &col.ingested_dtype, &mut key_buf).unwrap();

    let expected = (val as i64 as u64).to_le_bytes();
    assert_eq!(key_buf, expected);
}

#[test]
fn test_decimal256_key_as_string() {
    let col = ColumnSpec {
        name: "id".to_string(),
        source_dtype: None,
        ingested_dtype: IngestedType::Decimal256,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: Some(76),
        scale: Some(0),
        semantic_type: None,
    };

    let val = arrow_buffer::i256::from_i128(12345);
    let array = Decimal256Array::from_iter_values([val])
        .with_precision_and_scale(76, 0)
        .unwrap();
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];

    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: Some(0),
        binary_string_hint: BinaryStringHint::Utf8,
    }];

    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    access.set_row(0);

    let mut key_buf = Vec::new();
    key_from_access_into_buf(&mut access, 0, &col, &col.ingested_dtype, &mut key_buf).unwrap();

    let expected = b"12345".to_vec();
    assert_eq!(key_buf, expected);
}

#[test]
fn test_bitwriter_packing() -> anyhow::Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let path = tmp.path().to_path_buf();

    let file = std::fs::File::create(&path)?;
    let sink = OutputSink::Local(file);
    let writer = BufWriter::new(sink);
    let mut bw = BitWriter::new(writer);

    bw.append_bits(&[0x0F], 4)?;
    bw.append_bits(&[0x0A], 4)?;

    bw.finish()?;

    let mut content = Vec::new();
    std::fs::File::open(&path)?.read_to_end(&mut content)?;

    assert_eq!(content, vec![0xAF]);
    Ok(())
}

use crate::errors::{KeyParseError, KeyParseResult};
use crate::graph_config::{TableSpec, TimestampUnit};
use crate::{TimestampEncodingKind, TimestampEncodingSpec};
use arrow_array::{BinaryArray, StringArray};
use chrono::NaiveDate;
use std::sync::atomic::AtomicU64;

// Helper from lib.rs
fn parse_uuid_bytes(bytes: &[u8]) -> KeyParseResult<[u8; 16]> {
    if bytes.len() != 16 {
        return Err(KeyParseError::InvalidUuid);
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(bytes);
    Ok(out)
}

// Tests moved from lib.rs

#[test]
fn parse_uuid_string_formats() {
    let raw = "00112233445566778899aabbccddeeff";
    let dashed = "00112233-4455-6677-8899-aabbccddeeff";
    let expected = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    use crate::parse_uuid_string;
    assert_eq!(parse_uuid_string(raw.as_bytes()).unwrap(), expected);
    assert_eq!(parse_uuid_string(dashed.as_bytes()).unwrap(), expected);
}

#[test]
fn parse_uuid256_string_formats() {
    let raw = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    let expected: [u8; 32] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
        0xee, 0xff,
    ];
    use crate::parse_uuid256_string;
    assert_eq!(parse_uuid256_string(raw.as_bytes()).unwrap(), expected);
}

#[test]
fn parse_ip_string_formats() {
    let v4 = "192.168.1.2";
    let v6 = "2001:db8::1";
    let v4_expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 1, 2];
    let v6_expected = [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
    use crate::parse_ip_string;
    assert_eq!(parse_ip_string(v4.as_bytes()).unwrap(), v4_expected);
    assert_eq!(parse_ip_string(v6.as_bytes()).unwrap(), v6_expected);
}

#[test]
fn parse_uuid_bytes_requires_len_16() {
    assert!(parse_uuid_bytes(&[0u8; 15]).is_err());
    assert!(parse_uuid_bytes(&[0u8; 17]).is_err());
    assert!(parse_uuid_bytes(&[0u8; 16]).is_ok());
}

fn single_col_table(ingested: IngestedType) -> TableSpec {
    TableSpec {
        name: "sample".to_string(),
        columns: vec![ColumnSpec {
            name: "id".to_string(),
            ingested_dtype: ingested,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    }
}

fn with_access(table: &TableSpec, cols: Vec<ArrayRef>, f: impl FnOnce(&mut RowAccess<'_>)) {
    let trimmed = std::sync::Arc::new(AtomicU64::new(0));
    use crate::build_row_access_hints;
    let hints = build_row_access_hints(table);
    let stats = RowAccessStats::new(Some(trimmed.as_ref()));
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&cols);
    access.set_row(0);
    f(&mut access);
}

fn days_from_secs(secs: i64) -> i32 {
    secs.div_euclid(86_400) as i32
}

#[test]
fn access_uuid_from_uuid_string_source() {
    let table = single_col_table(IngestedType::Uuid);
    let text = "00112233445566778899aabbccddeeff";
    let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
    use crate::parse_uuid_string;
    let expected = parse_uuid_string(text.as_bytes()).unwrap();
    with_access(&table, cols, |access| {
        let uuid = access.get_uuid(0).unwrap().unwrap();
        assert_eq!(uuid, expected);
    });
}

#[test]
fn access_uuid_from_string_to_uuid() {
    let table = single_col_table(IngestedType::Uuid);
    let text = "00112233445566778899aabbccddeeff";
    let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
    use crate::parse_uuid_string;
    let expected = parse_uuid_string(text.as_bytes()).unwrap();
    with_access(&table, cols, |access| {
        let uuid = access.get_uuid(0).unwrap().unwrap();
        assert_eq!(uuid, expected);
    });
}

#[test]
fn access_uuid256_from_string_to_uuid256() {
    let table = single_col_table(IngestedType::Uuid256);
    let text = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
    use crate::parse_uuid256_string;
    let expected = parse_uuid256_string(text.as_bytes()).unwrap();
    with_access(&table, cols, |access| {
        let uuid = access.get_uuid256(0).unwrap().unwrap();
        assert_eq!(uuid, expected);
    });
}

#[test]
fn access_ip_from_ip_string_source() {
    let table = single_col_table(IngestedType::Ip);
    let text = "192.168.1.2";
    let cols: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text]))];
    use crate::parse_ip_string;
    let expected = parse_ip_string(text.as_bytes()).unwrap();
    with_access(&table, cols, |access| {
        let ip = access.get_ip(0).unwrap().unwrap();
        assert_eq!(ip, expected);
    });
}

#[test]
fn convert_string_timestamp_to_rowpack_timestamp() {
    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "created_at".to_string(),
            ingested_dtype: IngestedType::Timestamp,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };
    let arr = StringArray::from(vec!["2020-01-02T03:04:05Z"]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let secs = access.get_timestamp(0).unwrap().unwrap();
        assert_eq!(secs, 1_577_934_245);
    });
}

#[test]
fn convert_int_timestamp_with_unit() {
    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "created_at".to_string(),
            ingested_dtype: IngestedType::Timestamp,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: Some(TimestampUnit::Millis),
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };
    let arr = Int64Array::from(vec![1_500_000_i64]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let secs = access.get_timestamp(0).unwrap().unwrap();
        assert_eq!(secs, 1500);
    });
}

#[test]
fn encode_timestamp_int8() {
    let encoding = TimestampEncodingSpec {
        kind: TimestampEncodingKind::Int8,
        anchor: 0,
        step: 1,
    };
    use crate::{encode_timestamp_value, ordered_signed_to_u64};
    let encoded = encode_timestamp_value(Some(10), &encoding);
    assert_eq!(encoded, ordered_signed_to_u64(10, 8));
    let encoded_neg = encode_timestamp_value(Some(-1), &encoding);
    assert_eq!(encoded_neg, ordered_signed_to_u64(-1, 8));
    let encoded_clamp = encode_timestamp_value(Some(1000), &encoding);
    assert_eq!(encoded_clamp, ordered_signed_to_u64(i8::MAX as i64, 8));
}

#[test]
fn convert_string_date_to_rowpack_date() {
    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "created_date".to_string(),
            ingested_dtype: IngestedType::Date,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: Some("%Y-%m-%d".to_string()),
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };
    let arr = StringArray::from(vec!["2024-02-03"]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let days = access.get_date(0).unwrap().unwrap();
        let expected = days_from_secs(
            NaiveDate::from_ymd_opt(2024, 2, 3)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
                .and_utc()
                .timestamp(),
        );
        assert_eq!(days, expected);
    });
}

#[test]
fn convert_numeric_to_string_for_rowpack() {
    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "id".to_string(),
            ingested_dtype: IngestedType::String,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };
    let arr = Int64Array::from(vec![42_i64]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let value = access
            .with_string(0, |v| Ok(v.to_string()))
            .unwrap()
            .unwrap();
        assert_eq!(value, "42");
    });
}

#[test]
fn convert_bytes_to_int_for_rowpack() {
    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "id".to_string(),
            ingested_dtype: IngestedType::Int32,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };
    let arr = BinaryArray::from(vec![b"123".as_slice()]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let value = access.get_int32(0).unwrap().unwrap();
        assert_eq!(value, 123);
    });
}

#[test]
fn pk_map_estimate_scales_with_rows() {
    use crate::mapping::KeyType;
    use crate::scratch::estimate_pk_map_bytes;
    let small = estimate_pk_map_bytes(1_000, KeyType::Int64, 0);
    let large = estimate_pk_map_bytes(2_000, KeyType::Int64, 0);
    assert!(large > small);
}

#[test]
fn pk_map_estimate_varbytes_respects_avg_len() {
    use crate::mapping::KeyType;
    use crate::scratch::estimate_pk_map_bytes;
    let small = estimate_pk_map_bytes(10_000, KeyType::VarBytes, 8);
    let large = estimate_pk_map_bytes(10_000, KeyType::VarBytes, 64);
    assert!(large > small);
}

#[test]
fn shard_assignment_consistent_for_same_pk() {
    // Verifies that same PK bytes always produce the same shard, enabling per-shard deduplication.
    use crate::shard_for_key;
    use std::collections::HashMap;

    let num_shards = 100;
    let num_unique_pks = 100_000;
    let num_rows = 500_000; // 5x rows to PKs guarantees duplicates

    // Simple xorshift PRNG for deterministic test
    let mut state: u64 = 12345;
    let mut next_rand = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut pk_to_shard: HashMap<String, usize> = HashMap::new();
    let mut inconsistencies = 0;

    for _ in 0..num_rows {
        let pk_num = next_rand() % num_unique_pks;
        let pk = format!("PK_{:07}", pk_num);
        let shard = shard_for_key(pk.as_bytes(), num_shards);

        if let Some(&prev_shard) = pk_to_shard.get(&pk) {
            if prev_shard != shard {
                inconsistencies += 1;
            }
        } else {
            pk_to_shard.insert(pk, shard);
        }
    }

    assert_eq!(inconsistencies, 0, "Same PK must always map to same shard");
    assert!(
        pk_to_shard.len() > 1,
        "Test should have multiple unique PKs"
    );

    // Verify shards are reasonably distributed (not all in one shard)
    let mut shard_counts = vec![0usize; num_shards];
    for &shard in pk_to_shard.values() {
        shard_counts[shard] += 1;
    }
    let non_empty_shards = shard_counts.iter().filter(|&&c| c > 0).count();
    assert!(
        non_empty_shards > num_shards / 2,
        "PKs should distribute across shards"
    );
}

#[test]
fn test_precompute_shards_consistency() {
    use crate::graph_config::{ColumnSpec, IngestedType};
    use crate::{precompute_shards, shard_for_key};
    use arrow_array::{FixedSizeBinaryArray, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let num_shards = 10;

    // 1. Test Int64
    {
        let values = vec![Some(1), Some(-1), None, Some(123456789)];
        let array = Int64Array::from(values.clone());
        let schema = Arc::new(Schema::new(vec![Field::new("pk", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let pk_col = ColumnSpec {
            name: "pk".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Int64,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        };

        let precomputed =
            precompute_shards(Some(batch.column(0)), Some(&pk_col), num_shards).unwrap();
        for (i, val) in values.iter().enumerate() {
            match val {
                Some(v) => {
                    let expected_shard = shard_for_key(&(*v as u64).to_le_bytes(), num_shards);
                    assert_eq!(precomputed.shards[i], Some(expected_shard));
                }
                None => {
                    assert_eq!(precomputed.shards[i], None);
                }
            }
        }
    }

    // 2. Test FixedSizeBinary(16)
    {
        let mut values = vec![None];
        let b1 = [1u8; 16];
        let b2 = [2u8; 16];
        values.push(Some(&b1[..]));
        values.push(Some(&b2[..]));

        let array =
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(values.clone().into_iter(), 16)
                .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "pk",
            DataType::FixedSizeBinary(16),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let pk_col = ColumnSpec {
            name: "pk".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Uuid,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        };

        let precomputed =
            precompute_shards(Some(batch.column(0)), Some(&pk_col), num_shards).unwrap();

        for (i, val) in values.iter().enumerate() {
            match val {
                Some(v) => {
                    let expected_shard = shard_for_key(v, num_shards);
                    assert_eq!(precomputed.shards[i], Some(expected_shard));
                }
                None => {
                    assert_eq!(precomputed.shards[i], None);
                }
            }
        }
    }
}

#[test]
fn test_precompute_shards_casting() {
    use crate::graph_config::{ColumnSpec, IngestedType};
    use crate::mapping::Key;
    use crate::{precompute_shards, shard_for_key};
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let num_shards = 10;

    // 1. Test Int64 -> Int32 Casting
    {
        let values = vec![Some(1), Some(-1), None, Some(100)];
        let array = Int64Array::from(values.clone());
        let schema = Arc::new(Schema::new(vec![Field::new("pk", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let pk_col = ColumnSpec {
            name: "pk".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Int32,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        };

        let precomputed =
            precompute_shards(Some(batch.column(0)), Some(&pk_col), num_shards).unwrap();
        assert!(precomputed.keys.is_some());
        let keys = precomputed.keys.unwrap();

        for (i, val) in values.iter().enumerate() {
            match val {
                Some(v) => {
                    let casted = *v as i32 as u64;
                    let expected_shard = shard_for_key(&casted.to_le_bytes(), num_shards);
                    assert_eq!(precomputed.shards[i], Some(expected_shard));
                    assert_eq!(keys[i], Some(Key::Int(casted)));
                }
                None => {
                    assert_eq!(precomputed.shards[i], None);
                    assert_eq!(keys[i], None);
                }
            }
        }
    }

    // 2. Test Utf8 -> Uuid Parsing
    {
        let uuid1 = "00112233-4455-6677-8899-aabbccddeeff";
        let uuid2 = "00112233445566778899aabbccddeeff";
        let values = vec![Some(uuid1), Some(uuid2), None];
        let array = StringArray::from(values.clone());
        let schema = Arc::new(Schema::new(vec![Field::new("pk", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let pk_col = ColumnSpec {
            name: "pk".to_string(),
            source_dtype: None,
            ingested_dtype: IngestedType::Uuid,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        };

        let precomputed =
            precompute_shards(Some(batch.column(0)), Some(&pk_col), num_shards).unwrap();
        assert!(precomputed.keys.is_some());
        let keys = precomputed.keys.unwrap();

        use crate::parse_uuid_string;
        for (i, val) in values.iter().enumerate() {
            match val {
                Some(v) => {
                    let parsed = parse_uuid_string(v.as_bytes()).unwrap();
                    let expected_shard = shard_for_key(&parsed, num_shards);
                    assert_eq!(precomputed.shards[i], Some(expected_shard));
                    assert_eq!(keys[i], Some(Key::Uuid(parsed)));
                }
                None => {
                    assert_eq!(precomputed.shards[i], None);
                    assert_eq!(keys[i], None);
                }
            }
        }
    }
}

#[test]
fn test_extract_batch_keys_resilience() {
    use crate::mapping::Key;
    use crate::{extract_batch_keys, EdgeKeyFastKind};
    use arrow_array::{ArrayRef, StringArray};
    use std::sync::Arc;

    // Test String -> Int64 with some invalid rows
    let values = vec![Some("123"), Some("not-a-number"), Some("456"), None];
    let array = Arc::new(StringArray::from(values)) as ArrayRef;

    let extracted = extract_batch_keys(&array, EdgeKeyFastKind::StringToInt64).unwrap();

    assert_eq!(extracted.len(), 4);
    assert_eq!(extracted[0], Some(Key::Int(123)));
    assert_eq!(extracted[1], None); // "not-a-number" -> None
    assert_eq!(extracted[2], Some(Key::Int(456)));
    assert_eq!(extracted[3], None); // NULL -> None
}

#[test]
fn test_extract_batch_keys_edge_cases() {
    use crate::mapping::Key;
    use crate::parse_uuid_string;
    use crate::{extract_batch_keys, EdgeKeyFastKind};
    use arrow_array::{ArrayRef, BinaryArray, Int64Array, StringArray};
    use std::sync::Arc;

    // 1. String -> UUID (Edge Case)
    {
        let uuid_str = "00112233-4455-6677-8899-aabbccddeeff";
        let expected_uuid = parse_uuid_string(uuid_str.as_bytes()).unwrap();
        let values = vec![Some(uuid_str), None, Some("invalid-uuid")];
        let array = Arc::new(StringArray::from(values)) as ArrayRef;

        let extracted = extract_batch_keys(&array, EdgeKeyFastKind::StringUuid).unwrap();

        assert_eq!(extracted.len(), 3);
        assert_eq!(extracted[0], Some(Key::Uuid(expected_uuid)));
        assert_eq!(extracted[1], None);
        assert_eq!(extracted[2], None); // Resilience check
    }

    // 2. Int64 -> Int32 Casting
    {
        let val = 12345_i64;
        let val_large = i32::MAX as i64 + 1; // Wraps when cast to i32
        let values = vec![Some(val), None, Some(val_large)];
        let array = Arc::new(Int64Array::from(values)) as ArrayRef;

        let extracted = extract_batch_keys(&array, EdgeKeyFastKind::Int64ToInt32).unwrap();

        assert_eq!(extracted.len(), 3);
        assert_eq!(extracted[0], Some(Key::Int(val as i32 as u64)));
        assert_eq!(extracted[1], None);
        assert_eq!(extracted[2], Some(Key::Int(val_large as i32 as u64)));
    }

    // 3. Bytes -> Bytes (Var)
    {
        let val = b"hello";
        let values = vec![Some(val.as_slice()), None];
        let array = Arc::new(BinaryArray::from(values)) as ArrayRef;

        let extracted = extract_batch_keys(&array, EdgeKeyFastKind::BytesVar).unwrap();

        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0], Some(Key::Bytes(val.to_vec())));
        assert_eq!(extracted[1], None);
    }
}

// ---------------------------------------------------------------------------
// Integration: Int32 YYYYMMDD → get_timestamp via timestamp_format
// ---------------------------------------------------------------------------

#[test]
fn int32_yyyymmdd_create_time_via_format() {
    use crate::graph_config::{SemanticType, TableSpec};
    use arrow_array::Int32Array;

    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "event_date".to_string(),
            ingested_dtype: IngestedType::Int32,
            source_dtype: Some("int".to_string()),
            fixed_len: None,
            timestamp_format: Some("%Y%m%d".to_string()),
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: Some(SemanticType::Timestamp),
        }],
        primary_key: None,
        create_time: Some("event_date".to_string()),
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };

    let arr = Int32Array::from(vec![20240301]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let secs = access.get_timestamp(0).unwrap().unwrap();
        let expected = NaiveDate::from_ymd_opt(2024, 3, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(secs, expected, "Int32 YYYYMMDD should parse via format hint");
    });
}

#[test]
fn int32_epoch_secs_no_format_unchanged() {
    use arrow_array::Int32Array;

    let table = TableSpec {
        name: "events".to_string(),
        columns: vec![ColumnSpec {
            name: "ts".to_string(),
            ingested_dtype: IngestedType::Int32,
            source_dtype: None,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
        }],
        primary_key: None,
        create_time: Some("ts".to_string()),
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
    };

    // Normal epoch seconds — no format hint, should pass through as-is
    let epoch = 1709251200i32; // 2024-03-01 00:00:00 UTC
    let arr = Int32Array::from(vec![epoch]);
    let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
    with_access(&table, cols, |access| {
        let secs = access.get_timestamp(0).unwrap().unwrap();
        assert_eq!(secs, epoch as i64, "Epoch seconds should pass through unchanged");
    });
}
