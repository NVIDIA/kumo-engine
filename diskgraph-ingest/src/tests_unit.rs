use super::{key_from_access_into_buf, BitWriter, OutputSink};
use crate::graph_config::{ColumnSpec, IngestedType};
use diskgraph_rowpack::{RowAccess, RowAccessStats, RowAccessHints};
use arrow_array::{Decimal128Array, Int64Array, ArrayRef, Decimal256Array};
use std::sync::Arc;
use std::io::{Read, BufWriter};

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
    };
    
    let val: i64 = -1;
    let array = Int64Array::from(vec![val]);
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
    
    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: None,
    }];
    
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    access.set_row(0);
    
    let mut key_buf = Vec::new();
    key_from_access_into_buf(&mut access, 0, &col, &mut key_buf).unwrap();
    
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
    };
    
    let val: i128 = -1;
    let array = Decimal128Array::from_iter_values([val])
        .with_precision_and_scale(38, 0).unwrap();
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
    
    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: Some(0),
    }];
    
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    access.set_row(0);
    
    let mut key_buf = Vec::new();
    key_from_access_into_buf(&mut access, 0, &col, &mut key_buf).unwrap();
    
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
    };
    
    let val = arrow_buffer::i256::from_i128(12345);
    let array = Decimal256Array::from_iter_values([val])
        .with_precision_and_scale(76, 0).unwrap();
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
    
    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
            timestamp_format: None,
            timestamp_unit: None,
            decimal_scale: Some(0),
    }];
    
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    access.set_row(0);
    
    let mut key_buf = Vec::new();
    key_from_access_into_buf(&mut access, 0, &col, &mut key_buf).unwrap();
    
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

use crate::graph_config::TimestampUnit;
use crate::graph_config::TableSpec;
use crate::graph_config::TableFormat;
use crate::graph_config::TimestampEncodingSpec;
use crate::graph_config::TimestampEncodingKind;
use crate::errors::{KeyParseResult, KeyParseError};
use chrono::NaiveDate;
use arrow_array::{BinaryArray, StringArray};
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
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff,
        ];
        use crate::parse_uuid_string;
        assert_eq!(parse_uuid_string(raw.as_bytes()).unwrap(), expected);
        assert_eq!(parse_uuid_string(dashed.as_bytes()).unwrap(), expected);
    }

    #[test]
    fn parse_uuid256_string_formats() {
        let raw = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        let expected: [u8; 32] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        use crate::parse_uuid256_string;
        assert_eq!(parse_uuid256_string(raw.as_bytes()).unwrap(), expected);
    }

    #[test]
    fn parse_ip_string_formats() {
        let v4 = "192.168.1.2";
        let v6 = "2001:db8::1";
        let v4_expected = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 1, 2,
        ];
        let v6_expected = [
            0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
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
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: ingested,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: None,
            fkeys: Vec::new(),
        }
    }

    fn with_access(
        table: &TableSpec,
        cols: Vec<ArrayRef>,
        f: impl FnOnce(&mut RowAccess<'_>),
    ) {
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
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_at".to_string(),
                ingested_dtype: IngestedType::Timestamp,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: None,
            fkeys: Vec::new(),
        };
        let arr = StringArray::from(vec!["2020-01-02T03:04:05Z"]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let (secs, nanos) = access.get_timestamp(0).unwrap().unwrap();
            assert_eq!(nanos, 0);
            assert_eq!(secs, 1_577_934_245);
        });
    }

    #[test]
    fn convert_int_timestamp_with_unit() {
        let table = TableSpec {
            name: "events".to_string(),
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_at".to_string(),
                ingested_dtype: IngestedType::Timestamp,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: Some(TimestampUnit::Millis),
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: None,
            fkeys: Vec::new(),
        };
        let arr = Int64Array::from(vec![1_500_000_i64]);
        let cols: Vec<ArrayRef> = vec![Arc::new(arr)];
        with_access(&table, cols, |access| {
            let (secs, nanos) = access.get_timestamp(0).unwrap().unwrap();
            assert_eq!(secs, 1500);
            assert_eq!(nanos, 0);
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
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "created_date".to_string(),
                ingested_dtype: IngestedType::Date,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: Some("%Y-%m-%d".to_string()),
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: None,
            fkeys: Vec::new(),
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
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: None,
            fkeys: Vec::new(),
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
            format: TableFormat::Parquet,
            paths: vec![],
            columns: vec![ColumnSpec {
                name: "id".to_string(),
                ingested_dtype: IngestedType::Int32,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: None,
            fkeys: Vec::new(),
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
        use crate::scratch::estimate_pk_map_bytes;
        use crate::mapping::KeyType;
        let small = estimate_pk_map_bytes(1_000, KeyType::Int64, 0);
        let large = estimate_pk_map_bytes(2_000, KeyType::Int64, 0);
        assert!(large > small);
    }

    #[test]
    fn pk_map_estimate_varbytes_respects_avg_len() {
        use crate::scratch::estimate_pk_map_bytes;
        use crate::mapping::KeyType;
        let small = estimate_pk_map_bytes(10_000, KeyType::VarBytes, 8);
        let large = estimate_pk_map_bytes(10_000, KeyType::VarBytes, 64);
        assert!(large > small);
    }