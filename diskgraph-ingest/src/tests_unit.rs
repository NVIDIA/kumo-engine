use super::{key_from_access_into_buf, BitWriter, OutputSink};
use crate::source_metadata::{ColumnSpec, IngestedType};
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
