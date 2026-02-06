use super::{RowAccess, RowAccessStats, RowAccessHints};
use arrow_array::{Float64Array, BooleanArray, Decimal128Array, ArrayRef};
use std::sync::Arc;

#[test]
fn test_cast_float_to_int() {
    let array = Float64Array::from(vec![1.2, 3.9, -2.5]);
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
    ..Default::default()
}];
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    
    access.set_row(0);
    assert_eq!(access.get_int64(0).unwrap(), Some(1));
    
    access.set_row(1);
    assert_eq!(access.get_int64(0).unwrap(), Some(3)); // Truncate
    
    access.set_row(2);
    assert_eq!(access.get_int64(0).unwrap(), Some(-2));
}

#[test]
fn test_cast_bool_to_int_float() {
    let array = BooleanArray::from(vec![true, false]);
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: None,
    ..Default::default()
}];
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    
    // True
    access.set_row(0);
    assert_eq!(access.get_int64(0).unwrap(), Some(1));
    assert_eq!(access.get_float64(0).unwrap(), Some(1.0));
    
    // False
    access.set_row(1);
    assert_eq!(access.get_int64(0).unwrap(), Some(0));
    assert_eq!(access.get_float64(0).unwrap(), Some(0.0));
}

#[test]
fn test_cast_decimal_to_string_scaled() {
    // 12345, scale 2 -> "123.45"
    let array = Decimal128Array::from_iter_values([12345])
        .with_precision_and_scale(10, 2).unwrap();
    let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
    let stats = RowAccessStats::default();
    let hints = vec![RowAccessHints {
        timestamp_format: None,
        timestamp_unit: None,
        decimal_scale: Some(2),
    ..Default::default()
}];
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    
    access.set_row(0);
    let s = access.with_string(0, |s| Ok(s.to_string())).unwrap().unwrap();
    assert_eq!(s, "123.45");
}

/// Tests verifying that RowAccess handles cross-type reads correctly.
/// These are the conversions that the "clone config + resolve FK types" approach
/// would rely on: when col.ingested_dtype is changed to the PK type but the
/// Arrow data remains in the original FK type.
#[cfg(test)]
mod clone_approach_verification {
    use super::*;
    use arrow_array::{Int32Array, StringArray, UInt64Array, FixedSizeBinaryArray};

    fn make_access(arrays: &[ArrayRef]) -> RowAccess<'static> {
        let hints = (0..arrays.len())
            .map(|_| RowAccessHints {
                timestamp_format: None,
                timestamp_unit: None,
                decimal_scale: None,
            ..Default::default()
})
            .collect();
        let mut access = RowAccess::new(hints, RowAccessStats::default());
        access.set_cols(arrays);
        access.set_row(0);
        access
    }

    // === Int32 Arrow data, FK resolved to Int64 (PK type) ===
    // key_from_access_into_buf would call get_int64() on Int32 array
    #[test]
    fn int32_arrow_read_as_int64() {
        let arrays: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![42, -7, i32::MAX]))];
        let mut access = make_access(&arrays);

        access.set_row(0);
        assert_eq!(access.get_int64(0).unwrap(), Some(42i64));
        access.set_row(1);
        assert_eq!(access.get_int64(0).unwrap(), Some(-7i64));
        access.set_row(2);
        assert_eq!(access.get_int64(0).unwrap(), Some(i32::MAX as i64));
    }

    // === Int32 Arrow data, FK resolved to String (PK type) ===
    // key_from_access_into_buf would call with_string() on Int32 array
    #[test]
    fn int32_arrow_read_as_string() {
        let arrays: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![42, -7, 0]))];
        let mut access = make_access(&arrays);

        access.set_row(0);
        let s = access.with_string(0, |s| Ok(s.to_string())).unwrap().unwrap();
        assert_eq!(s, "42");
        access.set_row(1);
        let s = access.with_string(0, |s| Ok(s.to_string())).unwrap().unwrap();
        assert_eq!(s, "-7");
    }

    // === Utf8 Arrow data, FK resolved to Int64 (PK type) ===
    // key_from_access_into_buf would call get_int64() on Utf8 array
    #[test]
    fn utf8_arrow_read_as_int64() {
        let arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec!["42", "-7", "0"]))];
        let mut access = make_access(&arrays);

        access.set_row(0);
        assert_eq!(access.get_int64(0).unwrap(), Some(42i64));
        access.set_row(1);
        assert_eq!(access.get_int64(0).unwrap(), Some(-7i64));
    }

    // === UInt64 Arrow data, FK resolved to Int64 (PK type) ===
    #[test]
    fn uint64_arrow_read_as_int64() {
        let arrays: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(vec![42u64, 0]))];
        let mut access = make_access(&arrays);

        access.set_row(0);
        assert_eq!(access.get_int64(0).unwrap(), Some(42i64));
    }

    // === UInt64 Arrow data, FK resolved to String (PK type) ===
    #[test]
    fn uint64_arrow_read_as_string() {
        let arrays: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(vec![42u64, 999]))];
        let mut access = make_access(&arrays);

        access.set_row(0);
        let s = access.with_string(0, |s| Ok(s.to_string())).unwrap().unwrap();
        assert_eq!(s, "42");
    }

    // === FixedSizeBinary(16) Arrow data (UUID stored as binary) ===
    // If FK resolved to String, with_string() on binary — does it work?
    #[test]
    fn uuid_binary_read_as_string() {
        // UUID raw bytes (16 bytes)
        let uuid_bytes: [u8; 16] = [0xc0, 0x99, 0x3c, 0xa4, 0x28, 0x04, 0x42, 0x56,
                                     0xa3, 0xb7, 0xba, 0x9b, 0x3a, 0xb7, 0x6f, 0xce];
        let array = FixedSizeBinaryArray::try_from_iter(
            vec![uuid_bytes.as_slice()].into_iter()
        ).unwrap();
        let arrays: Vec<ArrayRef> = vec![Arc::new(array)];
        let mut access = make_access(&arrays);

        // with_string on FixedSizeBinary — this reads raw bytes as UTF-8
        // Raw UUID bytes are NOT valid UTF-8, so this should fail or produce garbage
        let result = access.with_string(0, |s| Ok(s.to_string()));
        // Document what actually happens:
        match result {
            Ok(Some(s)) => println!("uuid binary -> string succeeded: {:?}", s),
            Ok(None) => println!("uuid binary -> string returned None"),
            Err(e) => println!("uuid binary -> string failed: {}", e),
        }
    }

    // === Utf8 Arrow data (UUID stored as hex string) ===
    // If FK resolved to UUID, get_uuid() on Utf8 — does it work?
    #[test]
    fn uuid_string_read_as_uuid() {
        // UUID as hyphenated string
        let arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(
            vec!["c0993ca4-2804-4256-a3b7-ba9b3ab76fce"]
        ))];
        let mut access = make_access(&arrays);

        let result = access.get_uuid(0);
        assert!(result.is_ok(), "get_uuid on UUID string should succeed");
        assert!(result.unwrap().is_some());
    }

    // === Utf8 Arrow data (UUID stored as hex string, no hyphens) ===
    // This is the actual format in the benchmark data
    #[test]
    fn uuid_hex_string_read_as_uuid() {
        let arrays: Vec<ArrayRef> = vec![Arc::new(StringArray::from(
            vec!["c0993ca428044256a3b7ba9b3ab76fce"]
        ))];
        let mut access = make_access(&arrays);

        let result = access.get_uuid(0);
        match &result {
            Ok(Some(v)) => println!("uuid hex -> uuid: {:02x?}", v),
            Ok(None) => println!("uuid hex -> uuid: None"),
            Err(e) => println!("uuid hex -> uuid failed: {}", e),
        }
        // Document if non-hyphenated UUID strings work
        assert!(result.is_ok() && result.unwrap().is_some(),
            "get_uuid should handle non-hyphenated UUID hex strings");
    }
}
