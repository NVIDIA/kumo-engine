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
    }];
    let mut access = RowAccess::new(hints, stats);
    access.set_cols(&arrays);
    
    access.set_row(0);
    let s = access.with_string(0, |s| Ok(s.to_string())).unwrap().unwrap();
    assert_eq!(s, "123.45");
}
