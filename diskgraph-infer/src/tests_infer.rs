use super::{map_kumo_dtype, unify_types, IngestedType};

#[test]
fn test_map_kumo_dtype() {
    assert_eq!(map_kumo_dtype("int"), Some(IngestedType::Int64));
    assert_eq!(map_kumo_dtype("float"), Some(IngestedType::Float32));
    assert_eq!(map_kumo_dtype("string"), Some(IngestedType::String));
    assert_eq!(map_kumo_dtype("timedelta"), Some(IngestedType::Int64));
    assert_eq!(map_kumo_dtype("unknown"), None);
}

#[test]
fn test_unify_types() {
    use IngestedType::*;
    // Equal
    assert_eq!(unify_types(&Int64, &Int64), Some(Int64));
    
    // String vs Uuid/Ip
    assert_eq!(unify_types(&String, &Uuid), Some(String));
    assert_eq!(unify_types(&Uuid, &String), Some(String));
    assert_eq!(unify_types(&String, &Ip), Some(String));
    
    // Bytes vs Uuid/Ip
    assert_eq!(unify_types(&Bytes, &Uuid), Some(Bytes));
    
    // Int
    assert_eq!(unify_types(&Int64, &Int32), Some(Int64));
    assert_eq!(unify_types(&Int32, &Int64), Some(Int64));
    
    // Mismatch
    assert_eq!(unify_types(&Int64, &String), None);
}
