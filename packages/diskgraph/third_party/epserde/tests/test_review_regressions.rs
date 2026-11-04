// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
use epserde::prelude::*;
use std::ops::ControlFlow;

#[test]
fn control_flow_round_trip() {
    for value in [ControlFlow::<u64, u64>::Break(17), ControlFlow::Continue(29)] {
        let mut bytes = AlignedCursor::<A64>::new();
        value.serialize(&mut bytes).unwrap();
        assert_eq!(ControlFlow::<u64, u64>::deserialize_full(&mut std::io::Cursor::new(bytes.as_bytes())).unwrap(), value);
        assert_eq!(ControlFlow::<u64, u64>::deserialize_eps(bytes.as_bytes()).unwrap(), value);
    }
}

#[test]
fn truncated_payloads_return_errors() {
    let value = vec![1_u64, 2, 3];
    let mut bytes = AlignedCursor::<A64>::new();
    value.serialize(&mut bytes).unwrap();
    for end in 0..bytes.as_bytes().len() {
        assert!(Vec::<u64>::deserialize_eps(&bytes.as_bytes()[..end]).is_err());
    }
}

#[test]
fn load_mem_rejects_empty_and_truncated_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    for bytes in [vec![], vec![0; 8], vec![0; 64]] {
        std::fs::write(&path, bytes).unwrap();
        assert!(u64::load_mem(&path).is_err());
    }
}

#[test]
fn load_mem_does_not_treat_padding_as_file_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index");
    let mut bytes = AlignedCursor::<A64>::new();
    42_u64.serialize(&mut bytes).unwrap();
    for end in 0..bytes.as_bytes().len() {
        std::fs::write(&path, &bytes.as_bytes()[..end]).unwrap();
        assert!(u64::load_mem(&path).is_err(), "accepted prefix {end}");
        assert!(u64::load_mmap(&path, Flags::empty()).is_err());
        assert!(u64::mmap(&path, Flags::empty()).is_err());
    }
    std::fs::write(&path, bytes.as_bytes()).unwrap();
    assert_eq!(*u64::load_mem(&path).unwrap(), 42);
}
