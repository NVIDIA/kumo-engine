use serde_json::Value;
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const INDEX_METADATA_FILENAME: &str = "index_metadata.json";

#[derive(Debug, Error)]
pub enum IndexMetadataError {
    #[error("invalid index metadata JSON: {0}")]
    InvalidPayload(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub fn read_index_metadata(index_path: &Path) -> Result<String, IndexMetadataError> {
    let metadata_path = index_path.join(INDEX_METADATA_FILENAME);
    let raw = fs::read_to_string(&metadata_path)?;
    let _ = serde_json::from_str::<Value>(&raw)?;
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_index_metadata_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let payload = serde_json::json!({
            "node_types": [{"name": "users", "max_id": 7}],
            "edge_relations": [],
        })
        .to_string();
        fs::write(dir.path().join(INDEX_METADATA_FILENAME), &payload).unwrap();

        let read_back = read_index_metadata(dir.path()).unwrap();

        assert_eq!(read_back, payload);
    }

    #[test]
    fn test_read_index_metadata_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_index_metadata(dir.path()).unwrap_err();
        assert!(matches!(err, IndexMetadataError::Io(_)));
    }

    #[test]
    fn test_read_index_metadata_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(INDEX_METADATA_FILENAME), "{not json").unwrap();

        let err = read_index_metadata(dir.path()).unwrap_err();
        assert!(matches!(err, IndexMetadataError::InvalidPayload(_)));
    }
}
