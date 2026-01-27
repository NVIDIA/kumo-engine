use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    #[serde(default)]
    pub errors: Vec<ValidationMessage>,
    #[serde(default)]
    pub warnings: Vec<ValidationMessage>,
    pub stats: Option<ValidationStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ValidationStatus {
    Ok,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMessage {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub rows: u64,
}

impl ValidationReport {
    pub fn ok() -> Self {
        Self {
            status: ValidationStatus::Ok,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: None,
        }
    }

    pub fn error(errors: Vec<ValidationMessage>) -> Self {
        Self {
            status: ValidationStatus::Error,
            errors,
            warnings: Vec::new(),
            stats: None,
        }
    }

    pub fn with_stats(mut self, rows: u64) -> Self {
        self.stats = Some(ValidationStats { rows });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_ok_default() {
        let report = ValidationReport::ok();
        assert_eq!(report.status, ValidationStatus::Ok);
        assert!(report.errors.is_empty());
    }
}
