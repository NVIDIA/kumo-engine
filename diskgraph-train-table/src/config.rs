use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const FRACTION_EPSILON: f64 = 1e-6;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid split fractions: {0}")]
    InvalidFractions(String),
    #[error("invalid split config: {0}")]
    InvalidSplit(String),
    #[error("failed to read config: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobConfig {
    pub job_name: Option<String>,
    pub index_path: String,
    pub query: String,
    pub output_parquet: String,
    pub split: SplitConfig,
    #[serde(default)]
    pub options: QueryOptions,
    pub validation_output: Option<String>,
}

impl JobConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let contents = fs::read_to_string(path)?;
        let cfg: JobConfig = serde_json::from_str(&contents)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_str(contents: &str) -> Result<Self, ConfigError> {
        let cfg: JobConfig = serde_json::from_str(contents)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.index_path.is_empty() {
            return Err(ConfigError::MissingField("index_path"));
        }
        if self.query.is_empty() {
            return Err(ConfigError::MissingField("query"));
        }
        if self.output_parquet.is_empty() {
            return Err(ConfigError::MissingField("output_parquet"));
        }
        self.split.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    #[serde(default = "default_forecast_length")]
    pub forecast_length: i64,
    pub lag_timesteps: Option<i64>,
    #[serde(default)]
    pub year_over_year: bool,
    #[serde(default = "default_list_distinct_target_is_array")]
    pub list_distinct_target_is_array: bool,
    #[serde(default = "default_drop_earliest_timeframe")]
    pub drop_earliest_timeframe: bool,
    #[serde(default)]
    pub allow_empty_target: bool,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            forecast_length: default_forecast_length(),
            lag_timesteps: None,
            year_over_year: false,
            list_distinct_target_is_array: default_list_distinct_target_is_array(),
            drop_earliest_timeframe: default_drop_earliest_timeframe(),
            allow_empty_target: false,
        }
    }
}

fn default_forecast_length() -> i64 {
    1
}

fn default_list_distinct_target_is_array() -> bool {
    true
}

fn default_drop_earliest_timeframe() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitType {
    Time,
    Random,
    DateOffset,
    Range,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    #[serde(rename = "type")]
    pub split_type: SplitType,
    pub fractions: Vec<f64>,
    pub seed: Option<u64>,
    pub time: Option<SplitTimeConfig>,
    pub range: Option<SplitRangeConfig>,
}

impl SplitConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.fractions.len() != 3 {
            return Err(ConfigError::InvalidFractions(
                "fractions must have length 3".to_string(),
            ));
        }
        let sum: f64 = self.fractions.iter().sum();
        if (sum - 1.0).abs() > FRACTION_EPSILON {
            return Err(ConfigError::InvalidFractions(format!(
                "fractions must sum to 1.0, got {sum}"
            )));
        }

        match self.split_type {
            SplitType::Time | SplitType::DateOffset => {
                if self.time.is_none() {
                    return Err(ConfigError::InvalidSplit(
                        "time split requires time config".to_string(),
                    ));
                }
            }
            SplitType::Range => {
                if self.range.is_none() {
                    return Err(ConfigError::InvalidSplit(
                        "range split requires range config".to_string(),
                    ));
                }
            }
            SplitType::Random => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitTimeConfig {
    pub anchor: String,
    pub timeframe: String,
    pub anchor_start_offset: String,
    pub timeframe_step: String,
    pub train_start_offset: Option<String>,
    pub train_end_offset: Option<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRangeConfig {
    pub unit: String,
    pub ranges: Vec<SplitRangeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitRangeEntry {
    pub start: String,
    pub end: String,
    pub label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_random_split() {
        let json = r#"{
            "index_path": "/tmp/index",
            "query": "PREDICT COUNT(x) FOR EACH y",
            "output_parquet": "/tmp/out.parquet",
            "split": {
                "type": "random",
                "fractions": [0.8, 0.1, 0.1]
            }
        }"#;
        let cfg = JobConfig::from_str(json).expect("config should parse");
        assert_eq!(cfg.index_path, "/tmp/index");
        assert_eq!(cfg.options.forecast_length, 1);
        assert!(cfg.options.list_distinct_target_is_array);
    }

    #[test]
    fn parse_time_split_requires_time_config() {
        let json = r#"{
            "index_path": "/tmp/index",
            "query": "PREDICT COUNT(x) FOR EACH y",
            "output_parquet": "/tmp/out.parquet",
            "split": {
                "type": "time",
                "fractions": [0.8, 0.1, 0.1]
            }
        }"#;
        let err = JobConfig::from_str(json).unwrap_err();
        match err {
            ConfigError::InvalidSplit(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn invalid_fraction_sum() {
        let json = r#"{
            "index_path": "/tmp/index",
            "query": "PREDICT COUNT(x) FOR EACH y",
            "output_parquet": "/tmp/out.parquet",
            "split": {
                "type": "random",
                "fractions": [0.7, 0.1, 0.1]
            }
        }"#;
        let err = JobConfig::from_str(json).unwrap_err();
        match err {
            ConfigError::InvalidFractions(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
