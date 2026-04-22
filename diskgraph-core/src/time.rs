use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TimeUnit {
    #[default]
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl TimeUnit {
    pub fn to_nanos(&self) -> i64 {
        match self {
            Self::Nanoseconds => 1,
            Self::Microseconds => 1_000,
            Self::Milliseconds => 1_000_000,
            Self::Seconds => 1_000_000_000,
            Self::Minutes => 60 * 1_000_000_000,
            Self::Hours => 3600 * 1_000_000_000,
            Self::Days => 86400 * 1_000_000_000,
            Self::Weeks => 604800 * 1_000_000_000,
            Self::Months => 2592000 * 1_000_000_000,
            Self::Years => 31536000 * 1_000_000_000,
        }
    }

    pub fn multiplier_to(&self, target: TimeUnit) -> f64 {
        let self_nanos = self.to_nanos() as f64;
        let target_nanos = target.to_nanos() as f64;
        self_nanos / target_nanos
    }
}
