use std::str::FromStr;

use anyhow::{anyhow, Result};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::config::{SplitConfig, SplitRangeConfig, SplitTimeConfig, SplitType};
use crate::exec::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitLabel {
    Train,
    Val,
    Test,
}

impl SplitLabel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SplitLabel::Train => "train",
            SplitLabel::Val => "val",
            SplitLabel::Test => "test",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplitAssigner {
    kind: SplitKind,
}

#[derive(Debug, Clone)]
pub struct SplitBoundaryInfo {
    pub label: SplitLabel,
    pub start: i64,
    pub end: i64,
    pub end_inclusive: bool,
}

#[derive(Debug, Clone)]
enum SplitKind {
    Random { thresholds: (f64, f64), seed: u64 },
    Time { boundaries: Vec<SplitBoundary> },
    Range { ranges: Vec<SplitBoundary> },
}

#[derive(Debug, Clone)]
struct SplitBoundary {
    label: SplitLabel,
    start: i64,
    end: i64,
    end_inclusive: bool,
}

impl SplitBoundary {
    fn contains(&self, ts: i64) -> bool {
        if self.end_inclusive {
            ts >= self.start && ts <= self.end
        } else {
            ts >= self.start && ts < self.end
        }
    }
}

impl SplitAssigner {
    pub fn from_config(
        config: &SplitConfig,
        min_ts: Option<i64>,
        max_ts: Option<i64>,
    ) -> Result<Self> {
        match config.split_type {
            SplitType::Random => {
                let thresholds = fractions_to_thresholds(&config.fractions)?;
                Ok(Self {
                    kind: SplitKind::Random {
                        thresholds,
                        seed: config.seed.unwrap_or(0),
                    },
                })
            }
            SplitType::Range => {
                let range_cfg = config
                    .range
                    .as_ref()
                    .ok_or_else(|| anyhow!("range split requires range config"))?;
                let ranges = build_range_boundaries(range_cfg)?;
                Ok(Self {
                    kind: SplitKind::Range { ranges },
                })
            }
            SplitType::Time => {
                let time_cfg = config
                    .time
                    .as_ref()
                    .ok_or_else(|| anyhow!("time split requires time config"))?;
                let boundaries = build_time_boundaries(
                    &config.fractions,
                    time_cfg,
                    min_ts,
                    max_ts,
                    SplitType::Time,
                )?;
                Ok(Self {
                    kind: SplitKind::Time { boundaries },
                })
            }
            SplitType::DateOffset => {
                let time_cfg = config
                    .time
                    .as_ref()
                    .ok_or_else(|| anyhow!("date_offset split requires time config"))?;
                let boundaries = build_time_boundaries(
                    &config.fractions,
                    time_cfg,
                    min_ts,
                    max_ts,
                    SplitType::DateOffset,
                )?;
                Ok(Self {
                    kind: SplitKind::Time { boundaries },
                })
            }
        }
    }

    pub fn assign(&self, entity: &Value, timestamp: Option<i64>) -> Option<SplitLabel> {
        match &self.kind {
            SplitKind::Random { thresholds, seed } => {
                let hash = hash_entity(entity, *seed);
                let p = (hash as f64) / (u64::MAX as f64);
                if p < thresholds.0 {
                    Some(SplitLabel::Train)
                } else if p < thresholds.1 {
                    Some(SplitLabel::Val)
                } else {
                    Some(SplitLabel::Test)
                }
            }
            SplitKind::Time { boundaries } => {
                let ts = timestamp?;
                boundaries.iter().find(|b| b.contains(ts)).map(|b| b.label)
            }
            SplitKind::Range { ranges } => {
                let ts = timestamp?;
                ranges.iter().find(|b| b.contains(ts)).map(|b| b.label)
            }
        }
    }

    pub fn boundaries(&self) -> Option<Vec<SplitBoundaryInfo>> {
        match &self.kind {
            SplitKind::Time { boundaries } => Some(
                boundaries
                    .iter()
                    .map(|b| SplitBoundaryInfo {
                        label: b.label,
                        start: b.start,
                        end: b.end,
                        end_inclusive: b.end_inclusive,
                    })
                    .collect(),
            ),
            SplitKind::Range { ranges } => Some(
                ranges
                    .iter()
                    .map(|b| SplitBoundaryInfo {
                        label: b.label,
                        start: b.start,
                        end: b.end,
                        end_inclusive: b.end_inclusive,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

fn fractions_to_thresholds(fractions: &[f64]) -> Result<(f64, f64)> {
    if fractions.len() != 3 {
        return Err(anyhow!("fractions must have length 3"));
    }
    Ok((fractions[0], fractions[0] + fractions[1]))
}

fn hash_entity(entity: &Value, seed: u64) -> u64 {
    let mut bytes = Vec::new();
    match entity {
        Value::Int(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        Value::UInt(v) => bytes.extend_from_slice(&v.to_le_bytes()),
        Value::Bool(v) => bytes.push(*v as u8),
        Value::String(s) => bytes.extend_from_slice(s.as_bytes()),
        Value::Bytes(b) => bytes.extend_from_slice(b),
    }
    xxh3_64_with_seed(&bytes, seed)
}

fn build_range_boundaries(range_cfg: &SplitRangeConfig) -> Result<Vec<SplitBoundary>> {
    let mut boundaries = Vec::with_capacity(range_cfg.ranges.len());
    for entry in &range_cfg.ranges {
        let start = parse_timestamp(&entry.start)?;
        let end = parse_timestamp(&entry.end)?;
        let label = parse_label(&entry.label)?;
        boundaries.push(SplitBoundary {
            label,
            start,
            end,
            end_inclusive: true,
        });
    }
    Ok(boundaries)
}

fn build_time_boundaries(
    fractions: &[f64],
    time_cfg: &SplitTimeConfig,
    min_ts: Option<i64>,
    max_ts: Option<i64>,
    split_type: SplitType,
) -> Result<Vec<SplitBoundary>> {
    let max_ts = max_ts.ok_or_else(|| anyhow!("missing max timestamp for split"))?;
    let min_ts = min_ts.ok_or_else(|| anyhow!("missing min timestamp for split"))?;
    let max_ts = apply_bound_override(max_ts, time_cfg.end_time.as_deref())?;
    let min_ts = apply_bound_override(min_ts, time_cfg.start_time.as_deref())?;

    if min_ts > max_ts {
        return Err(anyhow!("invalid split bounds: min_ts > max_ts"));
    }

    let anchor_offset = if time_cfg.anchor_start_offset.trim().is_empty() {
        None
    } else {
        Some(parse_duration(&time_cfg.anchor_start_offset)?)
    };

    let step_delta = if !time_cfg.timeframe_step.trim().is_empty() {
        Some(parse_duration(&time_cfg.timeframe_step)?)
    } else if !time_cfg.timeframe.trim().is_empty() {
        Some(parse_duration(&time_cfg.timeframe)?)
    } else {
        None
    };

    match split_type {
        SplitType::Time => {
            let shifted_min = apply_delta(min_ts, anchor_offset.as_ref())?;
            let shifted_max = apply_delta(max_ts, anchor_offset.as_ref())?;
            let boundaries = build_fractional_boundaries(
                fractions,
                shifted_min,
                shifted_max,
                step_delta.as_ref(),
            )?;
            Ok(boundaries)
        }
        SplitType::DateOffset => {
            build_date_offset_boundaries(fractions, min_ts, max_ts, time_cfg, step_delta.as_ref())
        }
        _ => Err(anyhow!("unsupported split type for time boundaries")),
    }
}

fn build_date_offset_boundaries(
    fractions: &[f64],
    min_ts: i64,
    max_ts: i64,
    time_cfg: &SplitTimeConfig,
    step_delta: Option<&TimeDelta>,
) -> Result<Vec<SplitBoundary>> {
    let train_start = if let Some(offset) = time_cfg.train_start_offset.as_deref() {
        apply_delta(max_ts, Some(&parse_duration(offset)?))?
    } else {
        min_ts
    };
    let train_end = if let Some(offset) = time_cfg.train_end_offset.as_deref() {
        apply_delta(max_ts, Some(&parse_duration(offset)?))?
    } else {
        min_ts + ((max_ts - min_ts) as f64 * fractions[0]) as i64
    };

    if train_start >= train_end {
        return Err(anyhow!(
            "date_offset split requires train_start < train_end"
        ));
    }

    let remaining_start = train_end;
    let remaining_end = max_ts;
    let remaining_len = remaining_end.saturating_sub(remaining_start);
    let val_fraction = fractions[1] / (fractions[1] + fractions[2]);
    let val_end = remaining_start + (remaining_len as f64 * val_fraction).round() as i64;

    let mut boundaries = vec![
        SplitBoundary {
            label: SplitLabel::Train,
            start: train_start,
            end: train_end,
            end_inclusive: false,
        },
        SplitBoundary {
            label: SplitLabel::Val,
            start: train_end,
            end: val_end,
            end_inclusive: false,
        },
        SplitBoundary {
            label: SplitLabel::Test,
            start: val_end,
            end: remaining_end,
            end_inclusive: true,
        },
    ];

    if let Some(step) = step_delta {
        boundaries = align_boundaries(boundaries, step)?;
    }

    Ok(boundaries)
}

fn build_fractional_boundaries(
    fractions: &[f64],
    min_ts: i64,
    max_ts: i64,
    step_delta: Option<&TimeDelta>,
) -> Result<Vec<SplitBoundary>> {
    let thresholds = fractions_to_thresholds(fractions)?;
    let mut boundaries = Vec::new();
    if let Some(step) = step_delta {
        let buckets = build_time_buckets(min_ts, max_ts, step)?;
        if buckets.is_empty() {
            return Err(anyhow!("no time buckets for split boundaries"));
        }
        let num = buckets.len() as f64;
        let train_cut = (num * thresholds.0).ceil() as usize;
        let val_cut = (num * thresholds.1).ceil() as usize;
        let train_cut = train_cut.clamp(1, buckets.len());
        let val_cut = val_cut.clamp(train_cut, buckets.len());

        boundaries.push(SplitBoundary {
            label: SplitLabel::Train,
            start: buckets[0].0,
            end: buckets[train_cut - 1].1,
            end_inclusive: false,
        });
        boundaries.push(SplitBoundary {
            label: SplitLabel::Val,
            start: buckets[train_cut - 1].1,
            end: buckets[val_cut - 1].1,
            end_inclusive: false,
        });
        boundaries.push(SplitBoundary {
            label: SplitLabel::Test,
            start: buckets[val_cut - 1].1,
            end: buckets[buckets.len() - 1].1,
            end_inclusive: true,
        });
    } else {
        let total = max_ts.saturating_sub(min_ts).max(1) as f64;
        let train_end = min_ts + (total * thresholds.0).round() as i64;
        let val_end = min_ts + (total * thresholds.1).round() as i64;
        boundaries.push(SplitBoundary {
            label: SplitLabel::Train,
            start: min_ts,
            end: train_end,
            end_inclusive: false,
        });
        boundaries.push(SplitBoundary {
            label: SplitLabel::Val,
            start: train_end,
            end: val_end,
            end_inclusive: false,
        });
        boundaries.push(SplitBoundary {
            label: SplitLabel::Test,
            start: val_end,
            end: max_ts,
            end_inclusive: true,
        });
    }
    Ok(boundaries)
}

fn build_time_buckets(min_ts: i64, max_ts: i64, step: &TimeDelta) -> Result<Vec<(i64, i64)>> {
    let mut buckets = Vec::new();
    let mut cursor = ts_to_datetime(min_ts)?;
    let end = ts_to_datetime(max_ts)?;
    loop {
        let next = add_delta_datetime(cursor, step)?;
        if next <= cursor {
            return Err(anyhow!("invalid step delta for bucket"));
        }
        let start_ts = cursor.timestamp();
        let end_ts = next.timestamp();
        buckets.push((start_ts, end_ts));
        if next >= end {
            break;
        }
        cursor = next;
    }
    Ok(buckets)
}

fn align_boundaries(
    mut boundaries: Vec<SplitBoundary>,
    step: &TimeDelta,
) -> Result<Vec<SplitBoundary>> {
    for boundary in &mut boundaries {
        let start = ts_to_datetime(boundary.start)?;
        let end = ts_to_datetime(boundary.end)?;
        let aligned_start = add_delta_datetime(start, step)?;
        let aligned_end = add_delta_datetime(end, step)?;
        boundary.start = aligned_start.timestamp();
        boundary.end = aligned_end.timestamp();
    }
    Ok(boundaries)
}

fn parse_label(label: &str) -> Result<SplitLabel> {
    match label.to_lowercase().as_str() {
        "train" => Ok(SplitLabel::Train),
        "val" | "valid" | "validation" => Ok(SplitLabel::Val),
        "test" => Ok(SplitLabel::Test),
        other => Err(anyhow!("unsupported split label: {other}")),
    }
}

fn apply_bound_override(reference: i64, value: Option<&str>) -> Result<i64> {
    let Some(raw) = value else {
        return Ok(reference);
    };
    if raw.trim().is_empty() {
        return Ok(reference);
    }
    if let Ok(offset) = i64::from_str(raw.trim()) {
        return Ok(reference + offset * 86_400);
    }
    parse_timestamp(raw)
}

#[derive(Debug, Clone, Copy)]
enum TimeDelta {
    Seconds(i64),
    Months(i32),
}

fn parse_duration(value: &str) -> Result<TimeDelta> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty duration"));
    }
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.is_empty() {
        return Err(anyhow!("invalid duration"));
    }
    let (num_str, unit_str) = if parts.len() == 1 {
        (parts[0], "seconds")
    } else {
        (parts[0], parts[1])
    };
    let num: i64 = num_str
        .parse()
        .map_err(|_| anyhow!("invalid duration number"))?;
    let unit = unit_str.to_lowercase();
    match unit.as_str() {
        "second" | "seconds" => Ok(TimeDelta::Seconds(num)),
        "minute" | "minutes" => Ok(TimeDelta::Seconds(num * 60)),
        "hour" | "hours" => Ok(TimeDelta::Seconds(num * 3_600)),
        "day" | "days" => Ok(TimeDelta::Seconds(num * 86_400)),
        "week" | "weeks" => Ok(TimeDelta::Seconds(num * 604_800)),
        "month" | "months" => Ok(TimeDelta::Months(num as i32)),
        _ => Err(anyhow!("unsupported duration unit: {unit}")),
    }
}

fn apply_delta(ts: i64, delta: Option<&TimeDelta>) -> Result<i64> {
    let Some(delta) = delta else { return Ok(ts) };
    let dt = ts_to_datetime(ts)?;
    Ok(add_delta_datetime(dt, delta)?.timestamp())
}

fn add_delta_datetime(dt: DateTime<Utc>, delta: &TimeDelta) -> Result<DateTime<Utc>> {
    Ok(match delta {
        TimeDelta::Seconds(sec) => dt + chrono::Duration::seconds(*sec),
        TimeDelta::Months(months) => {
            let abs = months.unsigned_abs();
            let shifted = if *months == 0 {
                Some(dt)
            } else if *months > 0 {
                dt.checked_add_months(chrono::Months::new(abs))
            } else {
                dt.checked_sub_months(chrono::Months::new(abs))
            };
            shifted.ok_or_else(|| anyhow!("overflow in month arithmetic"))?
        }
    })
}

fn parse_timestamp(value: &str) -> Result<i64> {
    if let Ok(ts) = value.trim().parse::<i64>() {
        return Ok(ts);
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(value.trim()) {
        return Ok(dt.timestamp());
    }
    if let Ok(date) = NaiveDate::parse_from_str(value.trim(), "%Y-%m-%d") {
        let dt = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| anyhow!("invalid date"))?;
        return Ok(Utc.from_utc_datetime(&dt).timestamp());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(value.trim(), "%Y-%m-%d %H:%M:%S") {
        return Ok(Utc.from_utc_datetime(&dt).timestamp());
    }
    Err(anyhow!("unsupported timestamp format: {value}"))
}

fn ts_to_datetime(ts: i64) -> Result<DateTime<Utc>> {
    Utc.timestamp_opt(ts, 0)
        .single()
        .ok_or_else(|| anyhow!("invalid timestamp: {ts}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        SplitConfig, SplitRangeConfig, SplitRangeEntry, SplitTimeConfig, SplitType,
    };

    #[test]
    fn random_split_is_deterministic() {
        let config = SplitConfig {
            split_type: SplitType::Random,
            fractions: vec![0.5, 0.25, 0.25],
            seed: Some(42),
            time: None,
            range: None,
        };
        let assigner = SplitAssigner::from_config(&config, None, None).unwrap();
        let entity = Value::String("user-1".to_string());
        let first = assigner.assign(&entity, None);
        let second = assigner.assign(&entity, None);
        assert_eq!(first, second);
    }

    #[test]
    fn time_split_assigns_by_fraction() {
        let config = SplitConfig {
            split_type: SplitType::Time,
            fractions: vec![0.5, 0.25, 0.25],
            seed: None,
            time: Some(SplitTimeConfig {
                anchor: "max".to_string(),
                timeframe: "1 days".to_string(),
                anchor_start_offset: "0 days".to_string(),
                timeframe_step: "1 days".to_string(),
                train_start_offset: None,
                train_end_offset: None,
                start_time: None,
                end_time: None,
            }),
            range: None,
        };
        let assigner = SplitAssigner::from_config(&config, Some(0), Some(9 * 86_400)).unwrap();
        let entity = Value::Int(1);
        assert_eq!(
            assigner.assign(&entity, Some(86_400)),
            Some(SplitLabel::Train)
        );
        assert_eq!(
            assigner.assign(&entity, Some(6 * 86_400)),
            Some(SplitLabel::Val)
        );
        assert_eq!(
            assigner.assign(&entity, Some(9 * 86_400)),
            Some(SplitLabel::Test)
        );
    }

    #[test]
    fn date_offset_split_uses_train_end_offset() {
        let config = SplitConfig {
            split_type: SplitType::DateOffset,
            fractions: vec![0.6, 0.2, 0.2],
            seed: None,
            time: Some(SplitTimeConfig {
                anchor: "max".to_string(),
                timeframe: "1 days".to_string(),
                anchor_start_offset: "0 days".to_string(),
                timeframe_step: "1 days".to_string(),
                train_start_offset: Some("-10 days".to_string()),
                train_end_offset: Some("-5 days".to_string()),
                start_time: None,
                end_time: None,
            }),
            range: None,
        };
        let assigner = SplitAssigner::from_config(&config, Some(0), Some(20 * 86_400)).unwrap();
        let entity = Value::Int(1);
        assert_eq!(
            assigner.assign(&entity, Some(12 * 86_400)),
            Some(SplitLabel::Train)
        );
        assert_eq!(
            assigner.assign(&entity, Some(17 * 86_400)),
            Some(SplitLabel::Val)
        );
    }

    #[test]
    fn range_split_matches_labels() {
        let config = SplitConfig {
            split_type: SplitType::Range,
            fractions: vec![0.8, 0.1, 0.1],
            seed: None,
            time: None,
            range: Some(SplitRangeConfig {
                unit: "timestamp".to_string(),
                ranges: vec![
                    SplitRangeEntry {
                        start: "0".to_string(),
                        end: "9".to_string(),
                        label: "train".to_string(),
                    },
                    SplitRangeEntry {
                        start: "10".to_string(),
                        end: "19".to_string(),
                        label: "val".to_string(),
                    },
                ],
            }),
        };
        let assigner = SplitAssigner::from_config(&config, Some(0), Some(20)).unwrap();
        let entity = Value::Int(1);
        assert_eq!(assigner.assign(&entity, Some(5)), Some(SplitLabel::Train));
        assert_eq!(assigner.assign(&entity, Some(15)), Some(SplitLabel::Val));
        assert_eq!(assigner.assign(&entity, Some(25)), None);
    }
}
