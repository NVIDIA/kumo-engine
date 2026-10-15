// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::MAX_INPUT_STRING_BYTES;
use anyhow::{anyhow, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use chrono::{NaiveDate, NaiveDateTime};
use uuid_simd::{parse_hyphenated, parse_simple, AsOut};

pub(crate) fn parse_bool(value: &str) -> Option<bool> {
    let mut text = value.trim();
    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        text = &text[1..text.len() - 1];
    }
    let text = text.trim();
    let lower = text.to_ascii_lowercase();
    match lower.as_str() {
        "" | "false" | "f" | "no" | "n" | "off" | "disabled" | "disable" | "unchecked" => {
            return Some(false);
        }
        "true" | "t" | "yes" | "y" | "on" | "enabled" | "enable" | "checked" | "x" => {
            return Some(true);
        }
        _ => {}
    }
    if let Ok(int_value) = lower.parse::<i64>() {
        return Some(int_value != 0);
    }
    if let Ok(float_value) = lower.parse::<f64>() {
        return Some(!float_value.is_nan() && float_value != 0.0);
    }
    None
}

pub(crate) fn parse_i64(value: &str) -> Option<i64> {
    let text = value.trim();
    text.parse::<i64>()
        .ok()
        .or_else(|| zero_fraction_decimal_prefix(text)?.parse::<i64>().ok())
}

pub(crate) fn parse_i32(value: &str) -> Option<i32> {
    let text = value.trim();
    text.parse::<i32>()
        .ok()
        .or_else(|| zero_fraction_decimal_prefix(text)?.parse::<i32>().ok())
}

pub(crate) fn parse_u64(value: &str) -> Option<u64> {
    let text = value.trim();
    text.parse::<u64>()
        .ok()
        .or_else(|| zero_fraction_decimal_prefix(text)?.parse::<u64>().ok())
}

pub(crate) fn parse_u32(value: &str) -> Option<u32> {
    let text = value.trim();
    text.parse::<u32>()
        .ok()
        .or_else(|| zero_fraction_decimal_prefix(text)?.parse::<u32>().ok())
}

fn zero_fraction_decimal_prefix(text: &str) -> Option<&str> {
    let (whole, fractional) = text.split_once('.')?;
    if fractional.is_empty() || !fractional.bytes().all(|b| b == b'0') {
        return None;
    }
    Some(whole)
}

pub(crate) fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok()
}

pub(crate) fn parse_f32(value: &str) -> Option<f32> {
    value.trim().parse::<f32>().ok()
}

pub(crate) fn trim_str_bytes(value: &str, max: usize) -> &str {
    if value.len() <= max {
        return value;
    }
    let mut end = max;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

pub(crate) fn trim_borrowed_str<'a>(
    value: &'a str,
    max: usize,
    counter: Option<&std::sync::atomic::AtomicU64>,
) -> &'a str {
    if value.len() <= max {
        return value;
    }
    if let Some(counter) = counter {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    trim_str_bytes(value, max)
}

pub(crate) fn trim_borrowed_bytes<'a>(
    value: &'a [u8],
    max: usize,
    counter: Option<&std::sync::atomic::AtomicU64>,
) -> &'a [u8] {
    if value.len() <= max {
        return value;
    }
    if let Some(counter) = counter {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    &value[..max]
}

pub(crate) struct StackStr {
    buf: [u8; 64],
    len: usize,
}

impl StackStr {
    pub(crate) fn new() -> Self {
        Self {
            buf: [0u8; 64],
            len: 0,
        }
    }

    fn as_str(&self) -> Result<&str> {
        std::str::from_utf8(&self.buf[..self.len]).map_err(|_| anyhow!("utf8 format failed"))
    }
}

impl std::fmt::Write for StackStr {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        if self.len + bytes.len() > self.buf.len() {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

pub(crate) fn with_stack_str<Out, V: std::fmt::Display>(
    value: V,
    counter: Option<&std::sync::atomic::AtomicU64>,
    f: impl FnOnce(&str) -> Result<Out>,
) -> Result<Out> {
    use std::fmt::Write as _;
    let mut buf = StackStr::new();
    write!(&mut buf, "{}", value).map_err(|_| anyhow!("string format overflow"))?;
    let s = buf.as_str()?;
    let s = trim_borrowed_str(s, MAX_INPUT_STRING_BYTES, counter);
    f(s)
}

pub(crate) fn with_float_str<Out, V: std::fmt::Display + std::fmt::LowerExp>(
    value: V,
    counter: Option<&std::sync::atomic::AtomicU64>,
    f: impl FnOnce(&str) -> Result<Out>,
) -> Result<Out> {
    use std::fmt::Write as _;
    let mut buf = StackStr::new();
    if write!(&mut buf, "{}", value).is_err() {
        buf = StackStr::new();
        write!(&mut buf, "{:e}", value).map_err(|_| anyhow!("string format overflow"))?;
    }
    let s = buf.as_str()?;
    f(trim_borrowed_str(s, MAX_INPUT_STRING_BYTES, counter))
}

pub(crate) fn parse_uuid_string(bytes: &[u8]) -> Option<[u8; 16]> {
    let mut out = [0u8; 16];
    if bytes.len() == 32 {
        parse_simple(bytes, out.as_out()).ok()?;
        return Some(out);
    }
    if bytes.len() == 36 {
        parse_hyphenated(bytes, out.as_out()).ok()?;
        return Some(out);
    }
    None
}

pub(crate) fn parse_uuid256_string(bytes: &[u8]) -> Option<[u8; 32]> {
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    // Safety: `out` is 32 bytes, so the two 16-byte regions are in-bounds and non-overlapping.
    let out_ptr = out.as_mut_ptr();
    let hi = unsafe { &mut *(out_ptr as *mut [u8; 16]) };
    let lo = unsafe { &mut *(out_ptr.add(16) as *mut [u8; 16]) };
    parse_simple(&bytes[..32], hi.as_out()).ok()?;
    parse_simple(&bytes[32..], lo.as_out()).ok()?;
    Some(out)
}

pub(crate) fn parse_ip_string(bytes: &[u8]) -> Option<[u8; 16]> {
    let s = std::str::from_utf8(bytes).ok()?;
    let ip: std::net::IpAddr = s.parse().ok()?;
    Some(ip_to_ipv6_bytes(ip))
}

pub(crate) fn parse_ip_bytes(bytes: &[u8]) -> Option<[u8; 16]> {
    match bytes.len() {
        4 => Some(ip_to_ipv6_bytes(std::net::IpAddr::V4(
            std::net::Ipv4Addr::new(bytes[0], bytes[1], bytes[2], bytes[3]),
        ))),
        16 => {
            let mut out = [0u8; 16];
            out.copy_from_slice(bytes);
            Some(out)
        }
        _ => None,
    }
}

pub(crate) fn ip_to_ipv6_bytes(ip: std::net::IpAddr) -> [u8; 16] {
    match ip {
        std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        std::net::IpAddr::V6(v6) => v6.octets(),
    }
}

fn format_has_tz(fmt: &str) -> bool {
    fmt.contains("%z") || fmt.contains("%:z") || fmt.contains("%#z")
}

/// kumo#30447: chrono's `%z` rejects the `Z` that RFC3339 allows, so an
/// offset-bearing format must not be stricter than no format at all.
fn parse_rfc3339_utc(value: &str) -> Option<NaiveDateTime> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|v| v.naive_utc())
}

pub(crate) fn parse_timestamp_text(value: &str, format: Option<&str>) -> Option<i64> {
    let ts = if let Some(fmt) = format {
        if format_has_tz(fmt) {
            chrono::DateTime::parse_from_str(value, fmt)
                .ok()
                .map(|v| v.naive_utc())
                // Mixed-tz fallback: format expects tz but value has none → try without tz
                .or_else(|| {
                    let base_fmt = fmt
                        .replace(" %:z", "")
                        .replace("%:z", "")
                        .replace(" %#z", "")
                        .replace("%#z", "")
                        .replace(" %z", "")
                        .replace("%z", "");
                    NaiveDateTime::parse_from_str(value, &base_fmt)
                        .ok()
                        .or_else(|| {
                            NaiveDate::parse_from_str(value, &base_fmt)
                                .ok()
                                .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
                        })
                })
                .or_else(|| parse_rfc3339_utc(value))?
        } else {
            NaiveDateTime::parse_from_str(value, fmt)
                .ok()
                .or_else(|| {
                    NaiveDate::parse_from_str(value, fmt)
                        .ok()
                        .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
                })?
        }
    } else {
        parse_rfc3339_utc(value)
            .or_else(|| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").ok())
            .or_else(|| {
                NaiveDate::parse_from_str(value, "%Y-%m-%d")
                    .ok()
                    .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
            })?
    };
    let ts = ts.and_utc();
    Some(ts.timestamp())
}

pub(crate) fn parse_date_text(value: &str, format: Option<&str>) -> Option<i32> {
    let dt = if let Some(fmt) = format {
        NaiveDate::parse_from_str(value, fmt)
            .ok()
            .or_else(|| {
                NaiveDateTime::parse_from_str(value, fmt)
                    .ok()
                    .map(|v| v.date())
            })
            .or_else(|| {
                format_has_tz(fmt)
                    .then(|| parse_rfc3339_utc(value))
                    .flatten()
                    .map(|v| v.date())
            })
    } else {
        NaiveDate::parse_from_str(value, "%Y-%m-%d")
            .ok()
            .or_else(|| {
                NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .map(|v| v.date())
            })
    }?;
    let ts = dt.and_hms_opt(0, 0, 0)?.and_utc().timestamp();
    Some(days_from_secs(ts))
}

#[derive(Clone, Debug)]
enum TimestampFormatKind {
    Ymd,
    YmdHmSpace,
    YmdHmsSpace,
    YmdHmsSpaceFrac,
    YmdHmsT,
    YmdHmsTFrac,
    YmdHmsSpaceOffset,
    YmdHmsSpaceOffsetColon,
    YmdHmsSpaceOffsetSpaced,
    YmdHmsSpaceOffsetColonSpaced,
    YmdHmsTOffset,
    YmdHmsTOffsetColon,
    /// "%Y%m%d" — encoded integer date (e.g. 20230408)
    CompactYmd,
    /// "%Y%m%d%H%M%S" — encoded integer datetime (e.g. 20230408120530)
    CompactYmdHms,
}

#[derive(Clone, Debug)]
pub(crate) struct TimestampFormatParser {
    kind: TimestampFormatKind,
}

impl TimestampFormatParser {
    pub(crate) fn compile(format: &str) -> Option<Self> {
        let kind = match format {
            "%Y-%m-%d" => TimestampFormatKind::Ymd,
            "%Y-%m-%d %H:%M" => TimestampFormatKind::YmdHmSpace,
            "%Y-%m-%d %H:%M:%S" => TimestampFormatKind::YmdHmsSpace,
            "%Y-%m-%d %H:%M:%S%.f" => TimestampFormatKind::YmdHmsSpaceFrac,
            "%Y-%m-%dT%H:%M:%S" => TimestampFormatKind::YmdHmsT,
            "%Y-%m-%dT%H:%M:%S%.f" => TimestampFormatKind::YmdHmsTFrac,
            "%Y-%m-%d %H:%M:%S%z" => TimestampFormatKind::YmdHmsSpaceOffset,
            "%Y-%m-%d %H:%M:%S %z" => TimestampFormatKind::YmdHmsSpaceOffsetSpaced,
            "%Y-%m-%d %H:%M:%S%:z" => TimestampFormatKind::YmdHmsSpaceOffsetColon,
            "%Y-%m-%d %H:%M:%S %:z" => TimestampFormatKind::YmdHmsSpaceOffsetColonSpaced,
            "%Y-%m-%dT%H:%M:%S%z" => TimestampFormatKind::YmdHmsTOffset,
            "%Y-%m-%dT%H:%M:%S%:z" => TimestampFormatKind::YmdHmsTOffsetColon,
            "%Y%m%d" => TimestampFormatKind::CompactYmd,
            "%Y%m%d%H%M%S" => TimestampFormatKind::CompactYmdHms,
            _ => return None,
        };
        Some(Self { kind })
    }

    pub(crate) fn parse_timestamp(&self, value: &str) -> Option<i64> {
        match self.kind {
            TimestampFormatKind::Ymd => parse_ymd_fast(value.as_bytes()),
            TimestampFormatKind::YmdHmSpace => parse_ymd_hm_space_fast(value.as_bytes()),
            TimestampFormatKind::YmdHmsSpace => parse_ymd_hms_space_fast(value.as_bytes()),
            TimestampFormatKind::YmdHmsSpaceFrac => parse_ymd_hms_space_frac_fast(value.as_bytes()),
            TimestampFormatKind::YmdHmsT => parse_ymd_hms_space_fast(value.as_bytes()),
            TimestampFormatKind::YmdHmsTFrac => parse_ymd_hms_space_frac_fast(value.as_bytes()),
            TimestampFormatKind::YmdHmsSpaceOffset => {
                parse_ymd_hms_offset_nocolon_fast(value.as_bytes())
            }
            TimestampFormatKind::YmdHmsSpaceOffsetColon => {
                parse_ymd_hms_offset_colon_fast(value.as_bytes())
            }
            TimestampFormatKind::YmdHmsSpaceOffsetSpaced => {
                parse_ymd_hms_offset_nocolon_spaced_fast(value.as_bytes())
            }
            TimestampFormatKind::YmdHmsSpaceOffsetColonSpaced => {
                parse_ymd_hms_offset_colon_spaced_fast(value.as_bytes())
            }
            TimestampFormatKind::YmdHmsTOffset => {
                parse_ymd_hms_offset_nocolon_fast(value.as_bytes())
            }
            TimestampFormatKind::YmdHmsTOffsetColon => {
                parse_ymd_hms_offset_colon_fast(value.as_bytes())
            }
            TimestampFormatKind::CompactYmd => parse_compact_ymd_fast(value.as_bytes()),
            TimestampFormatKind::CompactYmdHms => parse_compact_ymd_hms_fast(value.as_bytes()),
        }
    }

    pub(crate) fn parse_date(&self, value: &str) -> Option<i32> {
        let secs = self.parse_timestamp(value)?;
        Some(days_from_secs(secs))
    }
}

/// Known timestamp format strings, ordered by detection priority.
/// ISO formats first (fast-path), then non-ISO formats (chrono fallback for detection only).
pub const TIMESTAMP_FORMATS: &[&str] = &[
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S %z",
    "%Y-%m-%d %H:%M:%S%:z",
    "%Y-%m-%d %H:%M:%S %:z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%:z",
    // %#z also accepts a 'Z' suffix, which %z/%:z reject
    "%Y-%m-%dT%H:%M:%S%#z",
    "%Y-%m-%d %H:%M:%S%#z",
    // Non-ISO formats (chrono fallback, lower priority)
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
];

/// Detect the timestamp format of a string value.
pub fn detect_timestamp_format(value: &str) -> Option<&'static str> {
    TIMESTAMP_FORMATS.iter().copied().find(|&fmt| {
        if let Some(p) = TimestampFormatParser::compile(fmt) {
            p.parse_timestamp(value).is_some()
        } else {
            // Formats without a compiled fast-path: try chrono directly
            parse_timestamp_text(value, Some(fmt)).is_some()
        }
    })
}

pub(crate) fn digit_count(mut value: u64) -> u32 {
    let mut n = 1;
    while value >= 10 {
        value /= 10;
        n += 1;
    }
    n
}

/// Detect if an integer value encodes a date in a compact format.
///
/// Checks digit count and validates plausibility (year 1900-2100, etc.):
/// - 4 digits  → `"%Y"`           (e.g. 2023)
/// - 6 digits  → `"%Y%m"`         (e.g. 202304)
/// - 8 digits  → `"%Y%m%d"`       (e.g. 20230408)
/// - 14 digits → `"%Y%m%d%H%M%S"` (e.g. 20230408120530)
///
/// Returns `None` for values that don't match or fail plausibility checks.
pub fn detect_encoded_int_format(value: i64) -> Option<&'static str> {
    let abs = value.unsigned_abs();
    match digit_count(abs) {
        4 => {
            // %Y — year only
            let year = abs as u32;
            if (1900..=2100).contains(&year) {
                Some("%Y")
            } else {
                None
            }
        }
        6 => {
            // %Y%m — year + month
            let year = (abs / 100) as u32;
            let month = (abs % 100) as u32;
            if (1900..=2100).contains(&year) && (1..=12).contains(&month) {
                Some("%Y%m")
            } else {
                None
            }
        }
        8 => {
            // %Y%m%d — year + month + day
            let year = (abs / 10_000) as u32;
            let month = ((abs / 100) % 100) as u32;
            let day = (abs % 100) as u32;
            if (1900..=2100).contains(&year)
                && (1..=12).contains(&month)
                && (1..=31).contains(&day)
            {
                Some("%Y%m%d")
            } else {
                None
            }
        }
        14 => {
            // %Y%m%d%H%M%S — year + month + day + hour + minute + second
            let year = (abs / 10_000_000_000) as u32;
            let month = ((abs / 100_000_000) % 100) as u32;
            let day = ((abs / 1_000_000) % 100) as u32;
            let hour = ((abs / 10_000) % 100) as u32;
            let minute = ((abs / 100) % 100) as u32;
            let second = (abs % 100) as u32;
            if (1900..=2100).contains(&year)
                && (1..=12).contains(&month)
                && (1..=31).contains(&day)
                && hour <= 23
                && minute <= 59
                && second <= 59
            {
                Some("%Y%m%d%H%M%S")
            } else {
                None
            }
        }
        _ => None,
    }
}

// "%Y%m%d" — compact date without separators
pub(crate) fn parse_compact_ymd_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 8 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    const ZERO: u64 = 0x3030_3030_3030_3030;
    if !all_digits(a) {
        return None;
    }
    let d = a.wrapping_sub(ZERO);
    let year = byte_u32(d, 0) * 1000 + byte_u32(d, 8) * 100 + byte_u32(d, 16) * 10 + byte_u32(d, 24);
    let month = byte_u32(d, 32) * 10 + byte_u32(d, 40);
    let day = byte_u32(d, 48) * 10 + byte_u32(d, 56);
    let secs = timestamp_from_parts_fast(year as i32, month, day, 0, 0, 0);
    Some(secs)
}

// "%Y%m%d%H%M%S" — compact datetime without separators
pub(crate) fn parse_compact_ymd_hms_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 14 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let mut tail = [0u8; 8];
    tail[..6].copy_from_slice(&bytes[8..14]);
    let b = u64::from_le_bytes(tail);

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]);
    if !all_digits(a) {
        return None;
    }
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    if !all_digits(x_b) {
        return None;
    }

    let da = a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let year = byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 32) * 10 + byte_u32(da, 40);
    let day = byte_u32(da, 48) * 10 + byte_u32(da, 56);
    let hour = byte_u32(db, 0) * 10 + byte_u32(db, 8);
    let minute = byte_u32(db, 16) * 10 + byte_u32(db, 24);
    let second = byte_u32(db, 32) * 10 + byte_u32(db, 40);

    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some(secs)
}

pub(crate) fn parse_fraction(bytes: &[u8], pos: usize) -> Option<(u32, usize, usize)> {
    let mut idx = pos;
    let mut val: u32 = 0;
    let mut digits = 0usize;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        if digits < 9 {
            val = val
                .saturating_mul(10)
                .saturating_add((bytes[idx] - b'0') as u32);
        }
        digits += 1;
        idx += 1;
    }
    if digits == 0 {
        return None;
    }
    Some((val, digits.min(9), idx))
}

// "%Y-%m-%d"
pub(crate) fn parse_ymd_fast(bytes: &[u8]) -> Option<i64> {
    // Separator bytes are not compared, so rejecting digits there is what keeps
    // a 10-digit epoch-seconds string from reading as a date.
    if bytes.len() != 10 || bytes[4].is_ascii_digit() || bytes[7].is_ascii_digit() {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let mut tail = [0u8; 8];
    tail[..2].copy_from_slice(&bytes[8..10]);
    let b = u64::from_le_bytes(tail);

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_A: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00]);
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);

    let x_a = (a & MASK_A) | (!MASK_A & ZERO);
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    if !all_digits(x_a) || !all_digits(x_b) {
        return None;
    }

    let da = x_a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let year =
        byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 40) * 10 + byte_u32(da, 48);
    let day = byte_u32(db, 0) * 10 + byte_u32(db, 8);

    let secs = timestamp_from_parts_fast(year as i32, month, day, 0, 0, 0);
    Some(secs)
}

// "%Y-%m-%d %H:%M"
pub(crate) fn parse_ymd_hm_space_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 16 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let b = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(8) as *const u64) };

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_A: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00]);
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF]);

    let x_a = (a & MASK_A) | (!MASK_A & ZERO);
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    if !all_digits(x_a) || !all_digits(x_b) {
        return None;
    }

    let da = x_a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let year =
        byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 40) * 10 + byte_u32(da, 48);
    let day = byte_u32(db, 0) * 10 + byte_u32(db, 8);
    let hour = byte_u32(db, 24) * 10 + byte_u32(db, 32);
    let minute = byte_u32(db, 48) * 10 + byte_u32(db, 56);

    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, 0);
    Some(secs)
}

#[inline]
pub(crate) fn parse_ymd_hms_digits(bytes: &[u8]) -> Option<(u32, u32, u32, u32, u32, u32)> {
    if bytes.len() < 19 {
        return None;
    }
    let a = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const u64) };
    let b = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(8) as *const u64) };
    let mut tail = [0u8; 8];
    tail[..3].copy_from_slice(&bytes[16..19]);
    let c = u64::from_le_bytes(tail);

    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK_A: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00]);
    const MASK_B: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF]);
    const MASK_C: u64 = u64::from_le_bytes([0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00]);

    let x_a = (a & MASK_A) | (!MASK_A & ZERO);
    let x_b = (b & MASK_B) | (!MASK_B & ZERO);
    let x_c = (c & MASK_C) | (!MASK_C & ZERO);
    if !all_digits(x_a) || !all_digits(x_b) || !all_digits(x_c) {
        return None;
    }

    let da = x_a.wrapping_sub(ZERO);
    let db = x_b.wrapping_sub(ZERO);
    let dc = x_c.wrapping_sub(ZERO);

    let year =
        byte_u32(da, 0) * 1000 + byte_u32(da, 8) * 100 + byte_u32(da, 16) * 10 + byte_u32(da, 24);
    let month = byte_u32(da, 40) * 10 + byte_u32(da, 48);
    let day = byte_u32(db, 0) * 10 + byte_u32(db, 8);
    let hour = byte_u32(db, 24) * 10 + byte_u32(db, 32);
    let minute = byte_u32(db, 48) * 10 + byte_u32(db, 56);
    let second = byte_u32(dc, 8) * 10 + byte_u32(dc, 16);

    Some((year, month, day, hour, minute, second))
}

// "%Y-%m-%d %H:%M:%S" / "%Y-%m-%dT%H:%M:%S"
pub(crate) fn parse_ymd_hms_space_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 19 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some(secs)
}

// "%Y-%m-%d %H:%M:%S%.f" / "%Y-%m-%dT%H:%M:%S%.f"
pub(crate) fn parse_ymd_hms_space_frac_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() < 21 {
        return None;
    }
    if bytes[19] != b'.' {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let (_fraction, _digits, next) = parse_fraction(bytes, 20)?;
    if next != bytes.len() {
        return None;
    }
    // Sub-second fraction discarded — standardized to epoch seconds.
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some(secs)
}

#[inline]
pub(crate) fn parse_offset_hhmm(bytes: &[u8], pos: usize) -> Option<i64> {
    let sign = match *bytes.get(pos)? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let mut tail = [0u8; 8];
    tail[..4].copy_from_slice(bytes.get(pos + 1..pos + 5)?);
    let word = u64::from_le_bytes(tail);
    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK: u64 = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]);
    let x = (word & MASK) | (!MASK & ZERO);
    if !all_digits(x) {
        return None;
    }
    let d = x.wrapping_sub(ZERO);
    let hour = byte_u32(d, 0) * 10 + byte_u32(d, 8);
    let minute = byte_u32(d, 16) * 10 + byte_u32(d, 24);
    Some(sign * (hour as i64 * 3_600 + minute as i64 * 60))
}

#[inline]
pub(crate) fn parse_offset_hhmm_colon(bytes: &[u8], pos: usize) -> Option<i64> {
    let sign = match *bytes.get(pos)? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let mut tail = [0u8; 8];
    tail[..5].copy_from_slice(bytes.get(pos + 1..pos + 6)?);
    let word = u64::from_le_bytes(tail);
    const ZERO: u64 = 0x3030_3030_3030_3030;
    const MASK: u64 = u64::from_le_bytes([0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00]);
    let x = (word & MASK) | (!MASK & ZERO);
    if !all_digits(x) {
        return None;
    }
    let d = x.wrapping_sub(ZERO);
    let hour = byte_u32(d, 0) * 10 + byte_u32(d, 8);
    let minute = byte_u32(d, 24) * 10 + byte_u32(d, 32);
    Some(sign * (hour as i64 * 3_600 + minute as i64 * 60))
}

// "%Y-%m-%d %H:%M:%S%z" / "%Y-%m-%dT%H:%M:%S%z"
pub(crate) fn parse_ymd_hms_offset_nocolon_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 24 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm(bytes, 19)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some(secs)
}

// "%Y-%m-%d %H:%M:%S%:z" / "%Y-%m-%dT%H:%M:%S%:z"
pub(crate) fn parse_ymd_hms_offset_colon_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 25 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm_colon(bytes, 19)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some(secs)
}

// "%Y-%m-%d %H:%M:%S %z"
pub(crate) fn parse_ymd_hms_offset_nocolon_spaced_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 25 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm(bytes, 20)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some(secs)
}

// "%Y-%m-%d %H:%M:%S %:z"
pub(crate) fn parse_ymd_hms_offset_colon_spaced_fast(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 26 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm_colon(bytes, 20)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some(secs)
}

#[inline]
pub(crate) fn byte_u32(value: u64, shift: u32) -> u32 {
    ((value >> shift) & 0xFF) as u32
}

#[inline]
pub(crate) fn all_digits(value: u64) -> bool {
    const ZERO: u64 = 0x3030_3030_3030_3030;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let add = value.wrapping_add(0x4646_4646_4646_4646);
    let sub = value.wrapping_sub(ZERO);
    ((add | sub) & HIGH) == 0
}

pub(crate) fn timestamp_from_parts_fast(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> i64 {
    let days = days_from_civil_fast(year, month, day);
    days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64
}

pub(crate) fn days_from_civil_fast(year: i32, month: u32, day: u32) -> i64 {
    let mut y = year as i64;
    let m = month as i64;
    let d = day as i64;
    y -= (m <= 2) as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = m + 9 - (m > 2) as i64 * 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Ceiling division for signed integers with a positive divisor.
/// Rounds toward positive infinity — ensures sub-second timestamps are
/// never assigned to an earlier second, preventing future data leakage
/// in temporal aggregations.
#[inline]
fn div_ceil_i64(value: i64, divisor: i64) -> i64 {
    let d = value.div_euclid(divisor);
    let r = value.rem_euclid(divisor);
    if r > 0 { d + 1 } else { d }
}

pub(crate) fn timestamp_from_int(value: i64, unit: Option<arrow_schema::TimeUnit>) -> i64 {
    let unit = unit.unwrap_or(arrow_schema::TimeUnit::Second);
    match unit {
        arrow_schema::TimeUnit::Second => value,
        arrow_schema::TimeUnit::Millisecond => div_ceil_i64(value, 1_000),
        arrow_schema::TimeUnit::Microsecond => div_ceil_i64(value, 1_000_000),
        arrow_schema::TimeUnit::Nanosecond => div_ceil_i64(value, 1_000_000_000),
    }
}

pub(crate) fn days_from_secs(secs: i64) -> i32 {
    div_ceil_i64(secs, 86_400) as i32
}

pub(crate) fn parse_decimal_i128(value: &str, scale: i32) -> Option<i128> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("");
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let scale = scale.max(0) as usize;
    if frac.len() > scale {
        return None;
    }
    let mut buf = String::with_capacity(whole.len() + scale);
    buf.push_str(whole);
    buf.push_str(frac);
    for _ in frac.len()..scale {
        buf.push('0');
    }
    let parsed = buf.parse::<i128>().ok()?;
    if negative {
        Some(-parsed)
    } else {
        Some(parsed)
    }
}

pub(crate) fn parse_decimal_i256(value: &str, scale: i32) -> Option<arrow_buffer::i256> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let negative = value.starts_with('-');
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let mut parts = digits.split('.');
    let whole = parts.next().unwrap_or("");
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return None;
    }
    let scale = scale.max(0) as usize;
    if frac.len() > scale {
        return None;
    }
    let mut buf = String::with_capacity(whole.len() + scale);
    buf.push_str(whole);
    buf.push_str(frac);
    for _ in frac.len()..scale {
        buf.push('0');
    }
    let mut parsed = arrow_buffer::i256::from_string(&buf)?;
    if negative {
        parsed = parsed.checked_neg()?;
    }
    Some(parsed)
}

pub(crate) fn timestamp_secs_from_array(array: &ArrayRef, row: usize) -> Result<i64> {
    let unit = match array.data_type() {
        DataType::Timestamp(unit, _) => *unit,
        _ => return Err(anyhow!("timestamp type mismatch")),
    };
    match unit {
        arrow_schema::TimeUnit::Second => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampSecondArray>()
                .ok_or_else(|| anyhow!("timestamp second cast"))?;
            Ok(arr.value(row))
        }
        arrow_schema::TimeUnit::Millisecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                .ok_or_else(|| anyhow!("timestamp millisecond cast"))?;
            Ok(div_ceil_i64(arr.value(row), 1_000))
        }
        arrow_schema::TimeUnit::Microsecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                .ok_or_else(|| anyhow!("timestamp microsecond cast"))?;
            Ok(div_ceil_i64(arr.value(row), 1_000_000))
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                .ok_or_else(|| anyhow!("timestamp nanosecond cast"))?;
            Ok(div_ceil_i64(arr.value(row), 1_000_000_000))
        }
    }
}

pub(crate) fn time_secs_from_array(array: &ArrayRef, row: usize) -> Result<i64> {
    match array.data_type() {
        DataType::Time32(arrow_schema::TimeUnit::Second) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Time32SecondArray>()
                .ok_or_else(|| anyhow!("time32 second cast"))?;
            Ok(i64::from(arr.value(row)))
        }
        DataType::Time32(arrow_schema::TimeUnit::Millisecond) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Time32MillisecondArray>()
                .ok_or_else(|| anyhow!("time32 millisecond cast"))?;
            Ok(div_ceil_i64(i64::from(arr.value(row)), 1_000))
        }
        DataType::Time64(arrow_schema::TimeUnit::Microsecond) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Time64MicrosecondArray>()
                .ok_or_else(|| anyhow!("time64 microsecond cast"))?;
            Ok(div_ceil_i64(arr.value(row), 1_000_000))
        }
        DataType::Time64(arrow_schema::TimeUnit::Nanosecond) => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::Time64NanosecondArray>()
                .ok_or_else(|| anyhow!("time64 nanosecond cast"))?;
            Ok(div_ceil_i64(arr.value(row), 1_000_000_000))
        }
        DataType::Time32(_) | DataType::Time64(_) => Err(anyhow!(
            "unsupported Arrow time unit for timestamp conversion: {:?}",
            array.data_type()
        )),
        _ => Err(anyhow!("time type mismatch")),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_i32, parse_i64, parse_u32, parse_u64};

    #[test]
    fn parses_integral_decimal_signed_ints() {
        for (value, expected) in [
            ("1.0", 1),
            ("0.0", 0),
            ("-7.0", -7),
            ("+42.000", 42),
            ("-2147483648.0", i32::MIN),
            ("2147483647.0", i32::MAX),
        ] {
            assert_eq!(parse_i64(value), Some(i64::from(expected)));
            assert_eq!(parse_i32(value), Some(expected));
        }
    }

    #[test]
    fn parses_integral_decimal_unsigned_ints() {
        for (value, expected) in [
            ("0.0", 0),
            ("1.0", 1),
            ("42.000", 42),
            ("4294967295.0", u32::MAX),
        ] {
            assert_eq!(parse_u64(value), Some(u64::from(expected)));
            assert_eq!(parse_u32(value), Some(expected));
        }
        assert_eq!(parse_u64("18446744073709551615.0"), Some(u64::MAX));
    }

    #[test]
    fn rejects_non_integral_decimal_ints() {
        for value in [
            "1.", "1.5", "-7.9", "1.00x", "1.0.0", "1e0", "NaN", "inf", "abc.0",
            "+.0", "--1.0",
        ] {
            assert_eq!(parse_i64(value), None);
        }
        assert_eq!(parse_u64("-1.0"), None);
    }

    #[test]
    fn rejects_integral_decimal_overflow() {
        assert_eq!(parse_i64("9223372036854775808.0"), None);
        assert_eq!(parse_i32("2147483648.0"), None);
        assert_eq!(parse_u64("18446744073709551616.0"), None);
        assert_eq!(parse_u32("4294967296.0"), None);
    }
}
