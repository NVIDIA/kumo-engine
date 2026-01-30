use uuid_simd::{parse_hyphenated, parse_simple, AsOut};
use anyhow::{anyhow, Result};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use chrono::{NaiveDate, NaiveDateTime};
use crate::MAX_INPUT_STRING_BYTES;

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
    value.trim().parse::<i64>().ok()
}

pub(crate) fn parse_i32(value: &str) -> Option<i32> {
    value.trim().parse::<i32>().ok()
}

pub(crate) fn parse_u64(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

pub(crate) fn parse_u32(value: &str) -> Option<u32> {
    value.trim().parse::<u32>().ok()
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

pub(crate) fn parse_timestamp_text(value: &str, format: Option<&str>) -> Option<(i64, u32)> {
    let ts = if let Some(fmt) = format {
        let has_tz = fmt.contains("%z") || fmt.contains("%:z");
        if has_tz {
            chrono::DateTime::parse_from_str(value, fmt)
                .ok()
                .map(|v| v.naive_utc())?
        } else {
            NaiveDateTime::parse_from_str(value, fmt).ok().or_else(|| {
                NaiveDate::parse_from_str(value, fmt)
                    .ok()
                    .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
            })?
        }
    } else {
        chrono::DateTime::parse_from_rfc3339(value)
            .ok()
            .map(|v| v.naive_utc())?
    };
    let ts = ts.and_utc();
    Some((ts.timestamp(), ts.timestamp_subsec_nanos()))
}

pub(crate) fn parse_date_text(value: &str, format: Option<&str>) -> Option<i32> {
    let dt = if let Some(fmt) = format {
        NaiveDate::parse_from_str(value, fmt).ok().or_else(|| {
            NaiveDateTime::parse_from_str(value, fmt)
                .ok()
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
            _ => return None,
        };
        Some(Self { kind })
    }

    pub(crate) fn parse_timestamp(&self, value: &str) -> Option<(i64, u32)> {
        match self.kind {
            TimestampFormatKind::Ymd => {
                return parse_ymd_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmSpace => {
                return parse_ymd_hm_space_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpace => {
                return parse_ymd_hms_space_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceFrac => {
                return parse_ymd_hms_space_frac_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsT => {
                return parse_ymd_hms_space_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsTFrac => {
                return parse_ymd_hms_space_frac_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffset => {
                return parse_ymd_hms_offset_nocolon_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffsetColon => {
                return parse_ymd_hms_offset_colon_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffsetSpaced => {
                return parse_ymd_hms_offset_nocolon_spaced_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsSpaceOffsetColonSpaced => {
                return parse_ymd_hms_offset_colon_spaced_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsTOffset => {
                return parse_ymd_hms_offset_nocolon_fast(value.as_bytes());
            }
            TimestampFormatKind::YmdHmsTOffsetColon => {
                return parse_ymd_hms_offset_colon_fast(value.as_bytes());
            }
        }
    }

    pub(crate) fn parse_date(&self, value: &str) -> Option<i32> {
        let (secs, _nanos) = self.parse_timestamp(value)?;
        Some(days_from_secs(secs))
    }
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

pub(crate) fn fraction_to_nanos(value: u32, digits: usize) -> u32 {
    const POW10: [u32; 10] = [
        1_000_000_000,
        100_000_000,
        10_000_000,
        1_000_000,
        100_000,
        10_000,
        1_000,
        100,
        10,
        1,
    ];
    if digits == 0 {
        0
    } else if digits >= 9 {
        value
    } else {
        value.saturating_mul(POW10[digits])
    }
}

// "%Y-%m-%d"
pub(crate) fn parse_ymd_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 10 {
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
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M"
pub(crate) fn parse_ymd_hm_space_fast(bytes: &[u8]) -> Option<(i64, u32)> {
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
    Some((secs, 0))
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
pub(crate) fn parse_ymd_hms_space_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 19 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S%.f" / "%Y-%m-%dT%H:%M:%S%.f"
pub(crate) fn parse_ymd_hms_space_frac_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() < 21 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let (fraction, digits, next) = parse_fraction(bytes, 20)?;
    if next != bytes.len() {
        return None;
    }
    let nanos = fraction_to_nanos(fraction, digits);
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second);
    Some((secs, nanos))
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
pub(crate) fn parse_ymd_hms_offset_nocolon_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 24 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm(bytes, 19)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S%:z" / "%Y-%m-%dT%H:%M:%S%:z"
pub(crate) fn parse_ymd_hms_offset_colon_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 25 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm_colon(bytes, 19)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S %z"
pub(crate) fn parse_ymd_hms_offset_nocolon_spaced_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 25 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm(bytes, 20)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
}

// "%Y-%m-%d %H:%M:%S %:z"
pub(crate) fn parse_ymd_hms_offset_colon_spaced_fast(bytes: &[u8]) -> Option<(i64, u32)> {
    if bytes.len() != 26 {
        return None;
    }
    let (year, month, day, hour, minute, second) = parse_ymd_hms_digits(bytes)?;
    let offset_secs = parse_offset_hhmm_colon(bytes, 20)?;
    let secs = timestamp_from_parts_fast(year as i32, month, day, hour, minute, second)
        .saturating_sub(offset_secs);
    Some((secs, 0))
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

pub(crate) fn timestamp_from_int(value: i64, unit: Option<arrow_schema::TimeUnit>) -> (i64, u32) {
    let unit = unit.unwrap_or(arrow_schema::TimeUnit::Second);
    match unit {
        arrow_schema::TimeUnit::Second => (value, 0),
        arrow_schema::TimeUnit::Millisecond => {
            let secs = value.div_euclid(1_000);
            let millis = value.rem_euclid(1_000) as u32;
            (secs, millis * 1_000_000)
        }
        arrow_schema::TimeUnit::Microsecond => {
            let secs = value.div_euclid(1_000_000);
            let micros = value.rem_euclid(1_000_000) as u32;
            (secs, micros * 1_000)
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let secs = value.div_euclid(1_000_000_000);
            let nanos = value.rem_euclid(1_000_000_000) as u32;
            (secs, nanos)
        }
    }
}

pub(crate) fn days_from_secs(secs: i64) -> i32 {
    secs.div_euclid(86_400) as i32
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

pub(crate) fn timestamp_parts_from_array(array: &ArrayRef, row: usize) -> Result<(i64, u32)> {
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
            Ok((arr.value(row), 0))
        }
        arrow_schema::TimeUnit::Millisecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                .ok_or_else(|| anyhow!("timestamp millisecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000);
            let millis = value.rem_euclid(1_000) as u32;
            Ok((secs, millis * 1_000_000))
        }
        arrow_schema::TimeUnit::Microsecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                .ok_or_else(|| anyhow!("timestamp microsecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000_000);
            let micros = value.rem_euclid(1_000_000) as u32;
            Ok((secs, micros * 1_000))
        }
        arrow_schema::TimeUnit::Nanosecond => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                .ok_or_else(|| anyhow!("timestamp nanosecond cast"))?;
            let value = arr.value(row);
            let secs = value.div_euclid(1_000_000_000);
            let nanos = value.rem_euclid(1_000_000_000) as u32;
            Ok((secs, nanos))
        }
    }
}


