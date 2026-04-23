use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ---------------------------------------------------------------------------
// TimeUnit enum (pre-existing)
// ---------------------------------------------------------------------------

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

    /// Parse a unit name string (as used by Python callers).
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "nanoseconds" => Some(Self::Nanoseconds),
            "microseconds" => Some(Self::Microseconds),
            "milliseconds" => Some(Self::Milliseconds),
            "seconds" => Some(Self::Seconds),
            "minutes" => Some(Self::Minutes),
            "hours" => Some(Self::Hours),
            "days" => Some(Self::Days),
            "weeks" => Some(Self::Weeks),
            "months" => Some(Self::Months),
            "years" => Some(Self::Years),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// UnitSystem trait + Abs<U> / Dur<U> phantom-typed wrappers
// ---------------------------------------------------------------------------

/// Defines how raw i64 values map to real-world time for a given unit system.
///
/// Implementations provide construction from various source units and
/// extraction back to raw i64. Current implementations:
/// - [`Seconds`] — epoch seconds, used by rowpack storage and the core engine
/// - [`Nanoseconds`] — epoch nanoseconds, used at the PyO3/Arrow boundary
///   where timestamps arrive as `TimestampNanosecondArray`
///
/// Adding a new unit requires a new marker struct + impl block + one match arm
/// at the top-level dispatch site.
pub trait UnitSystem: 'static + Copy + Send + Sync + Eq + Ord + std::hash::Hash {
    fn abs_from_secs(secs: i64) -> Abs<Self>;
    fn abs_from_millis(ms: i64) -> Abs<Self>;
    fn abs_from_micros(us: i64) -> Abs<Self>;
    fn abs_from_nanos(ns: i64) -> Abs<Self>;
    fn abs_from_days(days: i32) -> Abs<Self>;
    fn abs_raw(ts: Abs<Self>) -> i64;
    fn abs_to_secs(ts: Abs<Self>) -> i64;

    fn dur_from_secs(secs: i64) -> Dur<Self>;
    fn dur_from_nanos(ns: i64) -> Dur<Self>;
    fn dur_from_days(days: i64) -> Dur<Self>;
    fn dur_raw(dur: Dur<Self>) -> i64;
    /// Convert `Abs<Self>` to nanoseconds (for interop with Arrow/nanosecond layers).
    fn abs_to_nanos(ts: Abs<Self>) -> i64;
}

/// An absolute timestamp in unit system `U`. Zero-cost wrapper over `i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Abs<U: UnitSystem>(i64, PhantomData<U>);

/// A duration in unit system `U`. Zero-cost wrapper over `i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Dur<U: UnitSystem>(i64, PhantomData<U>);

// -- Abs<U> convenience methods (delegate to UnitSystem) --------------------

impl<U: UnitSystem> Abs<U> {
    pub fn from_secs(secs: i64) -> Self {
        U::abs_from_secs(secs)
    }
    pub fn from_millis(ms: i64) -> Self {
        U::abs_from_millis(ms)
    }
    pub fn from_micros(us: i64) -> Self {
        U::abs_from_micros(us)
    }
    pub fn from_nanos(ns: i64) -> Self {
        U::abs_from_nanos(ns)
    }
    pub fn from_days(days: i32) -> Self {
        U::abs_from_days(days)
    }
    /// Create from a raw i64 already in the correct unit.
    /// Only use at I/O boundaries (rowpack read, Arrow output, PyO3).
    pub fn from_raw(raw: i64) -> Self {
        Abs(raw, PhantomData)
    }
    /// Extract the raw i64. Only use at I/O boundaries.
    pub fn raw(self) -> i64 {
        U::abs_raw(self)
    }
    /// Convert to epoch seconds (for chrono calendar arithmetic).
    pub fn to_secs(self) -> i64 {
        U::abs_to_secs(self)
    }
    /// Convert to nanoseconds (for interop with Arrow/nanosecond layers).
    pub fn to_nanos(self) -> i64 {
        U::abs_to_nanos(self)
    }
}

// -- Dur<U> convenience methods ---------------------------------------------

impl<U: UnitSystem> Dur<U> {
    pub fn from_secs(secs: i64) -> Self {
        U::dur_from_secs(secs)
    }
    pub fn from_nanos(ns: i64) -> Self {
        U::dur_from_nanos(ns)
    }
    pub fn from_millis(ms: i64) -> Self {
        U::dur_from_nanos(ms.saturating_mul(1_000_000))
    }
    pub fn from_micros(us: i64) -> Self {
        U::dur_from_nanos(us.saturating_mul(1_000))
    }
    pub fn from_minutes(mins: i64) -> Self {
        U::dur_from_secs(mins.saturating_mul(60))
    }
    pub fn from_hours(hours: i64) -> Self {
        U::dur_from_secs(hours.saturating_mul(3600))
    }
    pub fn from_days(days: i64) -> Self {
        U::dur_from_days(days)
    }
    pub fn from_weeks(weeks: i64) -> Self {
        U::dur_from_days(weeks.saturating_mul(7))
    }
    /// Create from a raw i64 already in the correct unit.
    pub fn from_raw(raw: i64) -> Self {
        Dur(raw, PhantomData)
    }
    /// Extract the raw i64.
    pub fn raw(self) -> i64 {
        U::dur_raw(self)
    }
    pub fn zero() -> Self {
        Dur(0, PhantomData)
    }
}

// -- Arithmetic: Abs<U> +/- Dur<U>, Abs<U> - Abs<U>, Dur<U> +/- Dur<U> ----

impl<U: UnitSystem> std::ops::Add<Dur<U>> for Abs<U> {
    type Output = Abs<U>;
    fn add(self, rhs: Dur<U>) -> Abs<U> {
        Abs(self.0 + rhs.0, PhantomData)
    }
}

impl<U: UnitSystem> std::ops::Sub<Dur<U>> for Abs<U> {
    type Output = Abs<U>;
    fn sub(self, rhs: Dur<U>) -> Abs<U> {
        Abs(self.0 - rhs.0, PhantomData)
    }
}

impl<U: UnitSystem> std::ops::Sub for Abs<U> {
    type Output = Dur<U>;
    fn sub(self, rhs: Abs<U>) -> Dur<U> {
        Dur(self.0 - rhs.0, PhantomData)
    }
}

impl<U: UnitSystem> std::ops::Add for Dur<U> {
    type Output = Dur<U>;
    fn add(self, rhs: Dur<U>) -> Dur<U> {
        Dur(self.0 + rhs.0, PhantomData)
    }
}

impl<U: UnitSystem> std::ops::Sub for Dur<U> {
    type Output = Dur<U>;
    fn sub(self, rhs: Dur<U>) -> Dur<U> {
        Dur(self.0 - rhs.0, PhantomData)
    }
}

impl<U: UnitSystem> std::ops::Mul<i64> for Dur<U> {
    type Output = Dur<U>;
    fn mul(self, rhs: i64) -> Dur<U> {
        Dur(self.0.saturating_mul(rhs), PhantomData)
    }
}

impl<U: UnitSystem> std::ops::Neg for Dur<U> {
    type Output = Dur<U>;
    fn neg(self) -> Dur<U> {
        Dur(-self.0, PhantomData)
    }
}

// -- Serde: transparent serialization as plain i64 --------------------------

impl<U: UnitSystem> Serialize for Abs<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de, U: UnitSystem> Deserialize<'de> for Abs<U> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        i64::deserialize(deserializer).map(|v| Abs(v, PhantomData))
    }
}

impl<U: UnitSystem> Serialize for Dur<U> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de, U: UnitSystem> Deserialize<'de> for Dur<U> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        i64::deserialize(deserializer).map(|v| Dur(v, PhantomData))
    }
}

// -- Seconds unit system ----------------------------------------------------

/// The seconds unit system. `Abs<Seconds>` stores epoch seconds, `Dur<Seconds>`
/// stores duration in seconds. This is the only implementation today — it
/// matches the rowpack storage contract where `access.get_timestamp()` always
/// returns epoch seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Seconds;

impl UnitSystem for Seconds {
    fn abs_from_secs(secs: i64) -> Abs<Self> {
        Abs(secs, PhantomData)
    }
    fn abs_from_millis(ms: i64) -> Abs<Self> {
        Abs(ms / 1_000, PhantomData)
    }
    fn abs_from_micros(us: i64) -> Abs<Self> {
        Abs(us / 1_000_000, PhantomData)
    }
    fn abs_from_nanos(ns: i64) -> Abs<Self> {
        Abs(ns / 1_000_000_000, PhantomData)
    }
    fn abs_from_days(days: i32) -> Abs<Self> {
        Abs(days as i64 * 86_400, PhantomData)
    }
    fn abs_raw(ts: Abs<Self>) -> i64 {
        ts.0
    }
    fn abs_to_secs(ts: Abs<Self>) -> i64 {
        ts.0
    }
    fn dur_from_secs(secs: i64) -> Dur<Self> {
        Dur(secs, PhantomData)
    }
    fn dur_from_nanos(ns: i64) -> Dur<Self> {
        Dur(ns / 1_000_000_000, PhantomData)
    }
    fn dur_from_days(days: i64) -> Dur<Self> {
        Dur(days * 86_400, PhantomData)
    }
    fn dur_raw(dur: Dur<Self>) -> i64 {
        dur.0
    }
    fn abs_to_nanos(ts: Abs<Self>) -> i64 {
        ts.0.saturating_mul(1_000_000_000)
    }
}

// -- Nanoseconds unit system ------------------------------------------------

/// The nanoseconds unit system. `Abs<Nanoseconds>` stores epoch nanoseconds,
/// `Dur<Nanoseconds>` stores duration in nanoseconds. Used at the PyO3/Arrow
/// boundary where timestamps arrive as `TimestampNanosecondArray`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Nanoseconds;

const NS_PER_SEC: i64 = 1_000_000_000;

impl UnitSystem for Nanoseconds {
    fn abs_from_secs(secs: i64) -> Abs<Self> {
        Abs(secs.saturating_mul(NS_PER_SEC), PhantomData)
    }
    fn abs_from_millis(ms: i64) -> Abs<Self> {
        Abs(ms.saturating_mul(1_000_000), PhantomData)
    }
    fn abs_from_micros(us: i64) -> Abs<Self> {
        Abs(us.saturating_mul(1_000), PhantomData)
    }
    fn abs_from_nanos(ns: i64) -> Abs<Self> {
        Abs(ns, PhantomData)
    }
    fn abs_from_days(days: i32) -> Abs<Self> {
        Abs((days as i64).saturating_mul(86_400).saturating_mul(NS_PER_SEC), PhantomData)
    }
    fn abs_raw(ts: Abs<Self>) -> i64 {
        ts.0
    }
    fn abs_to_secs(ts: Abs<Self>) -> i64 {
        ts.0 / NS_PER_SEC
    }
    fn dur_from_secs(secs: i64) -> Dur<Self> {
        Dur(secs.saturating_mul(NS_PER_SEC), PhantomData)
    }
    fn dur_from_nanos(ns: i64) -> Dur<Self> {
        Dur(ns, PhantomData)
    }
    fn dur_from_days(days: i64) -> Dur<Self> {
        Dur(days.saturating_mul(86_400).saturating_mul(NS_PER_SEC), PhantomData)
    }
    fn dur_raw(dur: Dur<Self>) -> i64 {
        dur.0
    }
    fn abs_to_nanos(ts: Abs<Self>) -> i64 {
        ts.0
    }
}

// -- Helpers for converting from a dynamic TimeUnit -------------------------

/// Create an `Abs<U>` from a raw value whose unit is given dynamically.
pub fn abs_from_unit<U: UnitSystem>(value: i64, unit: TimeUnit) -> Abs<U> {
    match unit {
        TimeUnit::Seconds => Abs::<U>::from_secs(value),
        TimeUnit::Milliseconds => Abs::<U>::from_millis(value),
        TimeUnit::Microseconds => Abs::<U>::from_micros(value),
        TimeUnit::Nanoseconds => Abs::<U>::from_nanos(value),
        TimeUnit::Days => Abs::<U>::from_days(value as i32),
        other => panic!("unsupported unit for absolute timestamp: {:?}", other),
    }
}

/// Create a `Dur<U>` from a raw value whose unit is given dynamically.
pub fn dur_from_unit<U: UnitSystem>(value: i64, unit: TimeUnit) -> Dur<U> {
    match unit {
        TimeUnit::Nanoseconds => Dur::<U>::from_nanos(value),
        TimeUnit::Microseconds => Dur::<U>::from_micros(value),
        TimeUnit::Milliseconds => Dur::<U>::from_millis(value),
        TimeUnit::Seconds => Dur::<U>::from_secs(value),
        TimeUnit::Minutes => Dur::<U>::from_minutes(value),
        TimeUnit::Hours => Dur::<U>::from_hours(value),
        TimeUnit::Days => Dur::<U>::from_days(value),
        TimeUnit::Weeks => Dur::<U>::from_weeks(value),
        // Months/Years are calendar-relative, not fixed durations.
        // Use add_calendar_months() instead.
        other => panic!("unsupported unit for duration: {:?}", other),
    }
}

/// Add calendar months to a timestamp. Uses chrono for correct month
/// boundaries (e.g. Jan 31 + 1 month = Feb 28). Matches
/// `pd.DateOffset(months=N)` behavior.
pub fn add_calendar_months<U: UnitSystem>(ts: Abs<U>, months: i32) -> Abs<U> {
    use chrono::{DateTime, Datelike, NaiveDate, Timelike};

    let secs = ts.to_secs();
    let dt = DateTime::from_timestamp(secs, 0)
        .expect("timestamp out of range for calendar arithmetic")
        .naive_utc();

    let total_months = dt.month0() as i32 + months;
    // Euclidean division to handle negative months correctly
    let target_year = dt.year() + total_months.div_euclid(12);
    let target_month = (total_months.rem_euclid(12) + 1) as u32;

    // Clamp day to last day of target month (e.g. Jan 31 + 1 mo = Feb 28)
    let max_day = days_in_month(target_year, target_month);
    let day = dt.day().min(max_day);

    let result = NaiveDate::from_ymd_opt(target_year, target_month, day)
        .expect("invalid date in calendar arithmetic")
        .and_hms_opt(dt.hour(), dt.minute(), dt.second())
        .expect("invalid time in calendar arithmetic");

    Abs::<U>::from_secs(result.and_utc().timestamp())
}

/// Add an offset to a timestamp. For fixed-size units (seconds through weeks)
/// this is pure arithmetic. For months/years this delegates to calendar
/// arithmetic via [`add_calendar_months`].
///
/// This is the typed replacement for the ad-hoc `add_unit` that operated on
/// raw nanosecond i64 values.
pub fn add_offset<U: UnitSystem>(ts: Abs<U>, unit: TimeUnit, value: i64) -> Abs<U> {
    match unit {
        TimeUnit::Months => add_calendar_months(ts, value as i32),
        TimeUnit::Years => add_calendar_months(ts, (value * 12) as i32),
        _ => ts + dur_from_unit(value, unit),
    }
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => unreachable!("invalid month: {}", month),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // -- Construction tests -------------------------------------------------

    #[test]
    fn abs_from_secs_identity_for_seconds_unit() {
        let ts = Abs::<Seconds>::from_secs(1_000_000);
        assert_eq!(ts.raw(), 1_000_000);
    }

    #[test]
    fn abs_from_nanos_divides_for_seconds_unit() {
        // 1_500_000_000 ns = 1.5 sec → truncates to 1
        let ts = Abs::<Seconds>::from_nanos(1_500_000_000);
        assert_eq!(ts.raw(), 1);
    }

    #[test]
    fn abs_from_millis_divides_for_seconds_unit() {
        let ts = Abs::<Seconds>::from_millis(3_500);
        assert_eq!(ts.raw(), 3);
    }

    #[test]
    fn abs_from_micros_divides_for_seconds_unit() {
        let ts = Abs::<Seconds>::from_micros(2_500_000);
        assert_eq!(ts.raw(), 2);
    }

    #[test]
    fn abs_from_days_multiplies_for_seconds_unit() {
        let ts = Abs::<Seconds>::from_days(1);
        assert_eq!(ts.raw(), 86_400);
    }

    #[test]
    fn dur_from_secs_identity() {
        let d = Dur::<Seconds>::from_secs(3600);
        assert_eq!(d.raw(), 3600);
    }

    #[test]
    fn dur_from_days_multiplies() {
        let d = Dur::<Seconds>::from_days(7);
        assert_eq!(d.raw(), 7 * 86_400);
    }

    #[test]
    fn dur_zero() {
        assert_eq!(Dur::<Seconds>::zero().raw(), 0);
    }

    // -- from_raw round-trip ------------------------------------------------

    #[test]
    fn abs_from_raw_round_trip() {
        let raw = 1_713_500_000i64;
        let ts = Abs::<Seconds>::from_raw(raw);
        assert_eq!(ts.raw(), raw);
    }

    #[test]
    fn dur_from_raw_round_trip() {
        let raw = 86_400i64;
        let d = Dur::<Seconds>::from_raw(raw);
        assert_eq!(d.raw(), raw);
    }

    // -- to_secs ------------------------------------------------------------

    #[test]
    fn abs_to_secs_identity_for_seconds_unit() {
        let ts = Abs::<Seconds>::from_secs(42);
        assert_eq!(ts.to_secs(), 42);
    }

    // -- Arithmetic ---------------------------------------------------------

    #[test]
    fn abs_plus_dur_yields_abs() {
        let ts = Abs::<Seconds>::from_secs(1000);
        let d = Dur::<Seconds>::from_secs(200);
        let result = ts + d;
        assert_eq!(result.raw(), 1200);
    }

    #[test]
    fn abs_minus_dur_yields_abs() {
        let ts = Abs::<Seconds>::from_secs(1000);
        let d = Dur::<Seconds>::from_secs(200);
        let result = ts - d;
        assert_eq!(result.raw(), 800);
    }

    #[test]
    fn abs_minus_abs_yields_dur() {
        let a = Abs::<Seconds>::from_secs(1000);
        let b = Abs::<Seconds>::from_secs(300);
        let d: Dur<Seconds> = a - b;
        assert_eq!(d.raw(), 700);
    }

    #[test]
    fn dur_plus_dur() {
        let a = Dur::<Seconds>::from_secs(100);
        let b = Dur::<Seconds>::from_secs(200);
        assert_eq!((a + b).raw(), 300);
    }

    #[test]
    fn dur_minus_dur() {
        let a = Dur::<Seconds>::from_secs(500);
        let b = Dur::<Seconds>::from_secs(200);
        assert_eq!((a - b).raw(), 300);
    }

    #[test]
    fn dur_mul_scalar() {
        let d = Dur::<Seconds>::from_secs(100);
        assert_eq!((d * 3).raw(), 300);
    }

    #[test]
    fn dur_neg() {
        let d = Dur::<Seconds>::from_secs(100);
        assert_eq!((-d).raw(), -100);
    }

    // -- Ordering -----------------------------------------------------------

    #[test]
    fn abs_ordering() {
        let a = Abs::<Seconds>::from_secs(100);
        let b = Abs::<Seconds>::from_secs(200);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, Abs::<Seconds>::from_secs(100));
    }

    #[test]
    fn dur_ordering() {
        let a = Dur::<Seconds>::from_secs(100);
        let b = Dur::<Seconds>::from_secs(200);
        assert!(a < b);
    }

    // -- Hash ---------------------------------------------------------------

    #[test]
    fn abs_in_hashset() {
        let mut set = HashSet::new();
        set.insert(Abs::<Seconds>::from_secs(100));
        set.insert(Abs::<Seconds>::from_secs(200));
        set.insert(Abs::<Seconds>::from_secs(100)); // duplicate
        assert_eq!(set.len(), 2);
    }

    // -- repr(transparent): same size as i64 --------------------------------

    #[test]
    fn abs_same_size_as_i64() {
        assert_eq!(
            std::mem::size_of::<Abs<Seconds>>(),
            std::mem::size_of::<i64>()
        );
    }

    #[test]
    fn dur_same_size_as_i64() {
        assert_eq!(
            std::mem::size_of::<Dur<Seconds>>(),
            std::mem::size_of::<i64>()
        );
    }

    // -- Serde: transparent as plain i64 ------------------------------------

    #[test]
    fn abs_serde_json_round_trip() {
        let ts = Abs::<Seconds>::from_secs(1_713_500_000);
        let json = serde_json::to_string(&ts).unwrap();
        assert_eq!(json, "1713500000");
        let back: Abs<Seconds> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ts);
    }

    #[test]
    fn dur_serde_json_round_trip() {
        let d = Dur::<Seconds>::from_secs(86400);
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, "86400");
        let back: Dur<Seconds> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn abs_serde_backward_compat_with_raw_i64() {
        // A plan serialized with raw i64 should deserialize into Abs<Seconds>
        let json = "1713500000";
        let ts: Abs<Seconds> = serde_json::from_str(json).unwrap();
        assert_eq!(ts.raw(), 1_713_500_000);
    }

    // -- abs_from_unit / dur_from_unit dynamic dispatch ----------------------

    #[test]
    fn abs_from_unit_seconds() {
        let ts = abs_from_unit::<Seconds>(1000, TimeUnit::Seconds);
        assert_eq!(ts.raw(), 1000);
    }

    #[test]
    fn abs_from_unit_nanos() {
        let ts = abs_from_unit::<Seconds>(1_500_000_000_000, TimeUnit::Nanoseconds);
        assert_eq!(ts.raw(), 1500);
    }

    #[test]
    fn abs_from_unit_millis() {
        let ts = abs_from_unit::<Seconds>(5_000, TimeUnit::Milliseconds);
        assert_eq!(ts.raw(), 5);
    }

    #[test]
    fn abs_from_unit_days() {
        let ts = abs_from_unit::<Seconds>(7, TimeUnit::Days);
        assert_eq!(ts.raw(), 7 * 86_400);
    }

    #[test]
    fn dur_from_unit_seconds() {
        let d = dur_from_unit::<Seconds>(3600, TimeUnit::Seconds);
        assert_eq!(d.raw(), 3600);
    }

    #[test]
    fn dur_from_unit_days() {
        let d = dur_from_unit::<Seconds>(2, TimeUnit::Days);
        assert_eq!(d.raw(), 2 * 86_400);
    }

    #[test]
    #[should_panic(expected = "unsupported unit for absolute timestamp")]
    fn abs_from_unit_rejects_months() {
        abs_from_unit::<Seconds>(1, TimeUnit::Months);
    }

    #[test]
    fn dur_from_unit_hours() {
        let d = dur_from_unit::<Seconds>(2, TimeUnit::Hours);
        assert_eq!(d.raw(), 2 * 3600);
    }

    #[test]
    fn dur_from_unit_minutes() {
        let d = dur_from_unit::<Seconds>(5, TimeUnit::Minutes);
        assert_eq!(d.raw(), 5 * 60);
    }

    #[test]
    fn dur_from_unit_weeks() {
        let d = dur_from_unit::<Seconds>(3, TimeUnit::Weeks);
        assert_eq!(d.raw(), 3 * 7 * 86_400);
    }

    #[test]
    #[should_panic(expected = "unsupported unit for duration")]
    fn dur_from_unit_rejects_months() {
        dur_from_unit::<Seconds>(1, TimeUnit::Months);
    }

    // -- Calendar month arithmetic ------------------------------------------

    #[test]
    fn add_one_month_from_jan() {
        // 2024-01-15 00:00:00 UTC
        let ts = Abs::<Seconds>::from_secs(1_705_276_800);
        let result = add_calendar_months(ts, 1);
        // 2024-02-15 00:00:00 UTC
        assert_eq!(result.raw(), 1_707_955_200);
    }

    #[test]
    fn add_month_clamps_day_jan31_to_feb28() {
        // 2023-01-31 00:00:00 UTC = 1675123200
        let ts = Abs::<Seconds>::from_secs(1_675_123_200);
        let result = add_calendar_months(ts, 1);
        // 2023-02-28 00:00:00 UTC = 1677542400
        assert_eq!(result.raw(), 1_677_542_400);
    }

    #[test]
    fn subtract_one_month() {
        // 2024-03-15 00:00:00 UTC = 1710460800
        let ts = Abs::<Seconds>::from_secs(1_710_460_800);
        let result = add_calendar_months(ts, -1);
        // 2024-02-15 00:00:00 UTC = 1707955200
        assert_eq!(result.raw(), 1_707_955_200);
    }

    #[test]
    fn add_twelve_months_equals_one_year() {
        // 2024-06-01 00:00:00 UTC = 1717200000
        let ts = Abs::<Seconds>::from_secs(1_717_200_000);
        let result = add_calendar_months(ts, 12);
        // 2025-06-01 00:00:00 UTC = 1748736000
        assert_eq!(result.raw(), 1_748_736_000);
    }

    // -- Bincode serde (used by sortkey/sampling) ---------------------------

    #[test]
    fn abs_bincode_round_trip() {
        let ts = Abs::<Seconds>::from_secs(1_713_500_000);
        let bytes = bincode::serialize(&ts).unwrap();
        // Should be exactly 8 bytes (i64)
        assert_eq!(bytes.len(), 8);
        let back: Abs<Seconds> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, ts);
    }

    #[test]
    fn dur_bincode_round_trip() {
        let d = Dur::<Seconds>::from_secs(86400);
        let bytes = bincode::serialize(&d).unwrap();
        assert_eq!(bytes.len(), 8);
        let back: Dur<Seconds> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, d);
    }

    // -- Nanoseconds unit system tests ----------------------------------------

    #[test]
    fn nanoseconds_abs_from_raw_round_trip() {
        let raw = 1_713_500_000_000_000_000i64; // epoch nanos
        let ts = Abs::<Nanoseconds>::from_raw(raw);
        assert_eq!(ts.raw(), raw);
    }

    #[test]
    fn nanoseconds_abs_from_secs() {
        let ts = Abs::<Nanoseconds>::from_secs(1000);
        assert_eq!(ts.raw(), 1_000_000_000_000);
    }

    #[test]
    fn nanoseconds_abs_from_millis() {
        let ts = Abs::<Nanoseconds>::from_millis(5_000);
        assert_eq!(ts.raw(), 5_000_000_000);
    }

    #[test]
    fn nanoseconds_abs_from_nanos_identity() {
        let ts = Abs::<Nanoseconds>::from_nanos(42);
        assert_eq!(ts.raw(), 42);
    }

    #[test]
    fn nanoseconds_abs_to_secs() {
        let ts = Abs::<Nanoseconds>::from_raw(1_500_000_000_000i64); // 1500 nanos-epoch
        assert_eq!(ts.to_secs(), 1500);
    }

    #[test]
    fn nanoseconds_abs_to_nanos_identity() {
        let ts = Abs::<Nanoseconds>::from_raw(42_000_000_000i64);
        assert_eq!(ts.to_nanos(), 42_000_000_000);
    }

    #[test]
    fn nanoseconds_dur_from_secs() {
        let d = Dur::<Nanoseconds>::from_secs(2);
        assert_eq!(d.raw(), 2_000_000_000);
    }

    #[test]
    fn nanoseconds_dur_from_nanos_identity() {
        let d = Dur::<Nanoseconds>::from_nanos(42);
        assert_eq!(d.raw(), 42);
    }

    #[test]
    fn nanoseconds_dur_from_days() {
        let d = Dur::<Nanoseconds>::from_days(1);
        assert_eq!(d.raw(), 86_400_000_000_000);
    }

    #[test]
    fn nanoseconds_dur_from_hours() {
        let d = Dur::<Nanoseconds>::from_hours(1);
        assert_eq!(d.raw(), 3_600_000_000_000);
    }

    #[test]
    fn nanoseconds_dur_from_weeks() {
        let d = Dur::<Nanoseconds>::from_weeks(1);
        assert_eq!(d.raw(), 7 * 86_400_000_000_000);
    }

    #[test]
    fn nanoseconds_abs_plus_dur() {
        let ts = Abs::<Nanoseconds>::from_secs(1000);
        let d = Dur::<Nanoseconds>::from_secs(200);
        let result = ts + d;
        assert_eq!(result.raw(), 1_200_000_000_000);
    }

    #[test]
    fn nanoseconds_abs_minus_abs_yields_dur() {
        let a = Abs::<Nanoseconds>::from_secs(1000);
        let b = Abs::<Nanoseconds>::from_secs(300);
        let d: Dur<Nanoseconds> = a - b;
        assert_eq!(d.raw(), 700_000_000_000);
    }

    #[test]
    fn nanoseconds_same_size_as_i64() {
        assert_eq!(std::mem::size_of::<Abs<Nanoseconds>>(), std::mem::size_of::<i64>());
        assert_eq!(std::mem::size_of::<Dur<Nanoseconds>>(), std::mem::size_of::<i64>());
    }

    // -- add_offset tests -----------------------------------------------------

    #[test]
    fn add_offset_days_nanoseconds() {
        let ts = Abs::<Nanoseconds>::from_secs(1000);
        let result = add_offset(ts, TimeUnit::Days, 2);
        assert_eq!(result.raw(), ts.raw() + 2 * 86_400_000_000_000);
    }

    #[test]
    fn add_offset_months_nanoseconds() {
        // 2024-01-15 00:00:00 UTC in nanos
        let ts = Abs::<Nanoseconds>::from_secs(1_705_276_800);
        let result = add_offset(ts, TimeUnit::Months, 1);
        // 2024-02-15 00:00:00 UTC in nanos
        let expected = Abs::<Nanoseconds>::from_secs(1_707_955_200);
        assert_eq!(result.raw(), expected.raw());
    }

    #[test]
    fn add_offset_years_nanoseconds() {
        // 2024-01-15 00:00:00 UTC in nanos
        let ts = Abs::<Nanoseconds>::from_secs(1_705_276_800);
        let result = add_offset(ts, TimeUnit::Years, 1);
        // 2025-01-15 00:00:00 UTC in nanos
        let expected = Abs::<Nanoseconds>::from_secs(1_736_899_200);
        assert_eq!(result.raw(), expected.raw());
    }

    // -- TimeUnit::from_name tests -------------------------------------------

    #[test]
    fn time_unit_from_name_all_valid() {
        assert_eq!(TimeUnit::from_name("seconds"), Some(TimeUnit::Seconds));
        assert_eq!(TimeUnit::from_name("nanoseconds"), Some(TimeUnit::Nanoseconds));
        assert_eq!(TimeUnit::from_name("days"), Some(TimeUnit::Days));
        assert_eq!(TimeUnit::from_name("months"), Some(TimeUnit::Months));
        assert_eq!(TimeUnit::from_name("years"), Some(TimeUnit::Years));
        assert_eq!(TimeUnit::from_name("hours"), Some(TimeUnit::Hours));
        assert_eq!(TimeUnit::from_name("minutes"), Some(TimeUnit::Minutes));
        assert_eq!(TimeUnit::from_name("weeks"), Some(TimeUnit::Weeks));
        assert_eq!(TimeUnit::from_name("milliseconds"), Some(TimeUnit::Milliseconds));
        assert_eq!(TimeUnit::from_name("microseconds"), Some(TimeUnit::Microseconds));
    }

    #[test]
    fn time_unit_from_name_invalid() {
        assert_eq!(TimeUnit::from_name("invalid"), None);
        assert_eq!(TimeUnit::from_name(""), None);
    }
}
