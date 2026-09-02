use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum DecimalKeyError {
    #[error("empty")]
    Empty,
    #[error("not base 10")]
    NotBase10,
    #[error("not utf8")]
    NotUtf8,
}

/// Normalize a base-10 decimal key for stable PK hashing and comparison.
///
/// Leading integer zeroes and trailing fractional zeroes are removed while
/// preserving the represented value. Negative zero is normalized to zero.
pub fn normalize_base10_key_into(
    out: &mut Vec<u8>,
    bytes: &[u8],
) -> Result<usize, DecimalKeyError> {
    let text = std::str::from_utf8(bytes).map_err(|_| DecimalKeyError::NotUtf8)?;
    if text.is_empty() {
        return Err(DecimalKeyError::Empty);
    }

    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(unsigned) => (true, unsigned),
        None => (false, text),
    };
    let mut parts = unsigned.split('.');
    let int_part = parts.next().unwrap();
    if int_part.is_empty() {
        return Err(DecimalKeyError::NotBase10);
    }
    let frac_part = parts.next();
    if parts.next().is_some()
        || !int_part.bytes().all(|byte| byte.is_ascii_digit())
        || frac_part.is_some_and(|frac| !frac.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(DecimalKeyError::NotBase10);
    }

    let int_trimmed = int_part.trim_start_matches('0');
    let int_out = if int_trimmed.is_empty() {
        "0"
    } else {
        int_trimmed
    };
    let frac_out = frac_part.unwrap_or("").trim_end_matches('0');
    let is_zero = int_out == "0" && frac_out.is_empty();
    let start = out.len();
    if negative && !is_zero {
        out.push(b'-');
    }
    out.extend_from_slice(int_out.as_bytes());
    if !frac_out.is_empty() {
        out.push(b'.');
        out.extend_from_slice(frac_out.as_bytes());
    }
    Ok(out.len() - start)
}

pub fn normalize_base10_key(bytes: &[u8]) -> Result<Vec<u8>, DecimalKeyError> {
    let mut out = Vec::with_capacity(bytes.len());
    normalize_base10_key_into(&mut out, bytes)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_equivalent_decimal_keys() {
        for (input, expected) in [
            ("000123.4500", "123.45"),
            ("123.00", "123"),
            ("-00123.4500", "-123.45"),
            ("-0.000", "0"),
            ("0.", "0"),
        ] {
            assert_eq!(
                normalize_base10_key(input.as_bytes()).unwrap(),
                expected.as_bytes()
            );
        }
    }

    #[test]
    fn rejects_non_decimal_keys() {
        for input in ["", "-", ".5", "+1", "1.2.3", "1e3", "a"] {
            assert!(normalize_base10_key(input.as_bytes()).is_err(), "{input}");
        }
    }
}
