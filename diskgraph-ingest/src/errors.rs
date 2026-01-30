#[derive(Clone, Copy, Debug)]
pub enum KeyParseError {
    TypeMismatch,
    UnsupportedType,
    LengthMismatch,
    Empty,
    TooLarge,
    NotInteger,
    NotBase10,
    NotUtf8,
    InvalidUuid,
    InvalidUuid256,
    InvalidIp,
}

pub type KeyParseResult<T> = std::result::Result<T, KeyParseError>;
