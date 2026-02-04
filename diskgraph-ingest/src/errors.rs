use thiserror::Error;

#[derive(Clone, Copy, Debug, Error)]
pub enum KeyParseError {
    #[error("type mismatch")]
    TypeMismatch,
    #[error("unsupported type")]
    UnsupportedType,
    #[error("length mismatch")]
    LengthMismatch,
    #[error("empty")]
    Empty,
    #[error("too large")]
    TooLarge,
    #[error("not integer")]
    NotInteger,
    #[error("not base 10")]
    NotBase10,
    #[error("not utf8")]
    NotUtf8,
    #[error("invalid uuid")]
    InvalidUuid,
    #[error("invalid uuid 256")]
    InvalidUuid256,
    #[error("invalid ip")]
    InvalidIp,
}

pub type KeyParseResult<T> = std::result::Result<T, KeyParseError>;
