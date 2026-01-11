use std::env;
use std::fs;

use anyhow::{anyhow, Result};
use diskgraph_rowpack::{ColumnEncoding, DecimalStorage, FloatStorage, RowPackContext, StringEncoding};

const MAX_DICT_PREVIEW: usize = 10;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(path) => path,
        None => {
            return Err(anyhow!(
                "usage: rowpack_ctx_inspect <context.ctx>"
            ));
        }
    };
    if args.next().is_some() {
        return Err(anyhow!("usage: rowpack_ctx_inspect <context.ctx>"));
    }
    let bytes = fs::read(&path)?;
    let ctx = RowPackContext::from_bytes(&bytes)?;
    println!("Rowpack context: {}", path);
    println!("Version: {}", ctx.version);
    println!("Estimated row bytes: {}", ctx.estimated_row_bytes);
    println!("Columns: {}", ctx.columns.len());
    for (idx, col) in ctx.columns.iter().enumerate() {
        println!();
        println!(
            "[{}] {} ({:?})",
            idx,
            col.name,
            col.logical_type
        );
        print_encoding(&col.encoding)?;
    }
    Ok(())
}

fn print_encoding(encoding: &ColumnEncoding) -> Result<()> {
    match encoding {
        ColumnEncoding::Int { signed, varint, bits } => {
            println!("  Encoding: int (signed={}, varint={}, bits={})", signed, varint, bits);
        }
        ColumnEncoding::Float { storage } => {
            println!("  Encoding: float ({})", format_float_storage(*storage));
        }
        ColumnEncoding::Bool => {
            println!("  Encoding: bool");
        }
        ColumnEncoding::String { encoding } => {
            print_string_encoding(encoding)?;
        }
        ColumnEncoding::Bytes { fixed_len } => {
            if let Some(len) = fixed_len {
                println!("  Encoding: bytes (fixed_len={})", len);
            } else {
                println!("  Encoding: bytes");
            }
        }
        ColumnEncoding::Date { varint, bits } => {
            println!("  Encoding: date (varint={}, bits={})", varint, bits);
        }
        ColumnEncoding::Timestamp => {
            println!("  Encoding: timestamp");
        }
        ColumnEncoding::Decimal {
            precision,
            scale,
            bytes,
            storage,
        } => {
            println!(
                "  Encoding: decimal (precision={}, scale={}, bytes={}, storage={})",
                precision,
                scale,
                bytes,
                format_decimal_storage(storage)
            );
        }
        ColumnEncoding::FloatArray { storage } => {
            println!(
                "  Encoding: float_array ({})",
                format_float_storage(*storage)
            );
        }
    }
    Ok(())
}

fn print_string_encoding(encoding: &StringEncoding) -> Result<()> {
    match encoding {
        StringEncoding::Raw => {
            println!("  Encoding: string raw");
        }
        StringEncoding::Dict { values } => {
            println!("  Encoding: string dict (entries={})", values.len());
            let preview = values.iter().take(MAX_DICT_PREVIEW);
            for (idx, value) in preview.enumerate() {
                println!(
                    "    [{}] {}",
                    idx,
                    format_bytes(value)
                );
            }
        }
        StringEncoding::Alphabet { alphabet } => {
            println!("  Encoding: string alphabet (len={})", alphabet.len());
            println!("    Alphabet: {}", format_bytes(alphabet));
        }
        StringEncoding::Bpe { dict } => {
            println!("  Encoding: string bpe (entries={})", dict.len());
            let preview = dict.iter().take(MAX_DICT_PREVIEW);
            for (idx, value) in preview.enumerate() {
                println!("    [{}] {}", idx, format_bytes(value));
            }
        }
        StringEncoding::Lz4 => {
            println!("  Encoding: string lz4");
        }
    }
    Ok(())
}

fn format_float_storage(storage: FloatStorage) -> &'static str {
    match storage {
        FloatStorage::F16 => "f16",
        FloatStorage::F32 => "f32",
        FloatStorage::F64 => "f64",
    }
}

fn format_decimal_storage(storage: &DecimalStorage) -> String {
    match storage {
        DecimalStorage::Raw => "raw".to_string(),
        DecimalStorage::Int { varint, bits } => {
            format!("int(varint={}, bits={})", varint, bits)
        }
    }
}

fn format_bytes(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        if text.chars().all(|ch| ch.is_ascii_graphic() || ch == ' ') {
            return format!("{:?}", text);
        }
    }
    let mut out = String::with_capacity(bytes.len() * 2 + 2);
    out.push_str("0x");
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}
