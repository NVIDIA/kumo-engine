//! Bounded diagnostics for failures crossing a process boundary.
//!
//! Unmarked first-party failures are visible to operators by default. Call
//! sites that receive data-bearing or provider errors wrap them in
//! [`RestrictedDiagnostic`], whose source is retained for control flow but is
//! never rendered into the ordinary diagnostic fields. Explicitly approved
//! public detail is carried as a separate projection in the same envelope.

use anyhow::{anyhow, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DIAGNOSTIC_FILENAME: &str = ".diagnostic.json";
pub const WORKER_DIAGNOSTIC_FILENAME: &str = ".diagnostic.worker.json";
pub const DIAGNOSTIC_MAX_BYTES: usize = 32 * 1024;

const DIAGNOSTIC_VERSION: u8 = 1;
const MAX_SUMMARY_BYTES: usize = 2 * 1024;
const MAX_CHAIN_ENTRIES: usize = 8;
const MAX_CHAIN_SUMMARY_BYTES: usize = 2 * 1024;
static NEXT_DIAGNOSTIC_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticCategory {
    Internal,
    Validation,
    Provider,
    Io,
    Resource,
    Process,
}

impl DiagnosticCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::Internal => "internal",
            Self::Validation => "validation",
            Self::Provider => "provider",
            Self::Io => "io",
            Self::Resource => "resource",
            Self::Process => "process",
        }
    }
}

/// Marks a returned error chain as unsafe for automatic message rendering.
///
/// `Display` deliberately exposes only `safe_summary`. The original error is
/// still available as `source()` for retry decisions and local debugging.
#[derive(Debug)]
pub struct RestrictedDiagnostic {
    code: &'static str,
    safe_summary: &'static str,
    operation: &'static str,
    category: DiagnosticCategory,
    retryable: bool,
    source: anyhow::Error,
}

impl RestrictedDiagnostic {
    pub fn new(
        code: &'static str,
        safe_summary: &'static str,
        operation: &'static str,
        category: DiagnosticCategory,
        retryable: bool,
        source: impl Into<anyhow::Error>,
    ) -> Self {
        Self {
            code,
            safe_summary,
            operation,
            category,
            retryable,
            source: source.into(),
        }
    }
}

impl fmt::Display for RestrictedDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.safe_summary)
    }
}

impl StdError for RestrictedDiagnostic {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Apply the restricted marker after a boundary has made its retry decision.
pub fn restrict<T, E>(
    result: std::result::Result<T, E>,
    code: &'static str,
    safe_summary: &'static str,
    operation: &'static str,
    category: DiagnosticCategory,
    retryable: bool,
) -> Result<T>
where
    E: Into<anyhow::Error>,
{
    result.map_err(|source| {
        anyhow::Error::new(RestrictedDiagnostic::new(
            code,
            safe_summary,
            operation,
            category,
            retryable,
            source,
        ))
    })
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum DiagnosticVisibility {
    Ordinary,
    Restricted,
}

#[derive(Debug, Serialize)]
pub struct DiagnosticEnvelope {
    version: u8,
    event: &'static str,
    diagnostic_id: String,
    visibility: DiagnosticVisibility,
    component: &'static str,
    operation: String,
    category: DiagnosticCategory,
    code: String,
    retryable: bool,
    summary: String,
    public: Option<PublicDiagnostic>,
    fields: DiagnosticFields,
    chain: Vec<DiagnosticCause>,
    frames: Vec<DiagnosticFrame>,
    fingerprint: String,
    truncated: Truncation,
}

#[derive(Debug, Serialize)]
struct PublicDiagnostic {
    code: &'static str,
    message: String,
    non_retryable: bool,
}

#[derive(Debug, Serialize)]
struct DiagnosticFields {
    attempt: u32,
    phase: String,
    exit_code: Option<i32>,
    signal: Option<i32>,
}

#[derive(Debug, Serialize)]
struct DiagnosticCause {
    kind: &'static str,
    summary: String,
}

#[derive(Debug, Serialize)]
struct DiagnosticFrame {
    module: String,
    file: String,
    line: u32,
}

#[derive(Debug, Default, Serialize)]
struct Truncation {
    chain: bool,
    frames: bool,
    summary: bool,
}

fn truncate_utf8(value: &str, max_bytes: usize) -> (String, bool) {
    // Keep the envelope safe for single-record logging. In particular, a
    // first-party error containing a newline must not be able to forge a
    // second log record after crossing into Python.
    let normalized = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    if normalized.len() <= max_bytes {
        return (normalized, false);
    }
    let mut end = max_bytes;
    while !normalized.is_char_boundary(end) {
        end -= 1;
    }
    (normalized[..end].to_owned(), true)
}

fn diagnostic_id() -> String {
    let sequence = NEXT_DIAGNOSTIC_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut hasher = Sha256::new();
    hasher.update(std::process::id().to_le_bytes());
    hasher.update(sequence.to_le_bytes());
    hasher.update(nanos.to_le_bytes());
    format!("dg-{}", &format!("{:x}", hasher.finalize())[..24])
}

fn fingerprint(
    code: &str,
    operation: &str,
    category: DiagnosticCategory,
    ordinary_summary: Option<&str>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"diskgraph-diagnostic-v1\0");
    hasher.update(code.as_bytes());
    hasher.update(b"\0");
    hasher.update(operation.as_bytes());
    hasher.update(b"\0");
    hasher.update(category.as_str().as_bytes());
    if let Some(summary) = ordinary_summary {
        hasher.update(b"\0ordinary\0");
        hasher.update(summary.as_bytes());
    }
    format!("dg-v1-{:x}", hasher.finalize())
}

/// Render a returned worker error without serializing a marked source chain.
pub fn render_diagnostic(
    error: &anyhow::Error,
    attempt: u32,
    phase: &str,
) -> DiagnosticEnvelope {
    let restricted = error
        .chain()
        .find_map(|source| source.downcast_ref::<RestrictedDiagnostic>());
    if let Some(restricted) = restricted {
        let (summary, summary_truncated) =
            truncate_utf8(restricted.safe_summary, MAX_SUMMARY_BYTES);
        let fingerprint = fingerprint(
            restricted.code,
            restricted.operation,
            restricted.category,
            None,
        );
        return DiagnosticEnvelope {
            version: DIAGNOSTIC_VERSION,
            event: "diskgraph.ingest_failed",
            diagnostic_id: diagnostic_id(),
            visibility: DiagnosticVisibility::Restricted,
            component: "diskgraph-ingest",
            operation: restricted.operation.to_owned(),
            category: restricted.category,
            code: restricted.code.to_owned(),
            retryable: restricted.retryable,
            summary: summary.clone(),
            public: None,
            fields: DiagnosticFields {
                attempt,
                phase: phase.to_owned(),
                exit_code: None,
                signal: None,
            },
            chain: vec![DiagnosticCause {
                kind: "restricted_boundary",
                summary,
            }],
            frames: Vec::new(),
            fingerprint,
            truncated: Truncation {
                summary: summary_truncated,
                ..Truncation::default()
            },
        };
    }

    let public = crate::public_ingest_error_message(error).map(|message| {
        let (message, _) = truncate_utf8(&message, MAX_SUMMARY_BYTES);
        PublicDiagnostic {
            code: crate::PUBLIC_INGEST_ERROR_CODE,
            message,
            non_retryable: true,
        }
    });
    let (operation, category, code, retryable) = if public.is_some() {
        (
            "validate_input",
            DiagnosticCategory::Validation,
            "diskgraph_ingest_validation",
            false,
        )
    } else if crate::is_disk_space_error(error) {
        (
            "write_snapshot",
            DiagnosticCategory::Resource,
            "ingest.no_space",
            true,
        )
    } else {
        (
            "ingest",
            DiagnosticCategory::Internal,
            "ingest.unexpected_failure",
            false,
        )
    };
    let root_summary = public
        .as_ref()
        .map(|public| public.message.clone())
        .unwrap_or_else(|| error.to_string());
    let (summary, summary_truncated) = truncate_utf8(&root_summary, MAX_SUMMARY_BYTES);
    let mut chain = Vec::new();
    let mut chain_truncated = false;
    for (index, source) in error.chain().enumerate() {
        if index == MAX_CHAIN_ENTRIES {
            chain_truncated = true;
            break;
        }
        let (cause_summary, cause_truncated) =
            truncate_utf8(&source.to_string(), MAX_CHAIN_SUMMARY_BYTES);
        chain_truncated |= cause_truncated;
        chain.push(DiagnosticCause {
            kind: "error",
            summary: cause_summary,
        });
    }
    let fingerprint = fingerprint(code, operation, category, Some(&summary));
    DiagnosticEnvelope {
        version: DIAGNOSTIC_VERSION,
        event: "diskgraph.ingest_failed",
        diagnostic_id: diagnostic_id(),
        visibility: DiagnosticVisibility::Ordinary,
        component: "diskgraph-ingest",
        operation: operation.to_owned(),
        category,
        code: code.to_owned(),
        retryable,
        summary,
        public,
        fields: DiagnosticFields {
            attempt,
            phase: phase.to_owned(),
            exit_code: None,
            signal: None,
        },
        chain,
        frames: Vec::new(),
        fingerprint,
        truncated: Truncation {
            chain: chain_truncated,
            frames: false,
            summary: summary_truncated,
        },
    }
}

/// Produce a message-free fallback for a worker that could not return an error.
pub fn render_process_fallback(
    attempt: u32,
    phase: &str,
    exit_code: Option<i32>,
    signal: Option<i32>,
    oom: bool,
    retryable: bool,
) -> DiagnosticEnvelope {
    let (code, summary) = if oom {
        (
            "ingest.worker_memory_pressure",
            "DiskGraph ingest worker was terminated by memory pressure",
        )
    } else {
        (
            "ingest.worker_abnormal_exit",
            "DiskGraph ingest worker exited before returning a diagnostic",
        )
    };
    DiagnosticEnvelope {
        version: DIAGNOSTIC_VERSION,
        event: "diskgraph.ingest_failed",
        diagnostic_id: diagnostic_id(),
        visibility: DiagnosticVisibility::Restricted,
        component: "diskgraph-ingest",
        operation: "run_worker".to_owned(),
        category: if oom {
            DiagnosticCategory::Resource
        } else {
            DiagnosticCategory::Process
        },
        code: code.to_owned(),
        retryable,
        summary: summary.to_owned(),
        public: None,
        fields: DiagnosticFields {
            attempt,
            phase: phase.to_owned(),
            exit_code,
            signal,
        },
        chain: vec![DiagnosticCause {
            kind: "process",
            summary: summary.to_owned(),
        }],
        frames: Vec::new(),
        fingerprint: fingerprint(
            code,
            "run_worker",
            if oom {
                DiagnosticCategory::Resource
            } else {
                DiagnosticCategory::Process
            },
            None,
        ),
        truncated: Truncation::default(),
    }
}

/// Serialize and atomically publish a bounded envelope.
pub fn write_diagnostic_atomically(
    path: &Path,
    diagnostic: &DiagnosticEnvelope,
) -> Result<()> {
    let payload = serde_json::to_vec(diagnostic)?;
    if payload.len() > DIAGNOSTIC_MAX_BYTES {
        return Err(anyhow!("diagnostic exceeded its encoded size limit"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("diagnostic path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temp_path = parent.join(format!(
        ".diagnostic.tmp-{}-{}",
        std::process::id(),
        NEXT_DIAGNOSTIC_ID.fetch_add(1, Ordering::Relaxed)
    ));
    if let Err(error) = fs::write(&temp_path, payload).and_then(|_| fs::rename(&temp_path, path)) {
        let _ = fs::remove_file(&temp_path);
        return Err(error.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope_json(envelope: &DiagnosticEnvelope) -> serde_json::Value {
        serde_json::to_value(envelope).unwrap()
    }

    #[test]
    fn ordinary_errors_preserve_actionable_chain_by_default() {
        let error = anyhow!("inner invariant canary").context("partition encoder failed");
        let value = envelope_json(&render_diagnostic(&error, 2, "encode"));
        assert_eq!(value["visibility"], "ordinary");
        assert_eq!(value["summary"], "partition encoder failed");
        assert!(value["chain"]
            .to_string()
            .contains("inner invariant canary"));
    }

    #[test]
    fn restricted_errors_never_render_source_or_outer_context() {
        let restricted = RestrictedDiagnostic::new(
            "object_store.read_failed",
            "DiskGraph could not read an input object",
            "read_source_object",
            DiagnosticCategory::Provider,
            true,
            anyhow!("Bearer prohibited-canary signed-url restricted-canary"),
        );
        let error =
            anyhow::Error::new(restricted).context("outer context with customer-path-canary");
        let value = envelope_json(&render_diagnostic(&error, 1, "encode"));
        let serialized = value.to_string();
        assert_eq!(value["visibility"], "restricted");
        assert!(serialized.contains("DiskGraph could not read an input object"));
        assert!(!serialized.contains("prohibited-canary"));
        assert!(!serialized.contains("restricted-canary"));
        assert!(!serialized.contains("customer-path-canary"));
    }

    #[test]
    fn rendered_envelopes_are_bounded() {
        let error = anyhow!("{}", "x".repeat(DIAGNOSTIC_MAX_BYTES * 2));
        let envelope = render_diagnostic(&error, 1, "worker");
        let payload = serde_json::to_vec(&envelope).unwrap();
        assert!(payload.len() <= DIAGNOSTIC_MAX_BYTES);
        assert!(envelope.truncated.summary);
    }

    #[test]
    fn explicit_public_error_uses_a_separate_approved_projection() {
        let error = anyhow::Error::new(crate::PublicIngestError::new(
            "safe validation detail".to_owned(),
        ));
        let value = envelope_json(&render_diagnostic(&error, 1, "encode"));
        assert_eq!(value["summary"], "safe validation detail");
        assert_eq!(value["public"]["code"], "diskgraph_ingest_validation");
        assert_eq!(value["public"]["message"], "safe validation detail");
        assert_eq!(value["public"]["non_retryable"], true);
    }

    #[test]
    fn disk_space_errors_have_a_typed_retryable_diagnostic() {
        let error = anyhow::Error::new(std::io::Error::from_raw_os_error(28));
        let value = envelope_json(&render_diagnostic(&error, 1, "persist"));
        assert_eq!(value["operation"], "write_snapshot");
        assert_eq!(value["category"], "resource");
        assert_eq!(value["code"], "ingest.no_space");
        assert_eq!(value["retryable"], true);
        assert!(value["public"].is_null());
    }
}
