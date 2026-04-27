//! Microbenchmark: isolate multicategorical token-explosion cost vs the
//! per-token observe path.
//!
//! Compares per-token CPU time across:
//!   C1: Categorical observe_string (one token per row, baseline)
//!   C2: Multicat observe_string_list on List<Utf8>, 5 tokens/row
//!   C3: Pure for_each_str iteration on same List<Utf8>, no observe
//!   C4: Multicat observe_int64_list on List<Int64>, 5 tokens/row
//!   C5: Multicat observe_string scalar with comma delim ("a,b,c,d,e"/row)
//!
//! Token count is identical across cases so ns/token comparisons are valid.
//!
//! Derived quantities:
//!   - C3 ≈ pure list explosion cost (no token work)
//!   - C2 - C3 ≈ multicat per-token observe cost (should match C1)
//!   - C5 - C3 ≈ ascii-byte split cost on scalar string
//!
//! Single-thread, in-memory data, deterministic RNG. Pin to one core for
//! stable numbers: `taskset -c 0 cargo run --release --bin multicat_microbench`.
//!
//! Usage: cargo run --release --bin multicat_microbench [-- --tokens 10000000]

use arrow_array::{Int64Array, StringArray};
use std::sync::Arc;
use std::time::Instant;

use diskgraph_ingest::col_stats::FileColStats;
use diskgraph_ingest::graph_config::{ColumnSpec, IngestedType, SemanticType};
use diskgraph_rowpack::{
    ColumnValueObserver, Int64ListValues, Int64ListView, StringListValues, StringListView,
};

const VOCAB_SIZE: usize = 50_000;
const ROW_FANOUT: usize = 5;
const REPS: usize = 3;

#[inline]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn make_col_spec(
    ingested: IngestedType,
    stype: Option<SemanticType>,
    delim: Option<&str>,
) -> ColumnSpec {
    ColumnSpec {
        name: "c".into(),
        source_dtype: None,
        ingested_dtype: ingested,
        fixed_len: None,
        timestamp_format: None,
        timestamp_unit: None,
        precision: None,
        scale: None,
        semantic_type: stype,
        multicategorical_delimiter: delim.map(String::from),
    }
}

fn gen_token_indices(n: usize) -> Vec<u64> {
    let mut state: u64 = 0x517cc1b727220a95;
    (0..n)
        .map(|_| xorshift(&mut state) % VOCAB_SIZE as u64)
        .collect()
}

fn run_case<F: FnMut() -> usize>(name: &str, expected_tokens: usize, mut body: F) {
    let mut best = u128::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        let observed = body();
        let elapsed = t.elapsed().as_nanos();
        assert_eq!(
            observed, expected_tokens,
            "case {name} reported wrong token count"
        );
        if elapsed < best {
            best = elapsed;
        }
    }
    let ns_per = best as f64 / expected_tokens as f64;
    println!(
        "{:48}  best={:>9.3}ms  {:>8.2} ns/token",
        name,
        best as f64 / 1.0e6,
        ns_per
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw: usize = args
        .windows(2)
        .find(|w| w[0] == "--tokens")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(10_000_000);
    let num_rows = raw / ROW_FANOUT;
    let total_tokens = num_rows * ROW_FANOUT;

    println!("multicat microbench");
    println!("  total tokens     : {}", total_tokens);
    println!("  rows (5 tok/row) : {}", num_rows);
    println!("  vocab size       : {}", VOCAB_SIZE);
    println!("  reps per case    : {}", REPS);
    println!();

    // Generate token universe once. Same indices reused across cases so vocab
    // hit pattern (HLL/HashMap behavior) stays comparable.
    let idxs = gen_token_indices(total_tokens);
    let token_strs: Vec<String> = idxs.iter().map(|&i| format!("tok_{:05}", i)).collect();
    let int_tokens: Vec<i64> = idxs.iter().map(|&i| i as i64).collect();

    let str_array: arrow_array::ArrayRef = Arc::new(StringArray::from(token_strs.clone()));
    let int_array: arrow_array::ArrayRef = Arc::new(Int64Array::from(int_tokens));

    // Pre-build comma-delimited scalar strings, one row per ROW_FANOUT tokens.
    let scalar_delim_strs: Vec<String> = (0..num_rows)
        .map(|r| token_strs[r * ROW_FANOUT..(r + 1) * ROW_FANOUT].join(","))
        .collect();
    let scalar_delim_arr: arrow_array::ArrayRef = Arc::new(StringArray::from(scalar_delim_strs));

    // ── C1 ────────────────────────────────────────────────────────────────
    {
        let spec = make_col_spec(
            IngestedType::String,
            Some(SemanticType::Categorical),
            None,
        );
        let mut stats = FileColStats::new(&[spec]);
        let arr = str_array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .clone();
        run_case("C1 Categorical observe_string", total_tokens, || {
            for i in 0..total_tokens {
                stats.observe_string(0, arr.value(i));
            }
            total_tokens
        });
    }

    // ── C2 ────────────────────────────────────────────────────────────────
    {
        let spec = make_col_spec(
            IngestedType::StringArray,
            Some(SemanticType::Multicategorical),
            None,
        );
        let mut stats = FileColStats::new(&[spec]);
        run_case("C2 Multicat observe_string_list", total_tokens, || {
            for r in 0..num_rows {
                let view = StringListView {
                    start: r * ROW_FANOUT,
                    end: r * ROW_FANOUT + ROW_FANOUT,
                    values: StringListValues::Utf8(str_array.clone()),
                };
                stats.observe_string_list(0, &view);
            }
            num_rows * ROW_FANOUT
        });
    }

    // ── C3 ────────────────────────────────────────────────────────────────
    run_case("C3 for_each_str iter only (no observe)", total_tokens, || {
        let mut consumed: u64 = 0;
        for r in 0..num_rows {
            let view = StringListView {
                start: r * ROW_FANOUT,
                end: r * ROW_FANOUT + ROW_FANOUT,
                values: StringListValues::Utf8(str_array.clone()),
            };
            view.for_each_str(|s| {
                // Touch one byte so iteration cannot be DCE'd.
                consumed = consumed.wrapping_add(s.as_bytes()[0] as u64);
            });
        }
        std::hint::black_box(consumed);
        num_rows * ROW_FANOUT
    });

    // ── C4 ────────────────────────────────────────────────────────────────
    {
        let spec = make_col_spec(
            IngestedType::Int64Array,
            Some(SemanticType::Multicategorical),
            None,
        );
        let mut stats = FileColStats::new(&[spec]);
        run_case("C4 Multicat observe_int64_list", total_tokens, || {
            for r in 0..num_rows {
                let view = Int64ListView {
                    start: r * ROW_FANOUT,
                    end: r * ROW_FANOUT + ROW_FANOUT,
                    values: Int64ListValues::Native(int_array.clone()),
                };
                stats.observe_int64_list(0, &view);
            }
            num_rows * ROW_FANOUT
        });
    }

    // ── C5 ────────────────────────────────────────────────────────────────
    {
        let spec = make_col_spec(
            IngestedType::String,
            Some(SemanticType::Multicategorical),
            Some(","),
        );
        let mut stats = FileColStats::new(&[spec]);
        let arr = scalar_delim_arr
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .clone();
        run_case(
            "C5 Multicat scalar+ascii-delim observe_string",
            total_tokens,
            || {
                for r in 0..num_rows {
                    stats.observe_string(0, arr.value(r));
                }
                num_rows * ROW_FANOUT
            },
        );
    }

    println!();
    println!("Interpretation:");
    println!("  C1                 = baseline observe-string cost");
    println!("  C3                 = pure list-explosion cost (no token work)");
    println!("  C2 - C3            = multicat per-token observe cost (≈ should match C1)");
    println!("  C5 - C3            = ascii-byte delim split cost on scalar string");
    println!("  C4                 = multicat int64 path (no string allocations)");
}
