use crate::{ProgressCallback, ProgressEvent, ProgressKind, ProgressPhase};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::io::IsTerminal;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgressKey {
    phase: ProgressPhase,
    table: String,
    shard: Option<usize>,
}

struct ProgressState {
    mp: MultiProgress,
    bars: HashMap<ProgressKey, ProgressBar>,
}

pub fn build_progress() -> Option<ProgressCallback> {
    if !std::io::stderr().is_terminal() {
        return None;
    }
    let state = Arc::new(Mutex::new(ProgressState {
        mp: MultiProgress::new(),
        bars: HashMap::new(),
    }));
    let cb = Arc::new(move |event: ProgressEvent| {
        let mut guard = state.lock().unwrap();
        let label = progress_label(&event);
        let key = ProgressKey {
            phase: event.phase,
            table: event.table.clone(),
            shard: event.shard,
        };
        match event.kind {
            ProgressKind::Start => {
                if guard.bars.contains_key(&key) {
                    return;
                }
                let bar = match event.total {
                    Some(total) => {
                        let bar = guard.mp.add(ProgressBar::new(total));
                        bar.set_style(bar_style());
                        bar
                    }
                    None => {
                        let bar = guard.mp.add(ProgressBar::new_spinner());
                        bar.set_style(spinner_style());
                        bar
                    }
                };
                bar.set_message(label);
                if let Some(total) = event.total {
                    bar.set_prefix(format!("({}/{})", format_count(0), format_count(total)));
                }
                guard.bars.insert(key, bar);
            }
            ProgressKind::Advance => {
                if let Some(bar) = guard.bars.get(&key) {
                    bar.inc(event.delta);
                    if let Some(total) = bar.length() {
                        bar.set_prefix(format!(
                            "({}/{})",
                            format_count(bar.position()),
                            format_count(total)
                        ));
                    } else {
                        if event.status.is_some() {
                            bar.set_message(label);
                        } else {
                            let count = format_count(bar.position());
                            bar.set_message(format!("{} {}", label, count));
                        }
                    }
                }
            }
            ProgressKind::Finish => {
                if let Some(bar) = guard.bars.remove(&key) {
                    if let Some(total) = event.total.or_else(|| bar.length()) {
                        bar.set_length(total);
                        if bar.position() < total {
                            bar.set_position(total);
                        }
                        bar.set_prefix(format!(
                            "({}/{})",
                            format_count(bar.position()),
                            format_count(total)
                        ));
                    }
                    bar.finish_with_message(label);
                }
            }
        }
    });
    Some(cb)
}

fn progress_label(event: &ProgressEvent) -> String {
    let phase = match event.phase {
        ProgressPhase::Context => "context",
        ProgressPhase::Encode => "encode",
        ProgressPhase::PkMap => "pk_map",
        ProgressPhase::EdgesPassA => "edges_map",
        ProgressPhase::EdgesPassB => "edges_b",
        ProgressPhase::EdgesPassC => "edges_c",
        ProgressPhase::Index => "index",
    };
    if matches!(event.phase, ProgressPhase::Index) {
        let label = format!("{} {}", phase, event.table);
        return match &event.status {
            Some(status) => format!("{} {}", label, status),
            None => label,
        };
    }
    match event.shard {
        Some(shard) => {
            let label = format!("{} {} shard-{:05}", phase, event.table, shard);
            match &event.status {
                Some(status) => format!("{} {}", label, status),
                None => label,
            }
        }
        None => {
            let label = format!("{} {}", phase, event.table);
            match &event.status {
                Some(status) => format!("{} {}", label, status),
                None => label,
            }
        }
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix}")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

pub fn format_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut value = bytes as f64;
    let mut unit = units[0];
    for next in &units[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    if unit == "B" {
        format!("{} {}", bytes, unit)
    } else {
        format!("{:.1} {}", value, unit)
    }
}

pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        return format!("{}ms", ms);
    }
    let secs = ms / 1000;
    let rem_ms = ms % 1000;
    if secs < 60 {
        return format!("{}.{}s", secs, rem_ms / 100);
    }
    let mins = secs / 60;
    let rem_secs = secs % 60;
    format!("{}m {}s", mins, rem_secs)
}

pub fn format_count(value: u64) -> String {
    if value >= 1_000_000_000 {
        format!("{:.1}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}
