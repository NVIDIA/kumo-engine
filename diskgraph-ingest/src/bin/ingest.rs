use std::path::PathBuf;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::{
    ingest_with_stats, Dest, IngestOptions, ProgressCallback, ProgressEvent, ProgressKind,
    ProgressPhase,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs;
use std::io::IsTerminal;
use std::sync::{Arc, Mutex};

#[cfg(all(feature = "mem-profile", target_os = "linux"))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about = "Ingest raw CSV/Parquet into diskgraph index format.")]
struct Args {
    /// Path to source_metadata.json
    #[arg(long)]
    source: PathBuf,
    /// Output destination root (path or URL).
    #[arg(long)]
    output: String,
    /// Destination kind.
    #[arg(long, value_enum, default_value = "local")]
    dest: DestArg,
    /// Write pprof profiles per phase to this directory.
    #[arg(long, value_name = "DIR", num_args = 0..=1, default_missing_value = ".")]
    profile: Option<PathBuf>,
    /// Enable the shared worker pool.
    #[arg(long, default_value_t = true)]
    parallel: bool,
    /// Enable jemalloc heap dump watchdog (80% RSS anon threshold).
    #[arg(short = 'm', long = "mem-profile")]
    mem_profile: bool,
    /// Max bytes of index-build edge sort memory allowed across workers.
    #[arg(long, value_name = "BYTES")]
    index_mem_budget_bytes: Option<u64>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DestArg {
    Local,
    S3,
    Gcs,
    Adls,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info,pprof=warn")
        .init();
    let args = Args::parse();
    let source_path = if args.source.is_dir() {
        args.source.join("source_metadata.json")
    } else {
        args.source.clone()
    };
    if !source_path.is_file() {
        anyhow::bail!(
            "source metadata not found at {}",
            source_path.display()
        );
    }
    if args.profile.is_some() && cfg!(debug_assertions) && cfg!(target_arch = "aarch64") {
        anyhow::bail!(
            "pprof profiling in debug builds can panic on aarch64; rerun with `--release`"
        );
    }
    if let Some(profile_dir) = args.profile.as_ref() {
        fs::create_dir_all(profile_dir)?;
    }
    let mem_watchdog: Option<mem_watchdog::MemWatchdog> = if args.mem_profile {
        #[cfg(feature = "mem-profile")]
        {
            let dir = args
                .profile
                .clone()
                .unwrap_or_else(|| PathBuf::from("."));
            Some(mem_watchdog::start(dir))
        }
        #[cfg(not(feature = "mem-profile"))]
        {
            tracing::warn!("--mem-profile requires the cargo feature 'mem-profile'");
            None
        }
    } else {
        None
    };
    let options = IngestOptions {
        profile_dir: args.profile.clone(),
        progress: build_progress(),
        parallel: args.parallel,
        index_mem_budget_bytes: args.index_mem_budget_bytes,
    };
    let dest = match args.dest {
        DestArg::Local => Dest::Local,
        DestArg::S3 => Dest::S3,
        DestArg::Gcs => Dest::Gcs,
        DestArg::Adls => Dest::Adls,
    };
    let (_metadata, stats) = ingest_with_stats(&source_path, &args.output, dest, options)?;
    drop(mem_watchdog);
    println!("Output: {}", stats.output);
    println!(
        "Node tables: {} ({} nodes)",
        stats.node_tables, stats.total_nodes
    );
    println!(
        "Edge relations: {} ({} edges)",
        stats.edge_relations, stats.total_edges
    );
    if !stats.node_counts.is_empty() {
        println!("Nodes per table:");
        let mut entries: Vec<_> = stats.node_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
        }
    }
    if !stats.edge_counts.is_empty() {
        println!("Edges per relation:");
        let mut entries: Vec<_> = stats.edge_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
        }
    }
    if stats.pk_duplicate_total > 0 {
        println!("Duplicate primary keys:");
        println!("  Total skipped: {}", stats.pk_duplicate_total);
        let mut entries: Vec<_> = stats.pk_duplicate_counts.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, count) in entries {
            println!("  {}: {}", name, count);
            if let Some(samples) = stats.pk_duplicate_samples.get(name) {
                if !samples.is_empty() {
                    println!("    Samples: {}", samples.join(", "));
                }
            }
        }
    }
    if stats.key_parse_failures > 0 || stats.key_lookup_failures > 0 {
        println!("Key matching failures:");
        println!("  Parse failures: {}", stats.key_parse_failures);
        println!("  Lookup misses: {}", stats.key_lookup_failures);
        if !stats.key_failures_by_context.is_empty() {
            println!("  By context:");
            let mut entries: Vec<_> = stats.key_failures_by_context.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (name, counts) in entries {
                if counts.parse_failures == 0 && counts.lookup_failures == 0 {
                    continue;
                }
                println!(
                    "    {}: parse {}, lookup {}",
                    name, counts.parse_failures, counts.lookup_failures
                );
            }
        }
        if !stats.key_failure_samples.is_empty() {
            println!("  Examples:");
            for sample in &stats.key_failure_samples {
                println!("    {}", sample);
            }
        }
    }
    if let Some(trimmed) = stats.trimmed_strings {
        println!("Trimmed strings (>32KB): {}", trimmed);
    }
    let total_bytes = stats.perf.bytes_read.saturating_add(stats.perf.bytes_written);
    let wall_secs = (stats.perf.wall_time_ms as f64) / 1000.0;
    let bytes_per_sec = if wall_secs > 0.0 {
        (total_bytes as f64) / wall_secs
    } else {
        0.0
    };
    println!("Performance:");
    println!("  Wall time: {}", format_duration(stats.perf.wall_time_ms));
    println!(
        "  Bytes read: {}",
        format_bytes(stats.perf.bytes_read)
    );
    println!(
        "  Bytes written: {}",
        format_bytes(stats.perf.bytes_written)
    );
    println!("  Total bytes: {}", format_bytes(total_bytes));
    println!("  Throughput: {}/s", format_bytes(bytes_per_sec as u64));
    if stats.perf.temp_bytes_total > 0 {
        println!("  Temp bytes (edges): {}", format_bytes(stats.perf.temp_bytes_total));
        if stats.perf.temp_fkeys_bytes > 0 {
            println!(
                "    fkeys: {}",
                format_bytes(stats.perf.temp_fkeys_bytes)
            );
        }
        if stats.perf.temp_edges_src_bytes > 0 {
            println!(
                "    edges_src: {}",
                format_bytes(stats.perf.temp_edges_src_bytes)
            );
        }
        if stats.perf.temp_edges_map_bytes > 0 {
            println!(
                "    edges_map: {}",
                format_bytes(stats.perf.temp_edges_map_bytes)
            );
        }
    }
    if stats.perf.temp_node_data_bytes > 0 {
        println!(
            "  Temp bytes (nodes): {}",
            format_bytes(stats.perf.temp_node_data_bytes)
        );
    }
    println!("  Stage timings (wall):");
    println!(
        "    context: {}",
        format_duration(stats.perf.context_wall_ms)
    );
    println!(
        "    encode: {}",
        format_duration(stats.perf.encode_wall_ms)
    );
    println!(
        "    pk_map: {}",
        format_duration(stats.perf.pk_map_wall_ms)
    );
    println!(
        "    edges_map: {}",
        format_duration(stats.perf.edges_map_wall_ms)
    );
    println!(
        "    index: {}",
        format_duration(stats.perf.index_wall_ms)
    );
    println!(
        "    finalize: {}",
        format_duration(stats.perf.finalize_wall_ms)
    );
    if let Some(profile_dir) = args.profile.as_ref() {
        let mut profiles = Vec::new();
        for name in ["context", "encode", "pk_map", "edges_map", "index"] {
            let svg = profile_dir.join(format!("{name}-profile.svg"));
            let pb = profile_dir.join(format!("{name}-profile.pb"));
            if svg.exists() {
                profiles.push(svg);
            }
            if pb.exists() {
                profiles.push(pb);
            }
        }
        if !profiles.is_empty() {
            println!("Profiles:");
            for path in &profiles {
                println!("  {}", path.display());
            }
            let mut pb_paths = Vec::new();
            for name in ["context", "encode", "pk_map", "edges_map", "index"] {
                let pb = profile_dir.join(format!("{name}-profile.pb"));
                if pb.exists() {
                    pb_paths.push(pb);
                }
            }
            if !pb_paths.is_empty() {
                println!("Go pprof:");
                for path in pb_paths {
                    println!("  go tool pprof {}", path.display());
                }
            }
        }
    }
    println!(
        "Next: {}",
        explore_command_hint(&args.dest, &stats.output)
    );
    Ok(())
}

#[cfg(feature = "mem-profile")]
mod mem_watchdog {
    use std::ffi::CString;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    #[cfg(target_os = "linux")]
    use std::os::unix::ffi::OsStrExt;

    const MEM_WATCHDOG_INTERVAL: Duration = Duration::from_millis(500);
    const MEM_WATCHDOG_THRESHOLD: f64 = 0.80;

    pub struct MemWatchdog {
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Drop for MemWatchdog {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    pub fn start(dump_dir: PathBuf) -> MemWatchdog {
        log_jemalloc_conf();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let handle = thread::spawn(move || mem_watchdog_loop(dump_dir, stop_thread));
        MemWatchdog {
            stop,
            handle: Some(handle),
        }
    }

    fn mem_watchdog_loop(dump_dir: PathBuf, stop: Arc<AtomicBool>) {
        if !cfg!(target_os = "linux") {
            tracing::warn!("memory watchdog only supported on linux");
            return;
        }
        let total_bytes = match total_memory_bytes() {
            Some(total) => total,
            None => {
                tracing::warn!("memory watchdog disabled: unable to read total memory");
                return;
            }
        };
        let threshold_bytes = (total_bytes as f64 * MEM_WATCHDOG_THRESHOLD) as u64;
        let dumped = AtomicBool::new(false);
        while !stop.load(Ordering::Relaxed) {
            if let Some(rss_anon) = rss_anon_bytes() {
                if rss_anon >= threshold_bytes
                    && dumped
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                {
                    if let Err(err) = dump_heap_profile(&dump_dir) {
                        tracing::warn!("heap profile dump failed: {err}");
                    }
                }
            }
            thread::sleep(MEM_WATCHDOG_INTERVAL);
        }
    }

    fn dump_heap_profile(dump_dir: &PathBuf) -> Result<()> {
        if !jemalloc_prof_supported() {
            tracing::warn!("jemalloc profiling is not compiled in");
            return Ok(());
        }
        if !jemalloc_prof_enabled() {
            tracing::warn!("jemalloc opt.prof=false; set _RJEM_MALLOC_CONF=prof:true");
            return Ok(());
        }
        if let Err(err) = unsafe { jemalloc_ctl::raw::write(b"prof.active\0", true) } {
            tracing::warn!("jemalloc prof.active enable failed: {err}");
        }
        fs::create_dir_all(dump_dir)?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let dump_path = dump_dir.join(format!("heap-{}.heap", ts));
        let c_path = CString::new(dump_path.as_os_str().as_bytes())?;
        unsafe {
            jemalloc_ctl::raw::write(b"prof.dump\0", c_path.as_ptr())
                .map_err(|e| anyhow::anyhow!("prof.dump failed: {e}"))?;
        }
        tracing::info!("heap profile written to {}", dump_path.display());
        Ok(())
    }

    fn jemalloc_prof_supported() -> bool {
        unsafe { jemalloc_ctl::raw::read::<bool>(b"config.prof\0") }.unwrap_or(false)
    }

    fn jemalloc_prof_enabled() -> bool {
        unsafe { jemalloc_ctl::raw::read::<bool>(b"opt.prof\0") }.unwrap_or(false)
    }

    fn log_jemalloc_conf() {
        let prof = unsafe { jemalloc_ctl::raw::read::<bool>(b"config.prof\0") }.unwrap_or(false);
        let opt_prof = unsafe { jemalloc_ctl::raw::read::<bool>(b"opt.prof\0") }.unwrap_or(false);
        tracing::warn!("jemalloc config.prof={}, opt.prof={}", prof, opt_prof);
        let conf = unsafe { jemalloc_ctl::raw::read::<*const std::os::raw::c_char>(b"config.malloc_conf\0") };
        if let Ok(ptr) = conf {
            if ptr.is_null() {
                tracing::warn!("jemalloc config.malloc_conf is null");
                return;
            }
            unsafe {
                let cstr = std::ffi::CStr::from_ptr(ptr);
                tracing::warn!("jemalloc config.malloc_conf: {}", cstr.to_string_lossy());
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn rss_anon_bytes() -> Option<u64> {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("RssAnon:") {
                return parse_kb_line(value);
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    fn rss_anon_bytes() -> Option<u64> {
        None
    }

    #[cfg(target_os = "linux")]
    fn total_memory_bytes() -> Option<u64> {
        let mem_total = read_meminfo_total().unwrap_or(0);
        let cgroup_limit = read_cgroup_limit().unwrap_or(u64::MAX);
        let total = mem_total.min(cgroup_limit);
        if total == 0 || total == u64::MAX {
            None
        } else {
            Some(total)
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn total_memory_bytes() -> Option<u64> {
        None
    }

    #[cfg(target_os = "linux")]
    fn read_meminfo_total() -> Option<u64> {
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(value) = line.strip_prefix("MemTotal:") {
                return parse_kb_line(value);
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    fn read_cgroup_limit() -> Option<u64> {
        if let Ok(limit) = fs::read_to_string("/sys/fs/cgroup/memory.max") {
            let trimmed = limit.trim();
            if trimmed != "max" {
                if let Ok(bytes) = trimmed.parse::<u64>() {
                    return Some(bytes);
                }
            }
        }
        if let Ok(limit) = fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
            let trimmed = limit.trim();
            if let Ok(bytes) = trimmed.parse::<u64>() {
                return Some(bytes);
            }
        }
        None
    }

    #[cfg(target_os = "linux")]
    fn parse_kb_line(value: &str) -> Option<u64> {
        let mut iter = value.split_whitespace();
        let num = iter.next()?.parse::<u64>().ok()?;
        Some(num.saturating_mul(1024))
    }
}

#[cfg(not(feature = "mem-profile"))]
mod mem_watchdog {
    pub struct MemWatchdog;
}

fn explore_command_hint(dest: &DestArg, output: &str) -> String {
    let _ = dest;
    format!("./diskgraph-index/scripts/explore_local.sh {}", output)
}

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

fn build_progress() -> Option<ProgressCallback> {
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
                        let count = format_count(bar.position());
                        bar.set_message(format!("{} {}", label, count));
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
        return format!("{} {}", phase, event.table);
    }
    match event.shard {
        Some(shard) => format!("{} {} shard-{:05}", phase, event.table, shard),
        None => format!("{} {}", phase, event.table),
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

fn format_bytes(bytes: u64) -> String {
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

fn format_duration(ms: u64) -> String {
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

fn format_count(value: u64) -> String {
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
