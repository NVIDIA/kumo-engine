use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, TableState, Wrap, List, ListItem, ListState};
use ratatui::Terminal;
use serde::Deserialize;
use tokio::runtime::Runtime;
use futures::{pin_mut, StreamExt};
use futures::stream;
 
use object_store::{GetResult, ObjectStoreExt};

use diskgraph_ingest::io::{build_store_for_cloud_path, parse_az_url, parse_gs_url, parse_s3_url};

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactive ingest performance explorer.")]
struct Args {
    /// Stats destination root (s3://, gs://, az://).
    #[arg(long, value_name = "URL")]
    stats_dest: Option<String>,
    /// Dataset name (optional; defaults to first available).
    #[arg(long, value_name = "NAME")]
    dataset: Option<String>,
    /// Max runs to load per dataset.
    #[arg(long, default_value_t = 200)]
    max_runs: usize,
    #[arg(long, default_value_t = 30)]
    top_profiles: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
struct RunSummary {
    run_id: String,
    dataset: String,
    source_path: String,
    output_path: String,
    finished_at: String,
    user: String,
    git_sha: Option<String>,
    git_branch: Option<String>,
    git_dirty: Option<bool>,
    cloud: Option<String>,
    instance_type: Option<String>,
    cpu_logical: Option<u64>,
    cpu_physical: Option<u64>,
    memory_bytes_total: Option<u64>,
    disk_bytes_total: Option<u64>,
    disk_bytes_available: Option<u64>,
    wall_time_ms: u64,
    bytes_read: u64,
    bytes_written: u64,
    total_bytes: u64,
    throughput_bytes_per_sec: u64,
    stage_timings_ms: StageTimings,
    normalized: NormalizedMetrics,
    peak_rss_anon_bytes: Option<u64>,
    peak_context_rss_anon_bytes: Option<u64>,
    peak_encode_rss_anon_bytes: Option<u64>,
    peak_pk_map_rss_anon_bytes: Option<u64>,
    peak_edges_map_rss_anon_bytes: Option<u64>,
    peak_index_rss_anon_bytes: Option<u64>,
    peak_finalize_rss_anon_bytes: Option<u64>,
    num_shards: Option<usize>,
    profiles_present: bool,
    build_stats_present: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct StageTimings {
    context: u64,
    encode: u64,
    pk_map: u64,
    edges_map: u64,
    index: u64,
    finalize: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[allow(dead_code)]
struct NormalizedMetrics {
    throughput_bytes_per_sec_per_cpu: Option<f64>,
    wall_time_ms_per_cpu: Option<f64>,
}

#[derive(Clone, Debug)]
struct InstanceEntry {
    name: String,
    latest: Option<RunSummary>,
}

#[derive(Clone, Debug)]
struct RegressionInfo;

#[derive(Clone, Debug)]
struct RunEntry {
    summary: RunSummary,
    regression: RegressionInfo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Normal,
    Help,
    ProfileView,
}

struct AppState {
    datasets: Vec<String>,
    dataset_state: ListState,
    instances: Vec<InstanceEntry>,
    instance_state: ListState,
    runs: Vec<RunEntry>,
    run_state: TableState,
    mode: Mode,
    status: String,
    baseline: Option<RunSummary>,
    profile_tabs: Vec<ProfileTab>,
    profile_tab_index: usize,
    loaded_dataset_index: Option<usize>,
    pending_dataset_index: Option<usize>,
    pending_dataset_at: Option<Instant>,
    loaded_instance_index: Option<usize>,
    pending_instance_index: Option<usize>,
    pending_instance_at: Option<Instant>,
    last_refresh_at: Instant,
    dataset_rect: Rect,
    instance_rect: Rect,
    runs_rect: Rect,
}

struct ProfileTab {
    name: String,
    lines: Vec<String>,
}

struct StatsStore {
    store: Arc<dyn object_store::ObjectStore>,
    prefix: String,
    rt: Runtime,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let stats_root = resolve_stats_root(&args)?;
    let store = StatsStore::open(&stats_root)?;

    let datasets = store.list_datasets()?;
    let selected_dataset = select_initial_dataset(&datasets, args.dataset.as_deref());
    let instances = match selected_dataset.as_deref() {
        Some(dataset) => store.list_instances(dataset)?,
        None => Vec::new(),
    };
    let mut app = AppState {
        datasets,
        dataset_state: ListState::default(),
        instances,
        instance_state: ListState::default(),
        runs: Vec::new(),
        run_state: TableState::default(),
        mode: Mode::Normal,
        status: String::new(),
        baseline: None,
        profile_tabs: Vec::new(),
        profile_tab_index: 0,
        loaded_dataset_index: None,
        pending_dataset_index: None,
        pending_dataset_at: None,
        loaded_instance_index: None,
        pending_instance_index: None,
        pending_instance_at: None,
        last_refresh_at: Instant::now(),
        dataset_rect: Rect::default(),
        instance_rect: Rect::default(),
        runs_rect: Rect::default(),
    };
    if let Some(dataset) = selected_dataset.as_deref() {
        if let Some(idx) = app.datasets.iter().position(|d| d == dataset) {
            app.dataset_state.select(Some(idx));
        }
    } else {
        app.dataset_state.select(None);
    }
    if !app.instances.is_empty() {
        app.instance_state.select(Some(0));
    }
    load_runs_for_selected(&store, &mut app, &args)?;
    app.loaded_dataset_index = app.dataset_state.selected();
    app.loaded_instance_index = app.instance_state.selected();

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    stdout.execute(EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut running = true;
    while running {
        terminal.draw(|frame| draw(frame, &mut app))?;

        if event::poll(Duration::from_millis(200))? {
            match event::read()? {
                Event::Key(key) => {
                    running = handle_key(key, &store, &args, &mut app)?;
                }
                Event::Mouse(mouse) => {
                    handle_mouse(mouse, &store, &args, &mut app)?;
                }
                _ => {}
            }
        }
        apply_debounced_dataset_change(&store, &mut app, &args)?;
        apply_debounced_instance_change(&store, &mut app, &args)?;
        apply_auto_refresh(&store, &mut app, &args)?;
    }

    disable_raw_mode()?;
    terminal.backend_mut().execute(DisableMouseCapture)?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn resolve_stats_root(args: &Args) -> Result<String> {
    if let Some(dest) = args.stats_dest.as_ref() {
        return Ok(dest.clone());
    }
    if let Ok(dest) = std::env::var("DISKGRAPH_STATS_DEST") {
        if !dest.trim().is_empty() {
            return Ok(dest);
        }
    }
    if let Some(root) = stats_root_from_imds() {
        return Ok(root);
    }
    if std::env::var("AWS_REGION").is_ok() || std::env::var("AWS_DEFAULT_REGION").is_ok() {
        return Ok("s3://kumo-unit-test/diskgraph-profiles".to_string());
    }
    if std::env::var("GOOGLE_CLOUD_PROJECT").is_ok()
        || std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok()
    {
        return Ok("gs://kumo-unit-test/diskgraph-profiles".to_string());
    }
    if std::env::var("AZURE_TENANT_ID").is_ok() || std::env::var("AZURE_STORAGE_ACCOUNT").is_ok()
    {
        let account = std::env::var("AZURE_STORAGE_ACCOUNT")
            .or_else(|_| std::env::var("AZURE_ACCOUNT"))
            .unwrap_or_else(|_| "account".to_string());
        let container = std::env::var("AZURE_STORAGE_CONTAINER")
            .or_else(|_| std::env::var("AZURE_CONTAINER"))
            .unwrap_or_else(|_| "container".to_string());
        return Ok(format!("az://{account}/{container}/diskgraph-profiles"));
    }
    Err(anyhow!("stats root not specified; set DISKGRAPH_STATS_DEST"))
}

fn stats_root_from_imds() -> Option<String> {
    if fetch_aws_instance_type().is_some() {
        return Some("s3://kumo-unit-test/diskgraph-profiles".to_string());
    }
    if fetch_gcp_instance_type().is_some() {
        return Some("gs://kumo-unit-test/diskgraph-profiles".to_string());
    }
    if fetch_azure_instance_type().is_some() {
        let account = std::env::var("AZURE_STORAGE_ACCOUNT")
            .or_else(|_| std::env::var("AZURE_ACCOUNT"))
            .unwrap_or_else(|_| "account".to_string());
        let container = std::env::var("AZURE_STORAGE_CONTAINER")
            .or_else(|_| std::env::var("AZURE_CONTAINER"))
            .unwrap_or_else(|_| "container".to_string());
        return Some(format!("az://{account}/{container}/diskgraph-profiles"));
    }
    None
}

fn fetch_aws_instance_type() -> Option<String> {
    let token = ureq::put("http://169.254.169.254/latest/api/token")
        .timeout(Duration::from_millis(800))
        .set("X-aws-ec2-metadata-token-ttl-seconds", "300")
        .call()
        .ok()?
        .into_string()
        .ok()?;
    let resp = ureq::get("http://169.254.169.254/latest/meta-data/instance-type")
        .timeout(Duration::from_millis(800))
        .set("X-aws-ec2-metadata-token", &token)
        .call()
        .ok()?;
    resp.into_string().ok().map(|s| s.trim().to_string())
}

fn fetch_gcp_instance_type() -> Option<String> {
    let resp = ureq::get("http://metadata.google.internal/computeMetadata/v1/instance/machine-type")
        .timeout(Duration::from_millis(800))
        .set("Metadata-Flavor", "Google")
        .call()
        .ok()?;
    let raw = resp.into_string().ok()?;
    Some(
        raw.trim()
            .rsplit('/')
            .next()
            .unwrap_or(raw.trim())
            .to_string(),
    )
}

fn fetch_azure_instance_type() -> Option<String> {
    let resp = ureq::get("http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01")
        .timeout(Duration::from_millis(800))
        .set("Metadata", "true")
        .call()
        .ok()?;
    resp.into_string().ok().map(|s| s.trim().to_string())
}

impl StatsStore {
    fn open(root: &str) -> Result<Self> {
        if !(root.starts_with("s3://") || root.starts_with("gs://") || root.starts_with("az://")) {
            return Err(anyhow!("stats root must be s3://, gs://, or az://"));
        }
        let store = build_store_for_cloud_path(root)?;
        let prefix = cloud_prefix(root)?;
        Ok(StatsStore {
            store,
            prefix,
            rt: Runtime::new()?,
        })
    }

    fn list_instances(&self, dataset: &str) -> Result<Vec<InstanceEntry>> {
        let prefix = object_path(&join_prefix(&self.prefix, dataset));
        let result = self
            .rt
            .block_on(self.store.list_with_delimiter(Some(&prefix)))
            .context("list instances")?;
        let mut instances = Vec::new();
        for common in result.common_prefixes {
            if let Some(name) = common.filename() {
                let name = name.to_string();
                instances.push(InstanceEntry { name, latest: None });
            }
        }
        instances.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(instances)
    }

    fn list_datasets(&self) -> Result<Vec<String>> {
        let prefix = object_path(&self.prefix);
        let result = self
            .rt
            .block_on(self.store.list_with_delimiter(Some(&prefix)))
            .context("list datasets")?;
        let mut datasets = Vec::new();
        for common in result.common_prefixes {
            if let Some(name) = common.filename() {
                datasets.push(name.to_string());
            }
        }
        datasets.sort();
        Ok(datasets)
    }

    fn list_runs_for_instance(
        &self,
        dataset: &str,
        instance_dir: &str,
        max_runs: usize,
    ) -> Result<Vec<RunSummary>> {
        let runs_prefix = join_prefix(&self.prefix, &format!("{dataset}/{instance_dir}"));
        let run_prefix = object_path(&runs_prefix);
        let mut runs = self.rt.block_on(async {
            let run_result = self
                .store
                .list_with_delimiter(Some(&run_prefix))
                .await
                .context("list runs")?;
            let run_ids: Vec<String> = run_result
                .common_prefixes
                .into_iter()
                .filter_map(|run_path| run_path.filename().map(|name| name.to_string()))
                .collect();
            let mut summaries = Vec::new();
            let stream = stream::iter(run_ids.into_iter().map(|run_id| async move {
                self.fetch_run_summary_async(dataset, instance_dir, &run_id)
                    .await
            }))
            .buffer_unordered(16);
            pin_mut!(stream);
            while let Some(result) = stream.next().await {
                if let Ok(summary) = result {
                    summaries.push(summary);
                }
            }
            Ok::<_, anyhow::Error>(summaries)
        })?;
        runs.sort_by(|a, b| b.finished_at.cmp(&a.finished_at));
        if runs.len() > max_runs {
            runs.truncate(max_runs);
        }
        Ok(runs)
    }

    async fn fetch_run_summary_async(
        &self,
        dataset: &str,
        instance_dir: &str,
        run_id: &str,
    ) -> Result<RunSummary> {
        let path = join_prefix(
            &self.prefix,
            &format!("{dataset}/{instance_dir}/{run_id}/run_summary.json"),
        );
        let obj = object_path(&path);
        let get: GetResult = self.store.get(&obj).await?;
        let data = get.bytes().await?;
        let bytes = data.to_vec();
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn list_profiles(
        &self,
        dataset: &str,
        run_id: &str,
        instance_type: Option<&str>,
        source_path: &str,
    ) -> Result<Vec<String>> {
        let instance_dir = instance_dir_name(instance_type, source_path);
        let profiles_prefix = join_prefix(
            &self.prefix,
            &format!("{dataset}/{instance_dir}/{run_id}/profiles"),
        );
        let prefix = object_path(&profiles_prefix);
        let mut names = Vec::new();
        let stream = self.store.list(Some(&prefix));
        let results = self.rt.block_on(async {
            pin_mut!(stream);
            let mut collected = Vec::new();
            while let Some(item) = stream.next().await {
                collected.push(item?);
            }
            Ok::<_, object_store::Error>(collected)
        })?;
        for meta in results {
            if let Some(name) = meta.location.filename() {
                names.push(name.to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    fn download_profile(
        &self,
        dataset: &str,
        run_id: &str,
        instance_type: Option<&str>,
        source_path: &str,
        name: &str,
        dest: &Path,
    ) -> Result<()> {
        let instance_dir = instance_dir_name(instance_type, source_path);
        let path = join_prefix(
            &self.prefix,
            &format!("{dataset}/{instance_dir}/{run_id}/profiles/{name}"),
        );
        let obj = object_path(&path);
        let bytes = self.rt.block_on(async {
            let get: GetResult = self.store.get(&obj).await?;
            let data = get.bytes().await?;
            Ok::<_, object_store::Error>(data.to_vec())
        })?;
        std::fs::create_dir_all(dest.parent().unwrap_or(dest))?;
        std::fs::write(dest, bytes)?;
        Ok(())
    }
}

fn handle_key(key: KeyEvent, store: &StatsStore, args: &Args, app: &mut AppState) -> Result<bool> {
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return Ok(false);
    }
    match app.mode {
        Mode::Normal => handle_key_normal(key, store, args, app),
        Mode::Help => {
            app.mode = Mode::Normal;
            Ok(true)
        }
        Mode::ProfileView => {
            match key.code {
                KeyCode::Left | KeyCode::Char('h') => {
                    if app.profile_tab_index > 0 {
                        app.profile_tab_index -= 1;
                    }
                }
                KeyCode::Right | KeyCode::Char('l') => {
                    if app.profile_tab_index + 1 < app.profile_tabs.len() {
                        app.profile_tab_index += 1;
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => {
                    app.mode = Mode::Normal;
                }
                _ => {}
            }
            Ok(true)
        }
    }
}

fn handle_mouse(
    mouse: MouseEvent,
    store: &StatsStore,
    args: &Args,
    app: &mut AppState,
) -> Result<()> {
    if app.mode != Mode::Normal {
        return Ok(());
    }
    let pos = Position {
        x: mouse.column,
        y: mouse.row,
    };
    if matches!(mouse.kind, MouseEventKind::ScrollDown | MouseEventKind::ScrollUp) {
        let delta = if mouse.kind == MouseEventKind::ScrollDown { 1 } else { -1 };
        if app.dataset_rect.contains(pos) {
            move_dataset_selection(delta, store, app, args)?;
            return Ok(());
        }
        if app.instance_rect.contains(pos) {
            move_instance_selection(delta, store, app, args)?;
            return Ok(());
        }
        if app.runs_rect.contains(pos) {
            move_run_selection(delta, app);
            return Ok(());
        }
        return Ok(());
    }
    if mouse.kind != MouseEventKind::Down(MouseButton::Left) {
        return Ok(());
    }
    let y = mouse.row;
    if app.dataset_rect.contains(pos) {
        if let Some(idx) = list_index_from_click(
            app.dataset_rect,
            y,
            app.datasets.len(),
            app.dataset_state.offset(),
        ) {
            app.dataset_state.select(Some(idx));
            schedule_dataset_reload(app);
        }
        return Ok(());
    }
    if app.instance_rect.contains(pos) {
        if let Some(idx) = list_index_from_click(
            app.instance_rect,
            y,
            app.instances.len(),
            app.instance_state.offset(),
        ) {
            app.instance_state.select(Some(idx));
            schedule_instance_reload(app);
        }
        return Ok(());
    }
    if app.runs_rect.contains(pos) {
        if let Some(idx) =
            table_index_from_click(app.runs_rect, y, app.runs.len(), app.run_state.offset())
        {
            app.run_state.select(Some(idx));
        }
    }
    Ok(())
}

fn list_index_from_click(rect: Rect, y: u16, len: usize, offset: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let header_rows = 1u16;
    let top = rect.y.saturating_add(header_rows);
    let bottom = rect.y.saturating_add(rect.height.saturating_sub(1));
    if y < top || y >= bottom {
        return None;
    }
    let row = (y - top) as usize + offset;
    if row < len {
        Some(row)
    } else {
        None
    }
}

fn table_index_from_click(rect: Rect, y: u16, len: usize, offset: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let header_rows = 2u16;
    let top = rect.y.saturating_add(header_rows);
    let bottom = rect.y.saturating_add(rect.height.saturating_sub(1));
    if y < top || y >= bottom {
        return None;
    }
    let row = (y - top) as usize + offset;
    if row < len {
        Some(row)
    } else {
        None
    }
}
fn handle_key_normal(key: KeyEvent, store: &StatsStore, args: &Args, app: &mut AppState) -> Result<bool> {
    match key.code {
        KeyCode::Char('q') => return Ok(false),
        KeyCode::Char('p') => {
            show_profile_top(store, app, args.top_profiles)?;
            return Ok(true);
        }
        KeyCode::Char('j') => move_run_selection(1, app),
        KeyCode::Char('k') => move_run_selection(-1, app),
        KeyCode::Down => {
            move_dataset_selection(1, store, app, args)?;
        }
        KeyCode::Up => {
            move_dataset_selection(-1, store, app, args)?;
        }
        KeyCode::Char('h') | KeyCode::Left => {
            move_instance_selection(-1, store, app, args)?;
        }
        KeyCode::Char('l') | KeyCode::Right => {
            move_instance_selection(1, store, app, args)?;
        }
        KeyCode::Char('?') => {
            app.mode = Mode::Help;
        }
        _ => {}
    }
    Ok(true)
}

fn move_instance_selection(delta: isize, _store: &StatsStore, app: &mut AppState, _args: &Args) -> Result<()> {
    if app.instances.is_empty() {
        return Ok(());
    }
    let current = app
        .instance_state
        .selected()
        .unwrap_or(if delta >= 0 { 0 } else { app.instances.len().saturating_sub(1) }) as isize;
    let next = (current + delta).clamp(0, (app.instances.len() - 1) as isize) as usize;
    app.instance_state.select(Some(next));
    schedule_instance_reload(app);
    Ok(())
}

fn move_dataset_selection(delta: isize, _store: &StatsStore, app: &mut AppState, _args: &Args) -> Result<()> {
    if app.datasets.is_empty() {
        return Ok(());
    }
    let current = app
        .dataset_state
        .selected()
        .unwrap_or(if delta >= 0 { 0 } else { app.datasets.len().saturating_sub(1) }) as isize;
    let next = (current + delta).clamp(0, (app.datasets.len() - 1) as isize) as usize;
    if app.dataset_state.selected() == Some(next) {
        return Ok(());
    }
    app.dataset_state.select(Some(next));
    schedule_dataset_reload(app);
    Ok(())
}

fn move_run_selection(delta: isize, app: &mut AppState) {
    if app.runs.is_empty() {
        return;
    }
    let current = app
        .run_state
        .selected()
        .unwrap_or(if delta >= 0 { 0 } else { app.runs.len().saturating_sub(1) }) as isize;
    let next = (current + delta).clamp(0, (app.runs.len() - 1) as isize) as usize;
    app.run_state.select(Some(next));
}

fn reload_for_dataset(store: &StatsStore, app: &mut AppState, args: &Args) -> Result<()> {
    let dataset = match selected_dataset(app) {
        Some(name) => name.to_string(),
        None => {
            app.instances.clear();
            app.runs.clear();
            app.instance_state.select(None);
            app.run_state.select(None);
            *app.instance_state.offset_mut() = 0;
            *app.run_state.offset_mut() = 0;
            return Ok(());
        }
    };
    app.instances = store.list_instances(&dataset)?;
    app.instance_state.select(if app.instances.is_empty() { None } else { Some(0) });
    *app.instance_state.offset_mut() = 0;
    load_runs_for_selected(store, app, args)?;
    app.loaded_dataset_index = app.dataset_state.selected();
    app.loaded_instance_index = app.instance_state.selected();
    app.pending_instance_at = None;
    app.pending_instance_index = None;
    Ok(())
}

fn load_runs_for_selected(store: &StatsStore, app: &mut AppState, args: &Args) -> Result<()> {
    let dataset = match selected_dataset(app) {
        Some(name) => name,
        None => {
            app.runs.clear();
            app.run_state.select(None);
            *app.run_state.offset_mut() = 0;
            return Ok(());
        }
    };
    let instance = match app.instance_state.selected() {
        Some(idx) => app.instances.get(idx).map(|d| d.name.clone()),
        None => None,
    };
    let Some(instance) = instance else {
        app.runs.clear();
        app.run_state.select(None);
        *app.run_state.offset_mut() = 0;
        return Ok(());
    };
    let runs = store.list_runs_for_instance(dataset, &instance, args.max_runs)?;
    app.runs = runs
        .into_iter()
        .map(|summary| RunEntry {
            summary,
            regression: RegressionInfo,
        })
        .collect();
    app.run_state.select(if app.runs.is_empty() { None } else { Some(0) });
    *app.run_state.offset_mut() = 0;
    app.loaded_instance_index = app.instance_state.selected();
    if let Some(idx) = app.instance_state.selected() {
        if let Some(entry) = app.instances.get_mut(idx) {
            entry.latest = app.runs.first().map(|r| r.summary.clone());
        }
    }
    app.baseline = fastest_run(&app.runs).map(|r| r.summary.clone());
    recalc_regressions(app);
    Ok(())
}

fn recalc_regressions(app: &mut AppState) {
    let Some(baseline) = app.baseline.clone() else {
        return;
    };
    for entry in &mut app.runs {
        entry.regression = compare_run(&entry.summary, &baseline);
    }
}

fn compare_run(_current: &RunSummary, _baseline: &RunSummary) -> RegressionInfo {
    RegressionInfo
}

fn show_profile_top(store: &StatsStore, app: &mut AppState, top_profiles: usize) -> Result<()> {
    let run = selected_run(app)?;
    let profiles = store.list_profiles(
        &run.dataset,
        &run.run_id,
        run.instance_type.as_deref(),
        &run.source_path,
    )?;
    let pb_profiles: Vec<String> = profiles
        .into_iter()
        .filter(|name| name.ends_with(".pb"))
        .take(top_profiles)
        .collect();
    if pb_profiles.is_empty() {
        app.status = "no .pb profiles found".to_string();
        return Ok(());
    }
    let mut tabs = Vec::new();
    for name in pb_profiles {
        let cache_dir = profile_cache_dir(&run.dataset, &run.run_id)?;
        let path = cache_dir.join(&name);
        store.download_profile(
            &run.dataset,
            &run.run_id,
            run.instance_type.as_deref(),
            &run.source_path,
            &name,
            &path,
        )?;
        let mut lines = Vec::new();
        lines.push(format!("== {} ==", name));
        let output = Command::new("go")
            .args([
                "tool",
                "pprof",
                "-top",
                "-nodecount=10",
                path.to_string_lossy().as_ref(),
            ])
            .output();
        match output {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout);
                for line in text.lines() {
                    lines.push(line.to_string());
                }
            }
            _ => {
                lines.push(format!("pprof failed for {}", name));
            }
        }
        tabs.push(ProfileTab { name, lines });
    }
    app.profile_tabs = tabs;
    app.profile_tab_index = 0;
    app.mode = Mode::ProfileView;
    Ok(())
}

fn selected_run(app: &AppState) -> Result<&RunSummary> {
    let idx = app.run_state.selected().ok_or_else(|| anyhow!("no run selected"))?;
    let run = app.runs.get(idx).ok_or_else(|| anyhow!("no run selected"))?;
    Ok(&run.summary)
}


fn profile_cache_dir(dataset: &str, run_id: &str) -> Result<PathBuf> {
    let base = dirs::cache_dir().ok_or_else(|| anyhow!("cache dir not available"))?;
    Ok(base.join("diskgraph-profiles").join(dataset).join(run_id).join("profiles"))
}

fn draw(frame: &mut ratatui::Frame, app: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)].as_ref())
        .split(frame.size());
    let body = chunks[1];
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(20), Constraint::Percentage(45), Constraint::Percentage(35)])
        .split(body);
    let left_stack = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(80), Constraint::Percentage(20)].as_ref())
        .split(columns[0]);

    draw_header(frame, chunks[0], app);
    app.dataset_rect = left_stack[0];
    app.instance_rect = left_stack[1];
    app.runs_rect = columns[1];
    sync_scroll_offsets(app);
    draw_datasets(frame, left_stack[0], app);
    draw_instances(frame, left_stack[1], app);
    draw_runs(frame, columns[1], app);
    draw_details(frame, columns[2], app);
    draw_footer(frame, chunks[2], app);

    if app.mode == Mode::Help {
        draw_help(frame);
    }
    if app.mode == Mode::ProfileView {
        draw_profile_view(frame, app);
    }
}

fn draw_header(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let dataset = selected_dataset(app).unwrap_or("-");
    let text = Line::from(vec![
        Span::styled(
            format!("Diskgraph Ingest Perf Explorer — {}", dataset),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::styled(app.status.clone(), Style::default().fg(Color::Yellow)),
    ]);
    let widget = Paragraph::new(text).wrap(Wrap { trim: true });
    frame.render_widget(widget, area);
}

fn draw_footer(frame: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let hint = match app.mode {
        Mode::Normal => "j/k: run  ↑/↓: dataset  h/l: instance  p: top profiles  ?: help  q: quit",
        Mode::Help => "press any key to return",
        Mode::ProfileView => "h/l: tabs  esc/q: back",
    };
    let widget = Paragraph::new(Line::from(vec![
        Span::styled(hint, Style::default().fg(Color::Gray)),
    ]));
    frame.render_widget(widget, area);
}

fn draw_instances(frame: &mut ratatui::Frame, area: Rect, app: &mut AppState) {
    let items: Vec<ListItem> = app
        .instances
        .iter()
        .map(|d| {
            ListItem::new(d.name.clone())
        })
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Instances"))
        .highlight_style(
            Style::default()
                .bg(Color::Blue)
                .fg(Color::Indexed(0))
                .add_modifier(Modifier::BOLD),
        );
    frame.render_stateful_widget(list, area, &mut app.instance_state);
}

fn draw_datasets(frame: &mut ratatui::Frame, area: Rect, app: &mut AppState) {
    let items: Vec<ListItem> = app
        .datasets
        .iter()
        .map(|name| ListItem::new(name.clone()))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Datasets"))
        .highlight_style(
            Style::default()
                .bg(Color::Blue)
                .fg(Color::Indexed(0))
                .add_modifier(Modifier::BOLD),
        );
    frame.render_stateful_widget(list, area, &mut app.dataset_state);
}

fn draw_runs(frame: &mut ratatui::Frame, area: Rect, app: &mut AppState) {
    let header_style = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
    let rows = app.runs.iter().map(|entry| {
        let run = &entry.summary;
        let (wall_text, wall_style) = colorize_cell(
            format_duration_cell(run.wall_time_ms),
            run.wall_time_ms,
            app.baseline.as_ref().map(|b| b.wall_time_ms),
        );
        let (context_text, context_style) = colorize_cell(
            format_duration_cell(run.stage_timings_ms.context),
            run.stage_timings_ms.context,
            app.baseline.as_ref().map(|b| b.stage_timings_ms.context),
        );
        let (encode_text, encode_style) = colorize_cell(
            format_duration_cell(run.stage_timings_ms.encode),
            run.stage_timings_ms.encode,
            app.baseline.as_ref().map(|b| b.stage_timings_ms.encode),
        );
        let (pk_map_text, pk_map_style) = colorize_cell(
            format_duration_cell(run.stage_timings_ms.pk_map),
            run.stage_timings_ms.pk_map,
            app.baseline.as_ref().map(|b| b.stage_timings_ms.pk_map),
        );
        let (edges_map_text, edges_map_style) = colorize_cell(
            format_duration_cell(run.stage_timings_ms.edges_map),
            run.stage_timings_ms.edges_map,
            app.baseline.as_ref().map(|b| b.stage_timings_ms.edges_map),
        );
        let (index_text, index_style) = colorize_cell(
            format_duration_cell(run.stage_timings_ms.index),
            run.stage_timings_ms.index,
            app.baseline.as_ref().map(|b| b.stage_timings_ms.index),
        );
        let (finalize_text, finalize_style) = colorize_cell(
            format_duration_cell(run.stage_timings_ms.finalize),
            run.stage_timings_ms.finalize,
            app.baseline.as_ref().map(|b| b.stage_timings_ms.finalize),
        );
        let (peak_text, peak_style) = match run.peak_rss_anon_bytes {
            Some(bytes) => colorize_cell(
                format_bytes(bytes),
                bytes,
                app.baseline.as_ref().and_then(|b| b.peak_rss_anon_bytes),
            ),
            None => ("-".to_string(), Style::default().fg(Color::Gray)),
        };
        Row::new(vec![
            Cell::from(shorten(&date_only(&run.finished_at), 10)),
            Cell::from(run.git_sha.clone().unwrap_or_else(|| "-".to_string()).chars().take(8).collect::<String>()),
            Cell::from(wall_text).style(wall_style),
            Cell::from(context_text).style(context_style),
            Cell::from(encode_text).style(encode_style),
            Cell::from(pk_map_text).style(pk_map_style),
            Cell::from(edges_map_text).style(edges_map_style),
            Cell::from(index_text).style(index_style),
            Cell::from(finalize_text).style(finalize_style),
            Cell::from(peak_text).style(peak_style),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(8),
        ],
    )
    .header(
        Row::new(vec![
            "date",
            "git",
            "wall",
            "context",
            "encode",
            "pk_map",
            "edges_map",
            "index",
            "finalize",
            "mem",
        ])
        .style(header_style),
    )
    .block(Block::default().borders(Borders::ALL).title("Runs"))
    .highlight_style(Style::default().bg(Color::DarkGray));
    frame.render_stateful_widget(table, area, &mut app.run_state);
}

fn draw_details(frame: &mut ratatui::Frame, area: Rect, app: &mut AppState) {
    let run = app.run_state.selected().and_then(|idx| app.runs.get(idx));
    let lines = if let Some(run) = run {
        let r = &run.summary;
        let label_style = Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD);
        let value_style = Style::default().fg(Color::White);
        let stage_style = Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD);
        let peak_style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
        vec![
            Line::from(vec![Span::styled("Details", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))]),
            Line::from(vec![
                Span::styled("dataset: ", label_style),
                Span::styled(r.dataset.clone(), value_style),
            ]),
            Line::from(vec![
                Span::styled("run: ", label_style),
                Span::styled(r.run_id.clone(), value_style),
            ]),
            Line::from(vec![
                Span::styled("user: ", label_style),
                Span::styled(r.user.clone(), value_style),
            ]),
            Line::from(vec![
                Span::styled("git: ", label_style),
                Span::styled(format!(
                    "{} {}",
                    r.git_branch.clone().unwrap_or_else(|| "-".to_string()),
                    r.git_sha.clone().unwrap_or_else(|| "-".to_string())
                ), value_style),
            ]),
            Line::from(vec![
                Span::styled("instance: ", label_style),
                Span::styled(r.instance_type.clone().unwrap_or_else(|| "n/a".to_string()), value_style),
            ]),
            Line::from(vec![
                Span::styled("cpu: ", label_style),
                Span::styled(format!(
                    "{} logical, {} physical",
                    r.cpu_logical.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string()),
                    r.cpu_physical.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string())
                ), value_style),
            ]),
            Line::from(vec![
                Span::styled("shards: ", label_style),
                Span::styled(r.num_shards.map(|v| v.to_string()).unwrap_or_else(|| "-".to_string()), value_style),
            ]),
            Line::from(vec![
                Span::styled("mem: ", label_style),
                Span::styled(format_bytes_opt(r.memory_bytes_total), value_style),
            ]),
            Line::from(vec![
                Span::styled("disk: ", label_style),
                Span::styled(format_bytes_opt(r.disk_bytes_total), value_style),
            ]),
            Line::from(vec![
                Span::styled("throughput: ", label_style),
                Span::styled(format!("{}/s", format_bytes(r.throughput_bytes_per_sec)), value_style),
            ]),
            Line::from(vec![
                Span::styled("wall: ", label_style),
                Span::styled(format!("{} ms", r.wall_time_ms), value_style),
            ]),
            Line::from(vec![Span::styled("stages (ms):", stage_style)]),
            Line::from(vec![
                Span::styled("  context ", stage_style),
                Span::styled(r.stage_timings_ms.context.to_string(), value_style),
                Span::styled("  encode ", stage_style),
                Span::styled(r.stage_timings_ms.encode.to_string(), value_style),
                Span::styled("  pk_map ", stage_style),
                Span::styled(r.stage_timings_ms.pk_map.to_string(), value_style),
            ]),
            Line::from(vec![
                Span::styled("  edges_map ", stage_style),
                Span::styled(r.stage_timings_ms.edges_map.to_string(), value_style),
                Span::styled("  index ", stage_style),
                Span::styled(r.stage_timings_ms.index.to_string(), value_style),
                Span::styled("  finalize ", stage_style),
                Span::styled(r.stage_timings_ms.finalize.to_string(), value_style),
            ]),
            Line::from(vec![Span::styled("peak mem (rss anon):", peak_style)]),
            Line::from(vec![
                Span::styled("  overall ", peak_style),
                Span::styled(
                    format!(
                        "{}{}",
                        format_bytes_opt(r.peak_rss_anon_bytes),
                        mem_pct_of_total(r.peak_rss_anon_bytes, r.memory_bytes_total)
                    ),
                    value_style
                ),
            ]),
            Line::from(vec![
                Span::styled("  context ", peak_style),
                Span::styled(format_bytes_opt(r.peak_context_rss_anon_bytes), value_style),
                Span::styled("  encode ", peak_style),
                Span::styled(format_bytes_opt(r.peak_encode_rss_anon_bytes), value_style),
                Span::styled("  pk_map ", peak_style),
                Span::styled(format_bytes_opt(r.peak_pk_map_rss_anon_bytes), value_style),
            ]),
            Line::from(vec![
                Span::styled("  edges_map ", peak_style),
                Span::styled(format_bytes_opt(r.peak_edges_map_rss_anon_bytes), value_style),
                Span::styled("  index ", peak_style),
                Span::styled(format_bytes_opt(r.peak_index_rss_anon_bytes), value_style),
                Span::styled("  finalize ", peak_style),
                Span::styled(format_bytes_opt(r.peak_finalize_rss_anon_bytes), value_style),
            ]),
            Line::from(vec![
                Span::styled("profiles: ", label_style),
                Span::styled(if r.profiles_present { "yes".to_string() } else { "no".to_string() }, value_style),
            ]),
        ]
    } else {
        vec![Line::from(Span::styled("No run selected", Style::default().fg(Color::DarkGray)))]
    };
    let widget = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("Details"))
        .wrap(Wrap { trim: false });
    frame.render_widget(widget, area);
}

fn draw_help(frame: &mut ratatui::Frame) {
    let area = centered_rect(70, 60, frame.size());
    let text = vec![
        Line::from("Hotkeys:"),
        Line::from("  j/k: move runs"),
        Line::from("  ↑/↓: switch dataset"),
        Line::from("  h/l: switch instance"),
        Line::from("  p: top10 profiles"),
        Line::from("  ?: help"),
        Line::from("  q: quit"),
        Line::from("  profile view: h/l tabs, esc to exit"),
    ];
    let widget = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .wrap(Wrap { trim: true });
    frame.render_widget(widget, area);
}

fn draw_profile_view(frame: &mut ratatui::Frame, app: &AppState) {
    let area = centered_rect(90, 80, frame.size());
    frame.render_widget(Clear, area);
    let mut header = Vec::new();
    for (idx, tab) in app.profile_tabs.iter().enumerate() {
        let style = if idx == app.profile_tab_index {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        header.push(Span::styled(format!(" {} ", tab.name), style));
    }
    let mut text: Vec<Line> = Vec::new();
    if !header.is_empty() {
        text.push(Line::from(header));
        text.push(Line::from(""));
    }
    if let Some(tab) = app.profile_tabs.get(app.profile_tab_index) {
        text.extend(tab.lines.iter().map(|l| Line::from(l.as_str())));
    }
    let widget = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Profile Top 10"))
        .wrap(Wrap { trim: false });
    frame.render_widget(widget, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ]
            .as_ref(),
        )
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ]
            .as_ref(),
        )
        .split(popup_layout[1])[1]
}

fn colorize_cell(text: String, current_ms: u64, baseline_ms: Option<u64>) -> (String, Style) {
    let Some(baseline_ms) = baseline_ms else {
        return (text, Style::default().fg(Color::Gray));
    };
    if baseline_ms == 0 {
        return (text, Style::default().fg(Color::Gray));
    }
    let pct = ((current_ms as f64 - baseline_ms as f64) / baseline_ms as f64) * 100.0;
    let color = if pct <= 5.0 {
        Color::Green
    } else if pct <= 20.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    (text, Style::default().fg(color).add_modifier(Modifier::BOLD))
}

fn format_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut idx = 0;
    while value >= 1024.0 && idx < units.len() - 1 {
        value /= 1024.0;
        idx += 1;
    }
    format!("{value:.1}{}", units[idx])
}

fn format_bytes_opt(bytes: Option<u64>) -> String {
    bytes.map(format_bytes).unwrap_or_else(|| "-".to_string())
}

fn format_duration_cell(ms: u64) -> String {
    let secs = ms as f64 / 1000.0;
    format!("{secs:.1}s")
}

fn date_only(ts: &str) -> String {
    ts.split('T').next().unwrap_or(ts).to_string()
}

fn mem_pct_of_total(peak: Option<u64>, total: Option<u64>) -> String {
    match (peak, total) {
        (Some(peak), Some(total)) if total > 0 => {
            let pct = (peak as f64 / total as f64) * 100.0;
            format!(" ({pct:.1}%)")
        }
        _ => String::new(),
    }
}

fn fastest_run(runs: &[RunEntry]) -> Option<&RunEntry> {
    runs.iter().min_by_key(|r| r.summary.wall_time_ms)
}

fn sync_scroll_offsets(app: &mut AppState) {
    let dataset_rows = list_view_height(app.dataset_rect);
    sync_list_offset(&mut app.dataset_state, app.datasets.len(), dataset_rows);
    let instance_rows = list_view_height(app.instance_rect);
    sync_list_offset(&mut app.instance_state, app.instances.len(), instance_rows);
    let run_rows = table_view_height(app.runs_rect);
    sync_table_offset(&mut app.run_state, app.runs.len(), run_rows);
}

fn sync_list_offset(state: &mut ListState, len: usize, visible: usize) {
    if len == 0 || visible == 0 {
        *state.offset_mut() = 0;
        return;
    }
    let Some(selected) = state.selected() else {
        *state.offset_mut() = 0;
        return;
    };
    let max_offset = len.saturating_sub(visible);
    let mut offset = state.offset();
    if selected < offset {
        offset = selected;
    } else if selected >= offset.saturating_add(visible) {
        offset = selected + 1 - visible;
    }
    if offset > max_offset {
        offset = max_offset;
    }
    *state.offset_mut() = offset;
}

fn sync_table_offset(state: &mut TableState, len: usize, visible: usize) {
    if len == 0 || visible == 0 {
        *state.offset_mut() = 0;
        return;
    }
    let Some(selected) = state.selected() else {
        *state.offset_mut() = 0;
        return;
    };
    let max_offset = len.saturating_sub(visible);
    let mut offset = state.offset();
    if selected < offset {
        offset = selected;
    } else if selected >= offset.saturating_add(visible) {
        offset = selected + 1 - visible;
    }
    if offset > max_offset {
        offset = max_offset;
    }
    *state.offset_mut() = offset;
}

fn list_view_height(area: Rect) -> usize {
    area.height.saturating_sub(2) as usize
}

fn table_view_height(area: Rect) -> usize {
    area.height.saturating_sub(3) as usize
}

fn select_initial_dataset(datasets: &[String], requested: Option<&str>) -> Option<String> {
    if let Some(name) = requested {
        if datasets.iter().any(|d| d == name) {
            return Some(name.to_string());
        }
    }
    datasets.first().cloned()
}

fn selected_dataset(app: &AppState) -> Option<&str> {
    app.dataset_state
        .selected()
        .and_then(|idx| app.datasets.get(idx))
        .map(|s| s.as_str())
}

fn schedule_dataset_reload(app: &mut AppState) {
    app.pending_dataset_index = app.dataset_state.selected();
    app.pending_dataset_at = Some(Instant::now());
}

fn schedule_instance_reload(app: &mut AppState) {
    app.pending_instance_index = app.instance_state.selected();
    app.pending_instance_at = Some(Instant::now());
}

fn apply_debounced_dataset_change(
    store: &StatsStore,
    app: &mut AppState,
    args: &Args,
) -> Result<()> {
    let Some(pending_at) = app.pending_dataset_at else {
        return Ok(());
    };
    let pending_idx = app.pending_dataset_index;
    if pending_idx == app.loaded_dataset_index {
        app.pending_dataset_at = None;
        app.pending_dataset_index = None;
        return Ok(());
    }
    if pending_at.elapsed() < Duration::from_millis(200) {
        return Ok(());
    }
    app.pending_dataset_at = None;
    app.pending_dataset_index = None;
    reload_for_dataset(store, app, args)?;
    Ok(())
}

fn apply_debounced_instance_change(
    store: &StatsStore,
    app: &mut AppState,
    args: &Args,
) -> Result<()> {
    let Some(pending_at) = app.pending_instance_at else {
        return Ok(());
    };
    let pending_idx = app.pending_instance_index;
    if pending_idx == app.loaded_instance_index {
        app.pending_instance_at = None;
        app.pending_instance_index = None;
        return Ok(());
    }
    if pending_at.elapsed() < Duration::from_millis(150) {
        return Ok(());
    }
    app.pending_instance_at = None;
    app.pending_instance_index = None;
    load_runs_for_selected(store, app, args)?;
    Ok(())
}

fn apply_auto_refresh(store: &StatsStore, app: &mut AppState, args: &Args) -> Result<()> {
    if app.pending_dataset_at.is_some() || app.pending_instance_at.is_some() {
        return Ok(());
    }
    if app.last_refresh_at.elapsed() < Duration::from_secs(60) {
        return Ok(());
    }
    app.last_refresh_at = Instant::now();
    refresh_datasets(store, app)?;
    refresh_instances(store, app)?;
    refresh_runs_for_selected(store, app, args)?;
    Ok(())
}

fn refresh_datasets(store: &StatsStore, app: &mut AppState) -> Result<()> {
    let selected = app
        .dataset_state
        .selected()
        .and_then(|idx| app.datasets.get(idx))
        .cloned();
    let datasets = store.list_datasets()?;
    app.datasets = datasets;
    if let Some(name) = selected {
        app.dataset_state
            .select(app.datasets.iter().position(|d| d == &name));
    }
    if app.dataset_state.selected().is_none() && !app.datasets.is_empty() {
        app.dataset_state.select(Some(0));
    }
    app.loaded_dataset_index = app.dataset_state.selected();
    Ok(())
}

fn refresh_instances(store: &StatsStore, app: &mut AppState) -> Result<()> {
    let dataset = match selected_dataset(app) {
        Some(name) => name,
        None => {
            app.instances.clear();
            app.instance_state.select(None);
            *app.instance_state.offset_mut() = 0;
            return Ok(());
        }
    };
    let selected = app
        .instance_state
        .selected()
        .and_then(|idx| app.instances.get(idx))
        .map(|entry| entry.name.clone());
    app.instances = store.list_instances(dataset)?;
    if let Some(name) = selected {
        app.instance_state
            .select(app.instances.iter().position(|d| d.name == name));
    }
    if app.instance_state.selected().is_none() && !app.instances.is_empty() {
        app.instance_state.select(Some(0));
    }
    app.loaded_instance_index = app.instance_state.selected();
    Ok(())
}

fn refresh_runs_for_selected(store: &StatsStore, app: &mut AppState, args: &Args) -> Result<()> {
    let dataset = match selected_dataset(app) {
        Some(name) => name,
        None => return Ok(()),
    };
    let instance = match app.instance_state.selected() {
        Some(idx) => app.instances.get(idx).map(|d| d.name.clone()),
        None => None,
    };
    let Some(instance) = instance else {
        return Ok(());
    };
    let selected_run_id = app
        .run_state
        .selected()
        .and_then(|idx| app.runs.get(idx))
        .map(|entry| entry.summary.run_id.clone());
    let runs = store.list_runs_for_instance(dataset, &instance, args.max_runs)?;
    app.runs = runs
        .into_iter()
        .map(|summary| RunEntry {
            summary,
            regression: RegressionInfo,
        })
        .collect();
    let next_selected = if let Some(run_id) = selected_run_id {
        app.runs
            .iter()
            .position(|entry| entry.summary.run_id == run_id)
    } else {
        None
    };
    let next_selected = next_selected.or_else(|| {
        if app.runs.is_empty() {
            None
        } else {
            Some(0)
        }
    });
    app.run_state.select(next_selected);
    if let Some(idx) = app.instance_state.selected() {
        if let Some(entry) = app.instances.get_mut(idx) {
            entry.latest = app.runs.first().map(|r| r.summary.clone());
        }
    }
    app.baseline = fastest_run(&app.runs).map(|r| r.summary.clone());
    recalc_regressions(app);
    Ok(())
}
fn shorten(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}…", &text[..max.saturating_sub(1)])
    }
}

fn cloud_prefix(path: &str) -> Result<String> {
    if path.starts_with("s3://") {
        let (_bucket, prefix) = parse_s3_url(path)?;
        Ok(prefix)
    } else if path.starts_with("gs://") {
        let (_bucket, prefix) = parse_gs_url(path)?;
        Ok(prefix)
    } else if path.starts_with("az://") {
        let (_account, _container, prefix) = parse_az_url(path)?;
        Ok(prefix)
    } else {
        Err(anyhow!("unsupported cloud path {path}"))
    }
}

fn join_prefix(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.trim_start_matches('/').to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), suffix.trim_start_matches('/'))
    }
}

fn object_path(prefix: &str) -> object_store::path::Path {
    object_store::path::Path::from(prefix)
}

fn instance_dir_name(instance_type: Option<&str>, source_path: &str) -> String {
    let raw = instance_type.unwrap_or("unknown");
    let suffix = if source_path.starts_with("s3://")
        || source_path.starts_with("gs://")
        || source_path.starts_with("az://")
    {
        "remote"
    } else {
        "local"
    };
    let mut name: String = raw
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ' ' | ':' => '_',
            _ => ch,
        })
        .collect();
    name.push('-');
    name.push_str(suffix);
    name
}
