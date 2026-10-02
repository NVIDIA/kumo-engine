// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct ProgressState {
    pb: ProgressBar,
    last_stage: Option<String>,
    last_total: Option<u64>,
    last_status: Option<String>,
}

impl ProgressState {
    pub fn new() -> Self {
        let pb = ProgressBar::new(0);
        pb.set_style(spinner_style());
        pb.enable_steady_tick(Duration::from_millis(120));
        Self {
            pb,
            last_stage: None,
            last_total: None,
            last_status: None,
        }
    }

    pub fn start_stage(&mut self, message: impl Into<String>, total: Option<u64>) {
        let message = message.into();
        if self.last_stage.as_deref() != Some(message.as_str()) {
            self.pb.set_message(message.clone());
            self.last_stage = Some(message);
            self.last_total = None;
            self.last_status = None;
        }
        match total {
            Some(total) => {
                if self.last_total != Some(total) {
                    self.pb.set_style(bar_style());
                    self.pb.set_length(total);
                    self.pb.set_position(0);
                    self.last_total = Some(total);
                }
            }
            None => {
                if self.last_total.is_some() {
                    self.pb.set_style(spinner_style());
                    self.pb.set_position(0);
                    self.last_total = None;
                }
            }
        }
    }

    pub fn set_position(&mut self, position: u64) {
        if let Some(total) = self.last_total {
            self.pb.set_position(position.min(total));
            self.pb.set_prefix(format!(
                "{} / {}",
                format_count(position),
                format_count(total)
            ));
        } else {
            self.pb.tick();
        }
    }

    pub fn finish(&self, message: &str) {
        self.pb.finish_with_message(message.to_string());
    }

    pub fn set_status(&mut self, status: Option<String>) {
        if status == self.last_status {
            return;
        }
        self.last_status = status.clone();
        let Some(stage) = self.last_stage.as_ref() else {
            return;
        };
        if let Some(status) = status {
            self.pb.set_message(format!("{} {}", stage, status));
        } else {
            self.pb.set_message(stage.clone());
        }
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix} ({eta})")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

fn format_count(value: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const B: f64 = 1_000_000_000.0;
    let v = value as f64;
    if v >= B {
        format!("{:.2}B", v / B)
    } else if v >= M {
        format!("{:.2}M", v / M)
    } else if v >= K {
        format!("{:.1}K", v / K)
    } else {
        value.to_string()
    }
}
