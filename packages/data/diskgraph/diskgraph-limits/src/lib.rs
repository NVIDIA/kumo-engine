// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

const UNLIMITED: u64 = 1 << 62;

pub struct Limits {
    root: PathBuf,
}

impl Limits {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn total_memory(&self) -> Option<u64> {
        let meminfo = self.read("proc/meminfo").unwrap_or_default();
        let host = stat(&meminfo, "MemTotal:").map(|kb| kb.saturating_mul(1024));
        let mut limits = vec![host];
        for (v1, dir) in self.cgroup_levels("memory") {
            if v1 {
                let stats = fs::read_to_string(dir.join("memory.stat")).unwrap_or_default();
                limits.push(stat(&stats, "hierarchical_memory_limit"));
                limits.push(read_num(&dir.join("memory.limit_in_bytes")));
            } else {
                limits.push(read_num(&dir.join("memory.max")));
                limits.push(read_num(&dir.join("memory.high")));
            }
        }
        limits
            .into_iter()
            .flatten()
            .filter(|&l| l < UNLIMITED)
            .min()
    }

    /// Whole cores, rounded up.
    pub fn cpu_quota(&self) -> Option<usize> {
        let mut cores: Option<usize> = None;
        for (v1, dir) in self.cgroup_levels("cpu") {
            let (quota, period): (Option<i64>, Option<i64>) = if v1 {
                (
                    read_num(&dir.join("cpu.cfs_quota_us")),
                    read_num(&dir.join("cpu.cfs_period_us")),
                )
            } else {
                let max = fs::read_to_string(dir.join("cpu.max")).unwrap_or_default();
                let mut parts = max.split_whitespace();
                (
                    parts.next().and_then(|q| q.parse().ok()),
                    parts.next().and_then(|p| p.parse().ok()),
                )
            };
            if let (Some(quota @ 1..), Some(period @ 1..)) = (quota, period) {
                let c = ((quota + period - 1) / period) as usize;
                cores = Some(cores.map_or(c, |prior| prior.min(c)));
            }
        }
        cores
    }

    fn cgroup_levels(&self, controller: &str) -> Vec<(bool, PathBuf)> {
        let cgroups = self.read("proc/self/cgroup").unwrap_or_default();
        let mountinfo = self.read("proc/self/mountinfo").unwrap_or_default();
        let mut levels = Vec::new();
        for line in cgroups.lines() {
            let mut parts = line.splitn(3, ':');
            let (Some(id), Some(controllers), Some(path)) =
                (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            let v1 = id != "0";
            if v1 && !controllers.split(',').any(|c| c == controller) {
                continue;
            }
            let default_mount = if v1 {
                format!("/sys/fs/cgroup/{controller}")
            } else {
                "/sys/fs/cgroup".to_string()
            };
            let (mount, rel) = find_mount(&mountinfo, v1, controller, path).unwrap_or_else(|| {
                let rel = Path::new(path)
                    .strip_prefix("/")
                    .ok()
                    .map(Path::to_path_buf);
                (default_mount, rel)
            });
            let rel = rel
                .filter(|rel| rel.components().all(|c| matches!(c, Component::Normal(_))))
                .unwrap_or_default();
            let mount = self.root.join(mount.trim_start_matches('/'));
            let mut dir = mount.join(rel);
            loop {
                levels.push((v1, dir.clone()));
                if dir == mount || !dir.pop() {
                    break;
                }
            }
        }
        levels
    }

    fn read(&self, rel: &str) -> Option<String> {
        fs::read_to_string(self.root.join(rel)).ok()
    }
}

fn find_mount(
    mountinfo: &str,
    v1: bool,
    controller: &str,
    path: &str,
) -> Option<(String, Option<PathBuf>)> {
    let mounts: Vec<(String, Option<PathBuf>)> = mountinfo
        .lines()
        .filter_map(|line| {
            let (head, tail) = line.split_once(" - ")?;
            let head: Vec<&str> = head.split(' ').collect();
            let mut tail = tail.split(' ');
            let (fstype, _source, opts) = (tail.next()?, tail.next()?, tail.next()?);
            let matches = if v1 {
                fstype == "cgroup" && opts.split(',').any(|o| o == controller)
            } else {
                fstype == "cgroup2"
            };
            if !matches {
                return None;
            }
            let root = unescape(head.get(3)?);
            let rel = Path::new(path)
                .strip_prefix(&root)
                .ok()
                .map(Path::to_path_buf);
            Some((unescape(head.get(4)?), rel))
        })
        .collect();
    // May not be an ancestor.
    let under_root = mounts.iter().position(|(_, rel)| rel.is_some());
    mounts.into_iter().nth(under_root.unwrap_or(0))
}

/// Decodes mountinfo octal escapes.
fn unescape(field: &str) -> String {
    let mut out = String::new();
    let mut rest = field;
    while let Some(i) = rest.find('\\') {
        out.push_str(&rest[..i]);
        match u8::from_str_radix(rest.get(i + 1..i + 4).unwrap_or(""), 8) {
            Ok(byte) => {
                out.push(byte as char);
                rest = &rest[i + 4..];
            }
            Err(_) => {
                out.push('\\');
                rest = &rest[i + 1..];
            }
        }
    }
    out + rest
}

fn stat(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        (parts.next()? == key).then(|| parts.next()?.parse().ok())?
    })
}

fn read_num<T: FromStr>(path: &Path) -> Option<T> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

pub fn total_memory_bytes() -> Option<u64> {
    Limits::at("/").total_memory()
}

/// Ignores the cpu.max quota.
#[cfg(target_os = "linux")]
pub fn cpu_capacity() -> usize {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let status =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if status == 0 {
        let count = (0..libc::CPU_SETSIZE as usize)
            .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
            .count();
        if count > 0 {
            return count;
        }
    }
    let online = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if online > 0 {
        online as usize
    } else {
        1
    }
}

#[cfg(not(target_os = "linux"))]
pub fn cpu_capacity() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

pub fn effective_parallelism() -> usize {
    let host = std::thread::available_parallelism().map_or(1, |n| n.get());
    Limits::at("/")
        .cpu_quota()
        .map_or(host, |quota| host.min(quota))
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const V2_MOUNT: &str = "34 28 0:29 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw";

    fn tree(files: &[(&str, String)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (path, contents) in files {
            let path = dir.path().join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        }
        dir
    }

    fn total(files: &[(&str, String)]) -> Option<u64> {
        let mut files = files.to_vec();
        files.push((
            "proc/meminfo",
            format!("MemTotal: {} kB\n", 256 * GIB / 1024),
        ));
        Limits::at(tree(&files).path()).total_memory()
    }

    fn gib(n: u64) -> String {
        format!("{}\n", n * GIB)
    }

    #[test]
    fn v1_limit_on_ancestor_of_nested_cgroup() {
        let total = total(&[
            (
                "proc/self/cgroup",
                "5:memory:/kubepods/pod/ctr\n1:name=systemd:/x\n".into(),
            ),
            (
                "proc/self/mountinfo",
                "30 25 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n".into(),
            ),
            (
                "sys/fs/cgroup/memory/memory.limit_in_bytes",
                "9223372036854771712\n".into(),
            ),
            (
                "sys/fs/cgroup/memory/kubepods/memory.limit_in_bytes",
                "9223372036854710272\n".into(),
            ),
            (
                "sys/fs/cgroup/memory/kubepods/pod/memory.limit_in_bytes",
                gib(8),
            ),
            (
                "sys/fs/cgroup/memory/kubepods/pod/ctr/memory.limit_in_bytes",
                "9223372036854771712\n".into(),
            ),
        ]);
        assert_eq!(total, Some(8 * GIB));
    }

    #[test]
    fn v1_hierarchical_limit_covers_ancestors_outside_the_mount() {
        let total = total(&[
            ("proc/self/cgroup", "7:memory:/docker/abc\n".into()),
            (
                "proc/self/mountinfo",
                "30 25 0:27 /docker/abc /sys/fs/cgroup/memory ro - cgroup cgroup rw,memory\n"
                    .into(),
            ),
            (
                "sys/fs/cgroup/memory/memory.limit_in_bytes",
                "9223372036854771712\n".into(),
            ),
            (
                "sys/fs/cgroup/memory/memory.stat",
                format!("hierarchical_memory_limit {}\n", 16 * GIB),
            ),
        ]);
        assert_eq!(total, Some(16 * GIB));
    }

    #[test]
    fn v2_takes_tightest_max_or_high_up_the_tree() {
        let total = total(&[
            ("proc/self/cgroup", "0::/app.slice/job\n".into()),
            ("proc/self/mountinfo", format!("{V2_MOUNT}\n")),
            ("sys/fs/cgroup/app.slice/memory.max", gib(243)),
            ("sys/fs/cgroup/app.slice/memory.high", "max\n".into()),
            ("sys/fs/cgroup/app.slice/job/memory.max", "max\n".into()),
            ("sys/fs/cgroup/app.slice/job/memory.high", gib(200)),
        ]);
        assert_eq!(total, Some(200 * GIB));
    }

    #[test]
    fn hybrid_host_reads_v1_memory_controller() {
        let total = total(&[
            (
                "proc/self/cgroup",
                "0::/user.slice/job\n4:memory:/user.slice/job\n".into(),
            ),
            (
                "proc/self/mountinfo",
                "26 25 0:23 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n\
                 30 25 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
                    .into(),
            ),
            (
                "sys/fs/cgroup/unified/user.slice/job/cgroup.procs",
                "1\n".into(),
            ),
            (
                "sys/fs/cgroup/memory/user.slice/job/memory.limit_in_bytes",
                gib(4),
            ),
        ]);
        assert_eq!(total, Some(4 * GIB));
    }

    #[test]
    fn unlimited_cgroups_fall_back_to_host() {
        let total = total(&[
            ("proc/self/cgroup", "0::/a\n".into()),
            ("proc/self/mountinfo", format!("{V2_MOUNT}\n")),
            ("sys/fs/cgroup/a/memory.max", "max\n".into()),
        ]);
        assert_eq!(total, Some(256 * GIB));
    }

    #[test]
    fn path_outside_the_namespace_reads_the_mount_point() {
        let limit = ("sys/fs/cgroup/memory.max", gib(8));
        let dotdot_root = "34 28 0:29 /.. /sys/fs/cgroup rw - cgroup2 cgroup2 rw";
        for (cgroup, mountinfo) in [
            ("0::/..\n", V2_MOUNT),
            ("0::/../sibling\n", V2_MOUNT),
            ("0::/../sibling\n", dotdot_root),
            ("0::/a/b\n", ""),
        ] {
            let total = total(&[
                ("proc/self/cgroup", cgroup.into()),
                ("proc/self/mountinfo", format!("{mountinfo}\n")),
                limit.clone(),
            ]);
            assert_eq!(total, Some(8 * GIB), "{cgroup:?} {mountinfo:?}");
        }
    }

    #[test]
    fn mountinfo_escapes_and_optional_fields() {
        let total = total(&[
            ("proc/self/cgroup", "0::/my job\n".into()),
            (
                "proc/self/mountinfo",
                "34 28 0:29 / /sys/fs/cgroup\\040x rw shared:8 - cgroup2 cgroup2 rw\n".into(),
            ),
            ("sys/fs/cgroup x/my job/memory.max", gib(2)),
        ]);
        assert_eq!(total, Some(2 * GIB));
    }

    #[test]
    fn cpu_quota_takes_tightest_level_in_v1_and_v2() {
        let root = tree(&[
            ("proc/self/cgroup", "0::/a/b\n3:cpu,cpuacct:/a/b\n".into()),
            (
                "proc/self/mountinfo",
                format!("{V2_MOUNT}\n31 25 0:28 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n"),
            ),
            ("sys/fs/cgroup/a/b/cpu.max", "max 100000\n".into()),
            ("sys/fs/cgroup/a/cpu.max", "450000 100000\n".into()),
            ("sys/fs/cgroup/cpu,cpuacct/a/b/cpu.cfs_quota_us", "-1\n".into()),
            ("sys/fs/cgroup/cpu,cpuacct/a/b/cpu.cfs_period_us", "100000\n".into()),
            ("sys/fs/cgroup/cpu,cpuacct/a/cpu.cfs_quota_us", "600000\n".into()),
            ("sys/fs/cgroup/cpu,cpuacct/a/cpu.cfs_period_us", "100000\n".into()),
        ]);
        assert_eq!(Limits::at(root.path()).cpu_quota(), Some(5));
    }
}
